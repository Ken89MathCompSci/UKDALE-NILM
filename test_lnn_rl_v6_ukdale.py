"""
LNN + PPO Actor-Critic for NILM — UKDALE  (v6: targeted fine-tuning fixes).

Four fixes applied in order of impact, all active simultaneously:

  Fix 1 — Decaying supervised anchor (highest impact)
     MSE vs supervised NILM targets is kept throughout PPO and decays per
     epoch: sup_coef = max(SUP_COEF_MIN, SUP_COEF_INIT × DECAY^epoch).
     Starts strong (0.2) so PPO cannot immediately overwrite pretraining,
     decays to a floor (0.02) so the reward signal eventually dominates.
     Equivalent to behaviour-cloning regularisation that tapers off.

  Fix 2 — Two-stage encoder + critic freeze
     Stage A (first FREEZE_WARMUP epochs):
       Encoder (input_proj, tau_mod, rec_weights, gate, norm) + critic head
       frozen.  Only actor_mean + actor_log_std update.
       Motivation: critic loss was astronomically large (V=119 at epoch 1)
       and noisy critic gradients were reshaping the pretrained policy before
       the critic had any meaningful signal.
     Stage B (remaining epochs):
       All parameters unfrozen, fresh optimizer at PPO_LR.

  Fix 3 — Conservative PPO settings
     PPO_LR=1e-4, PPO_CLIP=0.1, PPO_EPOCHS=2, ENTROPY_COEF=0.01,
     grad_clip=0.25.  Each change reduces how aggressively RL updates the
     pretrained weights.  The original 3e-4 / 0.2 / 4 / 0.05 / 0.5 settings
     are appropriate for training from scratch, not for fine-tuning.

  Fix 4 — Soft-F1 reward
     Replaces hard-threshold F1 with a sigmoid-gated version:
       pred_prob = sigmoid((pred_raw − threshold) / SOFT_F1_TEMP)
     Temperature SOFT_F1_TEMP=5.0 → gradients remain meaningful within a
     ±5 W band around threshold.  Hard F1 creates discontinuous reward
     landscapes; soft F1 provides a smooth signal that RL can follow.
     FP penalty still uses hard threshold (we want actual FP suppression).

Additional stabilisers carried over from v5:
  - Separate PRETRAIN_LR (3e-4) vs PPO_LR (1e-4)
  - KL anchoring against pretrained mean (weight 0.01)
  - Rollback guard (restore pretrained checkpoint if F1 < 70% of best)
  - Tighter actor_log_std init (−2.5, std≈0.08)
  - FP_SCALE_MULT default lowered to 0.5 (logs showed FP panic → OFF collapse)

v2 reward structure (conservation guardrail, per-appliance FP scales) retained.
"""

import sys
import os
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Source Code'))
from utils import calculate_nilm_metrics


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APPLIANCES = ['dishwasher', 'fridge', 'microwave', 'washing_machine']
WIN    = 100
STRIDE = 5
BATCH  = 64
EPOCHS = 50
PATIENCE = 15

PRETRAIN_LR = 3e-4
PPO_LR      = 1e-4   # less aggressive than v2 (3e-4) for fine-tuning

GAMMA        = 0.99
GAE_LAMBDA   = 0.95
PPO_CLIP     = 0.1    # tighter clip
VALUE_COEF   = 0.5
ENTROPY_COEF = 0.01   # less exploration pressure (v2=0.05)
PPO_EPOCHS   = 2      # fewer inner epochs
ROLLOUT_SIZE = 512
GRAD_CLIP    = 0.25   # tighter grad clip (v2=0.5)

ALPHA        = 1.0
BETA_TRANS   = 0.05
NORM_POWER   = 3000.0
PRETRAIN_EPOCHS = 20

FP_SCALES = {
    'dishwasher':      1.5,
    'fridge':          1.0,
    'microwave':       0.5,
    'washing_machine': 0.75,
}
FP_SCALE_MULT = 0.5   # lowered from v2 (1.0) — FP panic was driving OFF collapse
BETA_CONSERVE = 1.0

THRESHOLDS = {a: 10.0 for a in ['dishwasher', 'fridge', 'microwave', 'washing_machine']}
THRESHOLD_ARR = np.array([THRESHOLDS[a] for a in APPLIANCES], dtype=np.float32)
FP_SCALE_ARR  = np.array([FP_SCALES[a]  for a in APPLIANCES], dtype=np.float32)

# Fix 1 — decaying supervised anchor
SUP_COEF_INIT  = 0.2   # initial supervised anchor weight
SUP_COEF_DECAY = 0.97  # per-epoch multiplicative decay
SUP_COEF_MIN   = 0.02  # floor (keeps anchor active throughout training)

# Fix 2 — two-stage freeze
FREEZE_WARMUP = 5      # Stage A duration (epochs)

# Fix 4 — soft-F1 temperature
SOFT_F1_TEMP = 5.0     # W; sigmoid band around threshold

# Carried from v5
KL_ANCHOR_COEF = 0.01  # MSE vs pretrained mean

DEFAULT_DATASET_DIR = 'dataset'


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LNNActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size, n_appliances, dt=0.1):
        super().__init__()
        self.hidden_size  = hidden_size
        self.n_appliances = n_appliances
        self.dt           = dt

        self.input_proj  = nn.Linear(input_size, hidden_size)
        self.tau_base    = nn.Parameter(torch.ones(hidden_size))
        self.tau_mod     = nn.Linear(input_size, hidden_size)
        self.rec_weights = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.rec_weights)
        self.gate        = nn.Linear(input_size + hidden_size, hidden_size)
        self.norm        = nn.LayerNorm(hidden_size)

        self.actor_mean    = nn.Linear(hidden_size, n_appliances)
        self.actor_log_std = nn.Parameter(torch.full((n_appliances,), -2.5))

        self.critic = nn.Linear(hidden_size, 1)

    def _encode(self, x):
        B, T, _ = x.size()
        h = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(T):
            x_t      = x[:, t, :]
            inp      = self.input_proj(x_t)
            rec      = torch.matmul(h, self.rec_weights)
            tau_base = F.softplus(self.tau_base).unsqueeze(0)
            tau_mod  = torch.sigmoid(self.tau_mod(x_t))
            tau      = (tau_base * tau_mod).clamp(min=self.dt)
            gate     = torch.sigmoid(self.gate(torch.cat([x_t, h], dim=1)))
            f_t      = torch.tanh(inp + rec)
            dh       = ((-h / tau) + gate * f_t) * self.dt
            h        = (h + dh).clamp(-10.0, 10.0)
        return self.norm(h)

    def get_action(self, x, deterministic=False):
        h    = self._encode(x)
        mean = torch.sigmoid(self.actor_mean(h))
        std  = self.actor_log_std.exp().clamp(0.01, 2.0)
        dist = Normal(mean, std)
        action   = mean if deterministic else dist.rsample().clamp(0.0, 1.0)
        log_prob = dist.log_prob(action.clamp(1e-6, 1 - 1e-6)).sum(-1)
        entropy  = dist.entropy().sum(-1)
        value    = self.critic(h).squeeze(-1)
        return action, log_prob, entropy, value

    def evaluate_actions(self, x, actions):
        h    = self._encode(x)
        mean = torch.sigmoid(self.actor_mean(h))
        std  = self.actor_log_std.exp().clamp(0.01, 2.0)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions.clamp(1e-6, 1 - 1e-6)).sum(-1)
        entropy  = dist.entropy().sum(-1)
        value    = self.critic(h).squeeze(-1)
        return log_prob, entropy, value, mean


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data(dataset_dir=DEFAULT_DATASET_DIR):
    print(f"Loading CSVs from '{dataset_dir}/'...")
    splits = {}
    for split in ('train', 'validation', 'test'):
        path = os.path.join(dataset_dir, f'UKDALE_HF_{split}.csv')
        df   = pd.read_csv(path, index_col='timestamp', parse_dates=True)
        splits[split] = df
        print(f"  {split:12s}: {df.shape}  "
              f"{df.index[0].date()} → {df.index[-1].date()}")
    return splits


def create_sequences(df):
    mains    = df['aggregate'].values.astype(np.float32)
    app_vals = {a: df[a].values.astype(np.float32) for a in APPLIANCES}
    X, Y, Agg = [], [], []
    for i in range(0, len(mains) - WIN + 1, STRIDE):
        X.append(mains[i: i + WIN])
        mid = i + WIN // 2
        Y.append([app_vals[a][mid] for a in APPLIANCES])
        Agg.append(mains[mid])
    return (
        np.array(X, dtype=np.float32).reshape(-1, WIN, 1),
        np.array(Y, dtype=np.float32),
        np.array(Agg, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Reward  (Fix 4: soft-F1 + lower FP, conservation guardrail unchanged)
# ---------------------------------------------------------------------------

def compute_rewards_batch(actions, y_true, prev_actions, y_mins, y_ranges, agg_watts):
    pred_raw = actions      * y_ranges + y_mins
    true_raw = y_true       * y_ranges + y_mins
    prev_raw = prev_actions * y_ranges + y_mins

    N = len(actions)

    mae_term   = np.abs(pred_raw - true_raw).mean(axis=1)
    trans_term = np.abs(pred_raw - prev_raw).mean(axis=1)

    # ── Fix 4: Soft-F1 primary (sigmoid-gated, differentiable) ───────────
    # pred_prob in [0,1] — smooth version of pred_on indicator.
    # Gradient is meaningful within ±SOFT_F1_TEMP W of threshold.
    pred_prob = 1.0 / (1.0 + np.exp(-((pred_raw - THRESHOLD_ARR) / SOFT_F1_TEMP)))
    true_prob = (true_raw > THRESHOLD_ARR).astype(np.float32)  # hard label
    soft_tp = pred_prob * true_prob
    soft_fp = pred_prob * (1.0 - true_prob)
    soft_fn = (1.0 - pred_prob) * true_prob
    soft_f1 = 2.0 * soft_tp / (2.0 * soft_tp + soft_fp + soft_fn + 1e-8)
    f1_reward = soft_f1.mean(axis=1)

    # ── FP penalty — hard threshold for actual suppression ────────────────
    pred_on_hard = pred_raw > THRESHOLD_ARR
    true_on_hard = true_raw > THRESHOLD_ARR
    fp_hard = (pred_on_hard & ~true_on_hard).astype(np.float32)
    fp_penalty = np.zeros(N, dtype=np.float32)
    for i in range(len(APPLIANCES)):
        fp_penalty += fp_hard[:, i] * (FP_SCALE_MULT * FP_SCALE_ARR[i] * THRESHOLD_ARR[i])

    # ── Conservation guardrail ─────────────────────────────────────────────
    conservation = np.maximum(0.0, pred_raw.sum(axis=1) - agg_watts * 1.1)

    reward = (f1_reward
              - (ALPHA * mae_term
                 + BETA_TRANS * trans_term
                 + fp_penalty
                 + BETA_CONSERVE * conservation) / NORM_POWER)
    return reward


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def compute_gae(rewards, values, last_value):
    adv = []
    gae = 0.0
    vals_ext = values + [last_value]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * vals_ext[t + 1] - vals_ext[t]
        gae   = delta + GAMMA * GAE_LAMBDA * gae
        adv.insert(0, gae)
    returns = [a + v for a, v in zip(adv, values)]
    return adv, returns


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X, Y_scaled, y_scalers, device):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            xb = torch.FloatTensor(X[i: i + BATCH]).to(device)
            action, _, _, _ = model.get_action(xb, deterministic=True)
            all_pred.append(action.cpu().numpy())
            all_true.append(Y_scaled[i: i + BATCH])
    pred_sc = np.concatenate(all_pred)
    true_sc = np.concatenate(all_true)
    results = {}
    for i, app in enumerate(APPLIANCES):
        raw_pred = y_scalers[i].inverse_transform(pred_sc[:, i:i+1]).flatten()
        raw_true = y_scalers[i].inverse_transform(true_sc[:, i:i+1]).flatten()
        thr      = THRESHOLDS[app]
        m        = calculate_nilm_metrics(raw_true, raw_pred, threshold=thr)
        m['TP']  = int(((raw_true > thr) &  (raw_pred > thr)).sum())
        m['FP']  = int(((raw_true <= thr) & (raw_pred > thr)).sum())
        m['FN']  = int(((raw_true > thr) &  (raw_pred <= thr)).sum())
        results[app] = m
    return results


# ---------------------------------------------------------------------------
# Supervised pre-training
# ---------------------------------------------------------------------------

def pretrain_supervised(model, X_tr, Y_tr, device, epochs=PRETRAIN_EPOCHS, lr=PRETRAIN_LR):
    if epochs <= 0:
        return
    print(f"\n--- Supervised pre-training ({epochs} epochs, lr={lr}) ---")
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = len(X_tr)
    for ep in range(epochs):
        perm = np.random.permutation(N)
        total_loss = 0.0; n_batches = 0
        for start in range(0, N, BATCH):
            idx  = perm[start: start + BATCH]
            xb   = torch.FloatTensor(X_tr[idx]).to(device)
            yb   = torch.FloatTensor(Y_tr[idx]).to(device)
            h    = model._encode(xb)
            mean = torch.sigmoid(model.actor_mean(h))
            loss = F.mse_loss(mean, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            total_loss += loss.item(); n_batches += 1
        print(f"  pretrain epoch {ep+1}/{epochs}  MSE={total_loss/n_batches:.6f}")
    print("--- Pre-training complete ---\n")


# ---------------------------------------------------------------------------
# Freeze / unfreeze helpers
# ---------------------------------------------------------------------------

def _freeze_encoder(model):
    for p in ([*model.input_proj.parameters(), model.tau_base,
                *model.tau_mod.parameters(), model.rec_weights,
                *model.gate.parameters(), *model.norm.parameters()]):
        p.requires_grad = False


def _freeze_critic(model):
    for p in model.critic.parameters():
        p.requires_grad = False


def _unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def _make_optimizer(model, lr):
    return torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, eps=1e-5)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(data_splits, save_dir, hidden_size=64, dt=0.1):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nDevice: {device}  hidden={hidden_size}  dt={dt}")
    print(f"Fix1 sup_coef: {SUP_COEF_INIT}→{SUP_COEF_MIN} (decay={SUP_COEF_DECAY}/epoch)")
    print(f"Fix2 freeze: encoder+critic frozen for {FREEZE_WARMUP} epochs (Stage A)")
    print(f"Fix3 PPO: LR={PPO_LR}  clip={PPO_CLIP}  epochs={PPO_EPOCHS}  "
          f"ent={ENTROPY_COEF}  grad_clip={GRAD_CLIP}")
    print(f"Fix4 soft-F1 temp={SOFT_F1_TEMP}W  FP_mult={FP_SCALE_MULT}")
    fp_eff = {a: round(FP_SCALE_MULT * FP_SCALES[a] * THRESHOLDS[a], 2) for a in APPLIANCES}
    print(f"FP flat cost/app (W): " + "  ".join(f"{a}={fp_eff[a]}" for a in APPLIANCES))

    tr_df = data_splits['train']
    va_df = data_splits['validation']
    te_df = data_splits['test']

    X_tr, Y_tr, Agg_tr = create_sequences(tr_df)
    X_va, Y_va, _       = create_sequences(va_df)
    X_te, Y_te, _       = create_sequences(te_df)

    x_scaler = MinMaxScaler()
    X_tr = x_scaler.fit_transform(X_tr.reshape(-1, 1)).reshape(X_tr.shape)
    X_va = x_scaler.transform(X_va.reshape(-1, 1)).reshape(X_va.shape)
    X_te = x_scaler.transform(X_te.reshape(-1, 1)).reshape(X_te.shape)

    y_scalers = []
    for i in range(len(APPLIANCES)):
        ys = MinMaxScaler()
        Y_tr[:, i:i+1] = ys.fit_transform(Y_tr[:, i:i+1])
        Y_va[:, i:i+1] = ys.transform(Y_va[:, i:i+1])
        Y_te[:, i:i+1] = ys.transform(Y_te[:, i:i+1])
        y_scalers.append(ys)

    y_mins   = np.array([float(ys.data_min_[0])   for ys in y_scalers])
    y_ranges = np.array([float(ys.data_range_[0]) for ys in y_scalers])

    print(f"\nTrain: {X_tr.shape}  Val: {X_va.shape}  Test: {X_te.shape}")

    model = LNNActorCritic(1, hidden_size, len(APPLIANCES), dt).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ── Phase 1: Supervised pretrain ──────────────────────────────────────
    pretrain_supervised(model, X_tr, Y_tr, device,
                        epochs=PRETRAIN_EPOCHS, lr=PRETRAIN_LR)

    pre_metrics = evaluate(model, X_va, Y_va, y_scalers, device)
    pre_f1 = np.mean([pre_metrics[a]['f1'] for a in APPLIANCES])
    print(f"Post-pretrain val avgF1: {pre_f1:.4f}")

    pretrained_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Frozen reference model for KL anchoring
    ref_model = LNNActorCritic(1, hidden_size, len(APPLIANCES), dt).to(device)
    ref_model.load_state_dict(pretrained_state)
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    # ── Phase 2: PPO fine-tuning ───────────────────────────────────────────
    # Stage A: freeze encoder + critic — train actor only
    _freeze_encoder(model)
    _freeze_critic(model)
    optimizer = _make_optimizer(model, PPO_LR)
    active_a  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_a  = n_params - active_a
    print(f"Stage A: {frozen_a:,} frozen  {active_a:,} active (actor only).")
    print(f"Starting LNN-PPO-v6 training...\n")

    train_start = time.time()
    history = {
        'policy_loss': [], 'value_loss': [], 'entropy': [],
        'mean_reward': [], 'bc_loss': [], 'kl_loss': [],
        'sup_coef': [], 'val_metrics': [],
    }
    best_f1    = pre_f1
    best_state = pretrained_state
    counter    = 0
    stage      = 'A'
    rolled_back = False

    for epoch in range(EPOCHS):
        ep_start = time.time()

        # ── Transition to Stage B after warmup ────────────────────────────
        if stage == 'A' and epoch >= FREEZE_WARMUP:
            _unfreeze_all(model)
            optimizer = _make_optimizer(model, PPO_LR)
            stage = 'B'
            print(f"  [Epoch {epoch+1}] → Stage B: all {n_params:,} params unfrozen.")

        # Fix 1: decaying supervised anchor coefficient
        sup_coef = max(SUP_COEF_MIN, SUP_COEF_INIT * (SUP_COEF_DECAY ** epoch))

        model.train()
        ep_pi = ep_v = ep_ent = ep_reward = ep_bc = ep_kl = 0.0
        n_updates = 0

        prev_action_np = np.zeros(len(APPLIANCES), dtype=np.float32)

        bar = tqdm(range(0, len(X_tr), ROLLOUT_SIZE),
                   desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        for rollout_start in bar:
            rollout_end = min(rollout_start + ROLLOUT_SIZE, len(X_tr))
            N = rollout_end - rollout_start

            x_chunk   = torch.FloatTensor(X_tr[rollout_start:rollout_end]).to(device)
            y_chunk   = Y_tr[rollout_start:rollout_end]
            y_chunk_t = torch.FloatTensor(y_chunk).to(device)
            agg_chunk = Agg_tr[rollout_start:rollout_end]

            # ── Collect rollout ────────────────────────────────────────────
            model.eval()
            with torch.no_grad():
                actions_t, logprobs_t, _, values_t = model.get_action(x_chunk)

            actions_np  = actions_t.cpu().numpy()
            logprobs_np = logprobs_t.cpu().numpy()
            values_np   = values_t.cpu().numpy().tolist()

            prev_np = np.vstack([prev_action_np[None, :], actions_np[:-1]])

            rewards_np = compute_rewards_batch(
                actions_np, y_chunk, prev_np, y_mins, y_ranges, agg_chunk)

            prev_action_np = actions_np[-1]

            with torch.no_grad():
                _, _, _, last_val_t = model.get_action(
                    x_chunk[-1:], deterministic=True)
            last_value = last_val_t[0].item()

            # ── GAE ────────────────────────────────────────────────────────
            advantages_list, returns_list = compute_gae(
                rewards_np.tolist(), values_np, last_value)

            advantages_t = torch.tensor(advantages_list, dtype=torch.float32, device=device)
            returns_t    = torch.tensor(returns_list,    dtype=torch.float32, device=device)
            old_lp_t     = torch.tensor(logprobs_np,     dtype=torch.float32, device=device)
            old_acts_t   = actions_t

            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
            ep_reward += float(rewards_np.mean())

            # ── PPO update ─────────────────────────────────────────────────
            model.train()
            for _ in range(PPO_EPOCHS):
                perm = torch.randperm(N, device=device)
                for start in range(0, N, BATCH):
                    idx = perm[start: start + BATCH]
                    xb  = x_chunk[idx]
                    ab  = old_acts_t[idx]
                    adv = advantages_t[idx]
                    ret = returns_t[idx]
                    olp = old_lp_t[idx]

                    new_lp, ent, vals, mean = model.evaluate_actions(xb, ab)

                    ratio   = (new_lp - olp).exp()
                    surr1   = ratio * adv
                    surr2   = ratio.clamp(1 - PPO_CLIP, 1 + PPO_CLIP) * adv
                    pi_loss = -torch.min(surr1, surr2).mean()

                    # In Stage A critic is frozen so v_loss has no gradient;
                    # keep it in for logging continuity
                    v_loss  = VALUE_COEF * (ret - vals).pow(2).mean()
                    e_loss  = -ENTROPY_COEF * ent.mean()

                    # Fix 1: decaying supervised anchor
                    sup_loss = F.mse_loss(mean, y_chunk_t[idx])

                    # KL anchoring vs pretrained mean
                    with torch.no_grad():
                        ref_h    = ref_model._encode(xb)
                        ref_mean = torch.sigmoid(ref_model.actor_mean(ref_h))
                    kl_loss = F.mse_loss(mean, ref_mean)

                    loss = (pi_loss + v_loss + e_loss
                            + sup_coef * sup_loss
                            + KL_ANCHOR_COEF * kl_loss)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()

                    ep_pi  += pi_loss.item()
                    ep_v   += v_loss.item()
                    ep_ent += ent.mean().item()
                    ep_bc  += sup_loss.item()
                    ep_kl  += kl_loss.item()
                    n_updates += 1

        if n_updates:
            ep_pi  /= n_updates; ep_v   /= n_updates
            ep_ent /= n_updates; ep_bc  /= n_updates; ep_kl /= n_updates

        rollouts_per_epoch = max(1, len(X_tr) // ROLLOUT_SIZE)
        ep_reward /= rollouts_per_epoch

        val_metrics = evaluate(model, X_va, Y_va, y_scalers, device)
        history['val_metrics'].append(val_metrics)
        history['policy_loss'].append(ep_pi)
        history['value_loss'].append(ep_v)
        history['entropy'].append(ep_ent)
        history['mean_reward'].append(ep_reward)
        history['bc_loss'].append(ep_bc)
        history['kl_loss'].append(ep_kl)
        history['sup_coef'].append(sup_coef)

        avg_f1  = np.mean([val_metrics[a]['f1']  for a in APPLIANCES])
        avg_mae = np.mean([val_metrics[a]['mae'] for a in APPLIANCES])

        ep_time = time.time() - ep_start
        print(
            f"  Epoch {epoch+1:3d}/{EPOCHS}  "
            f"π={ep_pi:.5f}  V={ep_v:.5f}  H={ep_ent:.3f}  R={ep_reward:.5f}  "
            f"sup={sup_coef:.4f}  BC={ep_bc:.5f}  KL={ep_kl:.5f}  "
            f"avgF1={avg_f1:.4f}  avgMAE={avg_mae:.2f}  [{stage}]  time={ep_time:.1f}s"
        )
        for app in APPLIANCES:
            m = val_metrics[app]
            print(f"    {app:<22s}  F1={m['f1']:.4f}  "
                  f"P={m['precision']:.4f}  R={m['recall']:.4f}  "
                  f"MAE={m['mae']:.2f}  SAE={m['sae']:.4f}  "
                  f"TP={m['TP']}  FP={m['FP']}  FN={m['FN']}")

        # ── Rollback guard ─────────────────────────────────────────────────
        if not rolled_back and best_f1 > 0.05 and avg_f1 < best_f1 * 0.7:
            print(f"  *** Rollback: F1 {avg_f1:.4f} < 70% of best {best_f1:.4f}. "
                  f"Restoring pretrained checkpoint. ***")
            model.load_state_dict(pretrained_state)
            rolled_back = True
            counter = 0
            continue

        if avg_f1 > best_f1:
            best_f1    = avg_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            counter    = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    total_time = time.time() - train_start
    print(f"\nTraining complete.  Total time: {total_time/60:.1f} min ({total_time:.0f}s)")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, X_te, Y_te, y_scalers, device)

    print(f"\n{'Appliance':<22} {'F1':>6} {'Prec':>6} {'Rec':>6} "
          f"{'MAE':>7} {'SAE':>7} {'TP':>7} {'FP':>7} {'FN':>7}")
    print("-" * 80)
    for app in APPLIANCES:
        m = test_metrics[app]
        print(f"{app:<22} {m['f1']:>6.4f} {m['precision']:>6.4f} {m['recall']:>6.4f} "
              f"{m['mae']:>7.2f} {m['sae']:>7.4f} {m['TP']:>7} {m['FP']:>7} {m['FN']:>7}")

    _plot_results(history, test_metrics, save_dir)
    _save_json(test_metrics, hidden_size, dt, save_dir)
    return test_metrics, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_results(history, test_metrics, save_dir):
    epochs_x = range(1, len(history['policy_loss']) + 1)

    plt.figure(figsize=(24, 4))
    panels = [
        ('policy_loss', 'Policy Loss (π)', 'blue'),
        ('value_loss',  'Value Loss (V)',   'red'),
        ('entropy',     'Entropy (H)',      'green'),
        ('mean_reward', 'Mean Reward',      'purple'),
        ('bc_loss',     'Supervised (BC)',  'darkorange'),
        ('kl_loss',     'KL Anchor',        'teal'),
        ('sup_coef',    'sup_coef decay',   'brown'),
    ]
    for i, (key, title, color) in enumerate(panels, 1):
        plt.subplot(1, 7, i)
        plt.plot(epochs_x, history[key], color=color)
        plt.title(title); plt.xlabel('Epoch'); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rl_training_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Mark Stage A / Stage B boundary
    fig, axes = plt.subplots(len(APPLIANCES), 2, figsize=(12, 4 * len(APPLIANCES)))
    fig.suptitle('LNN-PPO-v6 UKDALE — Per-Appliance Val Metrics', fontsize=13)
    for row, app in enumerate(APPLIANCES):
        f1s  = [m[app]['f1']  for m in history['val_metrics']]
        maes = [m[app]['mae'] for m in history['val_metrics']]
        for ax, vals, label, color in [
            (axes[row][0], f1s,  'F1',      'steelblue'),
            (axes[row][1], maes, 'MAE (W)', 'red'),
        ]:
            ax.plot(epochs_x, vals, color=color)
            ax.axvline(FREEZE_WARMUP + 0.5, color='grey', linestyle=':', alpha=0.7,
                       label='Stage A→B')
            ref = test_metrics[app]['f1'] if label == 'F1' else test_metrics[app]['mae']
            ax.axhline(ref, color='green', linestyle='--', label=f'Test {label}')
            ax.set_title(f'{app} — {label}')
            ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_appliance_metrics.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------

def _save_json(test_metrics, hidden_size, dt, save_dir):
    config = {
        'model': 'LNNActorCritic PPO-v6 (4 targeted fine-tuning fixes)',
        'dataset': 'UKDALE-dataset',
        'hyperparams': {
            'WIN': WIN, 'STRIDE': STRIDE, 'BATCH': BATCH,
            'EPOCHS': EPOCHS, 'PATIENCE': PATIENCE,
            'PRETRAIN_LR': PRETRAIN_LR, 'PPO_LR': PPO_LR,
            'hidden_size': hidden_size, 'dt': dt,
            'GAMMA': GAMMA, 'GAE_LAMBDA': GAE_LAMBDA, 'PPO_CLIP': PPO_CLIP,
            'VALUE_COEF': VALUE_COEF, 'ENTROPY_COEF': ENTROPY_COEF,
            'PPO_EPOCHS': PPO_EPOCHS, 'ROLLOUT_SIZE': ROLLOUT_SIZE,
            'GRAD_CLIP': GRAD_CLIP,
            'ALPHA': ALPHA, 'BETA_TRANS': BETA_TRANS, 'NORM_POWER': NORM_POWER,
            'FP_SCALES': FP_SCALES, 'FP_SCALE_MULT': FP_SCALE_MULT,
            'BETA_CONSERVE': BETA_CONSERVE,
            'SUP_COEF_INIT': SUP_COEF_INIT, 'SUP_COEF_DECAY': SUP_COEF_DECAY,
            'SUP_COEF_MIN': SUP_COEF_MIN,
            'FREEZE_WARMUP': FREEZE_WARMUP,
            'SOFT_F1_TEMP': SOFT_F1_TEMP,
            'KL_ANCHOR_COEF': KL_ANCHOR_COEF,
            'actor_log_std_init': -2.5,
        },
        'test_metrics': {
            app: {
                k: (int(v) if isinstance(v, (np.integer, int)) else float(v))
                for k, v in m.items()
            }
            for app, m in test_metrics.items()
        },
    }
    out = os.path.join(save_dir, 'results.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)
    print(f"Results saved → {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='LNN-PPO v6 NILM on UKDALE — 4 targeted fine-tuning fixes')
    p.add_argument('--dataset-dir',    default=DEFAULT_DATASET_DIR)
    p.add_argument('--hidden-size',    type=int,   default=64)
    p.add_argument('--dt',             type=float, default=0.1)
    p.add_argument('--pretrain-lr',    type=float, default=PRETRAIN_LR)
    p.add_argument('--ppo-lr',         type=float, default=PPO_LR)
    p.add_argument('--freeze-warmup',  type=int,   default=FREEZE_WARMUP,
                   help='Stage A duration in PPO epochs (default 5)')
    p.add_argument('--sup-coef-init',  type=float, default=SUP_COEF_INIT,
                   help='Initial supervised anchor coef (default 0.2)')
    p.add_argument('--sup-decay',      type=float, default=SUP_COEF_DECAY,
                   help='Per-epoch decay of sup_coef (default 0.97)')
    p.add_argument('--soft-f1-temp',   type=float, default=SOFT_F1_TEMP,
                   help='Soft-F1 sigmoid temperature in Watts (default 5.0)')
    p.add_argument('--fp-scale',       type=float, default=FP_SCALE_MULT,
                   help='Global FP penalty multiplier (default 0.5)')
    p.add_argument('--kl-coef',        type=float, default=KL_ANCHOR_COEF)
    p.add_argument('--beta-conserve',  type=float, default=BETA_CONSERVE)
    p.add_argument('--pretrain',       type=int,   default=PRETRAIN_EPOCHS)
    return p.parse_args()


if __name__ == '__main__':
    script_start = time.time()
    args = parse_args()

    for split in ('train', 'validation', 'test'):
        path = os.path.join(args.dataset_dir, f'UKDALE_HF_{split}.csv')
        if not os.path.exists(path):
            print(f"Error: {path} not found.  Run preprocess_hf.py first.")
            sys.exit(1)

    PRETRAIN_LR    = args.pretrain_lr
    PPO_LR         = args.ppo_lr
    FREEZE_WARMUP  = args.freeze_warmup
    SUP_COEF_INIT  = args.sup_coef_init
    SUP_COEF_DECAY = args.sup_decay
    SOFT_F1_TEMP   = args.soft_f1_temp
    FP_SCALE_MULT  = args.fp_scale
    KL_ANCHOR_COEF = args.kl_coef
    BETA_CONSERVE  = args.beta_conserve
    PRETRAIN_EPOCHS = args.pretrain

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir  = f'models/lnn_rl_v6_{timestamp}'

    data_splits = load_data(args.dataset_dir)
    train_model(data_splits, save_dir=save_dir,
                hidden_size=args.hidden_size, dt=args.dt)

    total_time = time.time() - script_start
    print(f"\nAll outputs saved to {save_dir}/")
    print(f"Total wall-clock time: {total_time/60:.1f} min ({total_time:.0f}s)")
