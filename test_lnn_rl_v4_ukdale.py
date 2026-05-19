"""
LNN + PPO Actor-Critic for NILM — UKDALE  (v4: PPO stabilisation).

Implements nine targeted fixes for the always-OFF entropy-collapse observed in
earlier versions.  Changes are listed in priority order:

  P1 — Behaviour Cloning (BC) loss during PPO
       bc_loss = Huber(actor_mean(h), scaled_target)  weight BC_COEF=0.3
       Prevents catastrophic forgetting of the supervised solution.  Without
       this, PPO freely destroys the regressor built by pre-training.

  P2 — Binary ON/OFF BCE auxiliary loss during PPO
       bce_loss = BCE(actor_mean(h), onoff_target)  weight BCE_COEF=0.2
       Aligns gradient with the F1 evaluation metric.  BCE directly
       penalises wrong binary state, whereas MAE ignores the 10W threshold.

  P3 — Reduced transition penalty
       BETA_TRANS: 0.05 → 0.005
       The original value penalised rapid state changes strongly enough to
       make the always-OFF policy attractive (zero transitions = zero penalty).

  P4 — Fixed reward scale
       NORM_POWER: 3000 → 300
       Original rewards were in [-0.08, 0], giving extremely weak PPO
       gradients.  Rescaling to [-0.8, 0] restores meaningful gradient signal.

  P5 — Sigmoid reparameterisation for action sampling
       Replaces  action = dist.rsample().clamp(0, 1)
       with      action = sigmoid(dist.rsample())
       Hard clamping zeros the gradient whenever the raw sample is outside
       [0,1].  Sigmoid maps the full real line to (0,1) with a well-defined
       gradient everywhere.  Log-probs are computed in the raw (pre-sigmoid)
       space; evaluate_actions inverts with logit() for consistency.

  P6 — Higher entropy coefficient + tighter std clamp
       ENTROPY_COEF: 0.01 → 0.05
       std clamp: (0.01, 2.0) → (0.05, 1.0)
       Keeps the policy exploring longer and prevents the degenerate Dirac
       delta at 0W that collapsed earlier runs.

  P7 — Huber loss (smooth_l1) everywhere
       Replaces MSE in both pre-training and BC regularisation.  Power
       signals contain large spikes; Huber is robust to these outliers and
       avoids the very large gradients that destabilise early training.

  P8 — Explicit ON-state reward
       on_reward = (pred_on * true_on).mean(axis=1)  weight ON_REWARD_COEF
       Adds a positive signal for correct detections, countering the bias of
       purely negative reward functions that make doing nothing attractive.

  P9 — Larger default hidden state
       hidden_size: 64 → 128
       With only 9k parameters the LNN was likely underpowered for capturing
       fridge cycles and washing-machine stage transitions.

Architecture : shared LNN (LiquidCell) encoder → Gaussian actor + critic V(s)
Training     : supervised Huber pre-training → PPO with BC + BCE auxiliary losses
Dataset      : medium_dataset/ (or --dataset-dir override)
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
WIN    = 100   # --win to override (P10: try 200 or 300 for longer context)
STRIDE = 5
BATCH  = 64
EPOCHS = 50
PATIENCE = 15
LR     = 3e-4

# PPO hyperparameters
GAMMA        = 0.99
GAE_LAMBDA   = 0.95
PPO_CLIP     = 0.2
VALUE_COEF   = 0.5
ENTROPY_COEF = 0.05   # P6: was 0.01
PPO_EPOCHS   = 4
ROLLOUT_SIZE = 512

# Reward shaping
ALPHA          = 1.0
BETA_TRANS     = 0.005   # P3: was 0.05 — strong transition penalty encouraged always-OFF
NORM_POWER     = 300.0   # P4: was 3000 — rewards were too small for useful PPO gradients
ON_REWARD_COEF = 0.2     # P8: positive bonus per correct ON detection
PRETRAIN_EPOCHS = 20

# Auxiliary losses during PPO  (P1, P2)
BC_COEF  = 0.3   # behaviour-cloning regression weight
BCE_COEF = 0.2   # binary ON/OFF cross-entropy weight

THRESHOLDS = {
    'dishwasher':      10.0,
    'fridge':          10.0,
    'microwave':       10.0,
    'washing_machine': 10.0,
}
THRESHOLD_ARR = np.array([THRESHOLDS[a] for a in APPLIANCES], dtype=np.float32)

DEFAULT_DATASET_DIR = 'dataset'


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LNNActorCritic(nn.Module):
    """
    Shared LNN encoder → Gaussian actor (P5: sigmoid reparameterisation) + critic.

    Key change from v1: the Normal distribution lives in the *raw* (unconstrained)
    space.  Stochastic actions are obtained via sigmoid(rsample()), which has
    a well-defined gradient everywhere unlike rsample().clamp(0,1).
    evaluate_actions inverts sigmoid with logit() to recover consistent log-probs.
    """

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

        self.actor_mean    = nn.Linear(hidden_size, n_appliances)   # outputs raw (unconstrained)
        self.actor_log_std = nn.Parameter(torch.full((n_appliances,), -1.0))

        self.critic = nn.Linear(hidden_size, 1)

    def _encode(self, x):
        """LiquidCell: (B, WIN, 1) → (B, hidden)"""
        B, T, _ = x.size()
        h = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(T):
            x_t        = x[:, t, :]
            input_proj = self.input_proj(x_t)
            rec_proj   = torch.matmul(h, self.rec_weights)
            tau_base   = F.softplus(self.tau_base).unsqueeze(0)
            tau_mod    = torch.sigmoid(self.tau_mod(x_t))
            tau        = (tau_base * tau_mod).clamp(min=self.dt)
            gate       = torch.sigmoid(self.gate(torch.cat([x_t, h], dim=1)))
            f_t        = torch.tanh(input_proj + rec_proj)
            dh         = ((-h / tau) + gate * f_t) * self.dt
            h          = (h + dh).clamp(-10.0, 10.0)
        return self.norm(h)

    def get_action(self, x, deterministic=False):
        """
        P5: sample action via sigmoid(rsample()) instead of rsample().clamp(0,1).

        Distribution is Normal(raw_mean, std) in unconstrained space.
        Log-probs are computed on raw values to keep rollout/update consistent.
        """
        h        = self._encode(x)
        raw_mean = self.actor_mean(h)               # unconstrained output
        mean     = torch.sigmoid(raw_mean)          # [0,1] deterministic action
        std      = self.actor_log_std.exp().clamp(0.05, 1.0)   # P6: tighter clamp
        dist     = Normal(raw_mean, std)

        if deterministic:
            action   = mean
            raw_used = raw_mean
        else:
            raw_used = dist.rsample()
            action   = torch.sigmoid(raw_used)      # gradient flows through sigmoid

        log_prob = dist.log_prob(raw_used).sum(-1)
        entropy  = dist.entropy().sum(-1)
        value    = self.critic(h).squeeze(-1)
        return action, log_prob, entropy, value

    def evaluate_actions(self, x, actions):
        """
        Recompute log_prob / entropy / value / mean for stored (s, a) pairs.

        actions ∈ [0,1] (sigmoid-space); invert with logit to recover raw values
        for consistent log_prob under Normal(raw_mean, std).
        Returns mean as well — used by the BC and BCE auxiliary losses (P1, P2)
        without a second _encode call.
        """
        h        = self._encode(x)
        raw_mean = self.actor_mean(h)
        mean     = torch.sigmoid(raw_mean)
        std      = self.actor_log_std.exp().clamp(0.05, 1.0)
        dist     = Normal(raw_mean, std)

        raw_actions = torch.logit(actions.clamp(1e-6, 1 - 1e-6))   # invert sigmoid
        log_prob    = dist.log_prob(raw_actions).sum(-1)
        entropy     = dist.entropy().sum(-1)
        value       = self.critic(h).squeeze(-1)
        return log_prob, entropy, value, mean   # mean returned for P1/P2


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
    print(f"  Columns: {list(splits['train'].columns)}")
    return splits


def create_sequences(df, win=WIN):
    """Sequential sliding windows; window order preserved for transition penalty."""
    mains    = df['aggregate'].values.astype(np.float32)
    app_vals = {a: df[a].values.astype(np.float32) for a in APPLIANCES}
    X, Y = [], []
    for i in range(0, len(mains) - win + 1, STRIDE):
        X.append(mains[i: i + win])
        mid = i + win // 2
        Y.append([app_vals[a][mid] for a in APPLIANCES])
    return (
        np.array(X, dtype=np.float32).reshape(-1, win, 1),
        np.array(Y, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def compute_rewards_batch(actions, y_true, prev_actions, y_mins, y_ranges):
    """
    R = -(α·MAE + β·transition) / NORM_POWER + ON_REWARD_COEF · on_reward

    P3: BETA_TRANS reduced so transition penalty no longer dominates.
    P4: NORM_POWER reduced so reward magnitude is in a useful range for PPO.
    P8: on_reward = mean(pred_on & true_on) gives explicit credit for TP detections.
    """
    pred_raw = actions      * y_ranges + y_mins
    true_raw = y_true       * y_ranges + y_mins
    prev_raw = prev_actions * y_ranges + y_mins

    mae_term   = np.abs(pred_raw - true_raw).mean(axis=1)
    trans_term = np.abs(pred_raw - prev_raw).mean(axis=1)

    pred_on   = (pred_raw > THRESHOLD_ARR).astype(np.float32)
    true_on   = (true_raw > THRESHOLD_ARR).astype(np.float32)
    on_reward = (pred_on * true_on).mean(axis=1)

    return (-(ALPHA * mae_term + BETA_TRANS * trans_term) / NORM_POWER
            + ON_REWARD_COEF * on_reward)


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
# Supervised pre-training (Huber warm-start, P7)
# ---------------------------------------------------------------------------

def pretrain_supervised(model, X_tr, Y_tr, device, epochs=PRETRAIN_EPOCHS):
    if epochs <= 0:
        return
    print(f"\n--- Supervised pre-training ({epochs} epochs, Huber loss) ---")
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    N = len(X_tr)
    for ep in range(epochs):
        perm = np.random.permutation(N)
        total_loss = 0.0
        n_batches  = 0
        for start in range(0, N, BATCH):
            idx  = perm[start: start + BATCH]
            xb   = torch.FloatTensor(X_tr[idx]).to(device)
            yb   = torch.FloatTensor(Y_tr[idx]).to(device)
            h    = model._encode(xb)
            mean = torch.sigmoid(model.actor_mean(h))
            loss = F.smooth_l1_loss(mean, yb)   # P7: Huber instead of MSE
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            total_loss += loss.item()
            n_batches  += 1
        print(f"  pretrain epoch {ep+1}/{epochs}  Huber={total_loss/n_batches:.6f}")
    print("--- Pre-training complete ---\n")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(data_splits, save_dir, hidden_size=128, dt=0.1, win=WIN):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}  hidden={hidden_size}  dt={dt}  win={win}")
    print(f"PPO: γ={GAMMA}  λ={GAE_LAMBDA}  ε={PPO_CLIP}  "
          f"ent_coef={ENTROPY_COEF}")
    print(f"Reward: α={ALPHA}  β_trans={BETA_TRANS}  norm={NORM_POWER}  "
          f"on_coef={ON_REWARD_COEF}")
    print(f"Aux losses: BC_COEF={BC_COEF}  BCE_COEF={BCE_COEF}")

    tr_df = data_splits['train']
    va_df = data_splits['validation']
    te_df = data_splits['test']

    X_tr, Y_tr = create_sequences(tr_df, win)
    X_va, Y_va = create_sequences(va_df, win)
    X_te, Y_te = create_sequences(te_df, win)

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

    # P2: binary ON/OFF targets for BCE auxiliary loss
    # Computed from inverse-transformed Y_tr so the threshold comparison is in Watts.
    Y_onoff_tr = np.zeros_like(Y_tr)
    for i, app in enumerate(APPLIANCES):
        raw_vals = y_scalers[i].inverse_transform(Y_tr[:, i:i+1]).flatten()
        Y_onoff_tr[:, i] = (raw_vals > THRESHOLDS[app]).astype(np.float32)

    on_frac = Y_onoff_tr.mean(axis=0)
    print(f"\nTrain: {X_tr.shape}  Val: {X_va.shape}  Test: {X_te.shape}")
    print(f"ON fraction train: "
          + "  ".join(f"{a}={on_frac[i]:.3f}" for i, a in enumerate(APPLIANCES)))

    model     = LNNActorCritic(1, hidden_size, len(APPLIANCES), dt).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    pretrain_supervised(model, X_tr, Y_tr, device, epochs=PRETRAIN_EPOCHS)

    print("Starting LNN-PPO-v4 training...\n")

    train_start = time.time()
    history = {
        'policy_loss': [], 'value_loss': [], 'entropy': [],
        'bc_loss': [], 'bce_loss': [],
        'mean_reward': [], 'val_metrics': [],
    }
    best_f1    = -1.0
    best_state = None
    counter    = 0

    for epoch in range(EPOCHS):
        ep_start   = time.time()
        model.train()
        ep_pi_loss = ep_v_loss = ep_ent = ep_reward = 0.0
        ep_bc_loss = ep_bce_loss = 0.0
        n_updates  = 0

        prev_action_np = np.zeros(len(APPLIANCES), dtype=np.float32)

        bar = tqdm(range(0, len(X_tr), ROLLOUT_SIZE),
                   desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        for rollout_start in bar:
            rollout_end  = min(rollout_start + ROLLOUT_SIZE, len(X_tr))
            N            = rollout_end - rollout_start

            x_chunk       = torch.FloatTensor(X_tr[rollout_start:rollout_end]).to(device)
            y_chunk       = Y_tr[rollout_start:rollout_end]        # scaled
            y_onoff_chunk = Y_onoff_tr[rollout_start:rollout_end]  # binary

            # Pre-convert to tensors for indexing inside PPO update
            y_chunk_t      = torch.FloatTensor(y_chunk).to(device)
            y_onoff_chunk_t = torch.FloatTensor(y_onoff_chunk).to(device)

            # ── Collect rollout (no-grad) ──────────────────────────────────
            model.eval()
            with torch.no_grad():
                actions_t, logprobs_t, _, values_t = model.get_action(x_chunk)

            actions_np  = actions_t.cpu().numpy()
            logprobs_np = logprobs_t.cpu().numpy()
            values_np   = values_t.cpu().numpy().tolist()

            prev_np = np.vstack([prev_action_np[None, :], actions_np[:-1]])

            rewards_np = compute_rewards_batch(
                actions_np, y_chunk, prev_np, y_mins, y_ranges)

            prev_action_np = actions_np[-1]

            with torch.no_grad():
                _, _, _, last_val_t = model.get_action(
                    x_chunk[-1:], deterministic=True)
            last_value = last_val_t[0].item()

            # ── GAE ────────────────────────────────────────────────────────
            advantages_list, returns_list = compute_gae(
                rewards_np.tolist(), values_np, last_value)

            advantages_t = torch.tensor(advantages_list, dtype=torch.float32, device=device)
            returns_t2   = torch.tensor(returns_list,    dtype=torch.float32, device=device)
            old_lp_t     = torch.tensor(logprobs_np,     dtype=torch.float32, device=device)
            old_acts_t   = actions_t

            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

            ep_reward += float(rewards_np.mean())

            # ── PPO update ──────────────────────────────────────────────────
            model.train()
            for _ in range(PPO_EPOCHS):
                perm = torch.randperm(N, device=device)
                for start in range(0, N, BATCH):
                    idx = perm[start: start + BATCH]
                    xb  = x_chunk[idx]
                    ab  = old_acts_t[idx]
                    adv = advantages_t[idx]
                    ret = returns_t2[idx]
                    olp = old_lp_t[idx]

                    # evaluate_actions also returns mean (P1, P2 — avoids second _encode)
                    new_lp, ent, vals, mean = model.evaluate_actions(xb, ab)

                    ratio   = (new_lp - olp).exp()
                    surr1   = ratio * adv
                    surr2   = ratio.clamp(1 - PPO_CLIP, 1 + PPO_CLIP) * adv
                    pi_loss = -torch.min(surr1, surr2).mean()
                    v_loss  = VALUE_COEF * (ret - vals).pow(2).mean()
                    e_loss  = -ENTROPY_COEF * ent.mean()

                    # P1: behaviour-cloning loss (Huber)
                    target_bc    = y_chunk_t[idx]
                    bc_loss      = F.smooth_l1_loss(mean, target_bc)

                    # P2: binary ON/OFF auxiliary BCE
                    target_onoff = y_onoff_chunk_t[idx]
                    bce_loss     = F.binary_cross_entropy(
                        mean.clamp(1e-6, 1 - 1e-6), target_onoff)

                    loss = (pi_loss + v_loss + e_loss
                            + BC_COEF * bc_loss
                            + BCE_COEF * bce_loss)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

                    ep_pi_loss  += pi_loss.item()
                    ep_v_loss   += v_loss.item()
                    ep_ent      += ent.mean().item()
                    ep_bc_loss  += bc_loss.item()
                    ep_bce_loss += bce_loss.item()
                    n_updates   += 1

        if n_updates:
            ep_pi_loss  /= n_updates
            ep_v_loss   /= n_updates
            ep_ent      /= n_updates
            ep_bc_loss  /= n_updates
            ep_bce_loss /= n_updates

        rollouts_per_epoch = max(1, len(X_tr) // ROLLOUT_SIZE)
        ep_reward /= rollouts_per_epoch

        val_metrics = evaluate(model, X_va, Y_va, y_scalers, device)
        history['val_metrics'].append(val_metrics)
        history['policy_loss'].append(ep_pi_loss)
        history['value_loss'].append(ep_v_loss)
        history['entropy'].append(ep_ent)
        history['bc_loss'].append(ep_bc_loss)
        history['bce_loss'].append(ep_bce_loss)
        history['mean_reward'].append(ep_reward)

        avg_f1  = np.mean([val_metrics[a]['f1']  for a in APPLIANCES])
        avg_mae = np.mean([val_metrics[a]['mae'] for a in APPLIANCES])

        ep_time = time.time() - ep_start
        print(
            f"  Epoch {epoch+1:3d}/{EPOCHS}  "
            f"π={ep_pi_loss:.5f}  V={ep_v_loss:.5f}  "
            f"H={ep_ent:.3f}  BC={ep_bc_loss:.5f}  BCE={ep_bce_loss:.5f}  "
            f"R={ep_reward:.5f}  avgF1={avg_f1:.4f}  avgMAE={avg_mae:.2f}  "
            f"time={ep_time:.1f}s"
        )
        for app in APPLIANCES:
            m = val_metrics[app]
            print(f"    {app:<22s}  F1={m['f1']:.4f}  "
                  f"P={m['precision']:.4f}  R={m['recall']:.4f}  "
                  f"MAE={m['mae']:.2f}  SAE={m['sae']:.4f}  "
                  f"TP={m['TP']}  FP={m['FP']}  FN={m['FN']}")

        if avg_f1 > best_f1:
            best_f1    = avg_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            counter    = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    total_train_time = time.time() - train_start
    print(f"Training complete.  Total time: {total_train_time/60:.1f} min "
          f"({total_train_time:.0f}s)")

    if best_state:
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
    _save_json(test_metrics, hidden_size, dt, win, save_dir)
    return test_metrics, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_results(history, test_metrics, save_dir):
    epochs_x = range(1, len(history['policy_loss']) + 1)

    plt.figure(figsize=(20, 4))
    for col, (key, color, title) in enumerate([
        ('policy_loss', 'blue',   'Policy Loss (π)'),
        ('value_loss',  'red',    'Value Loss (V)'),
        ('entropy',     'green',  'Entropy (H)'),
        ('bc_loss',     'orange', 'BC Loss (Huber)'),
        ('bce_loss',    'purple', 'BCE Loss (ON/OFF)'),
    ]):
        plt.subplot(1, 5, col + 1)
        plt.plot(epochs_x, history[key], color=color)
        plt.title(title); plt.xlabel('Epoch'); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rl_training_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(len(APPLIANCES), 2, figsize=(12, 4 * len(APPLIANCES)))
    fig.suptitle('LNN-PPO-v4 UKDALE — Per-Appliance Val Metrics', fontsize=13)
    for row, app in enumerate(APPLIANCES):
        f1s  = [m[app]['f1']  for m in history['val_metrics']]
        maes = [m[app]['mae'] for m in history['val_metrics']]
        axes[row][0].plot(epochs_x, f1s)
        axes[row][0].axhline(test_metrics[app]['f1'], color='green',
                             linestyle='--', label='Test F1')
        axes[row][0].set_title(f'{app} — F1')
        axes[row][0].legend(); axes[row][0].grid(True, alpha=0.3)
        axes[row][1].plot(epochs_x, maes, color='red')
        axes[row][1].axhline(test_metrics[app]['mae'], color='green',
                             linestyle='--', label='Test MAE')
        axes[row][1].set_title(f'{app} — MAE (W)')
        axes[row][1].legend(); axes[row][1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_appliance_metrics.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------

def _save_json(test_metrics, hidden_size, dt, win, save_dir):
    config = {
        'model': 'LNNActorCritic PPO-v4 (PPO stabilisation)',
        'dataset': 'UKDALE-dataset',
        'hyperparams': {
            'WIN': win, 'STRIDE': STRIDE, 'BATCH': BATCH,
            'EPOCHS': EPOCHS, 'PATIENCE': PATIENCE, 'LR': LR,
            'hidden_size': hidden_size, 'dt': dt,
            'GAMMA': GAMMA, 'GAE_LAMBDA': GAE_LAMBDA, 'PPO_CLIP': PPO_CLIP,
            'VALUE_COEF': VALUE_COEF, 'ENTROPY_COEF': ENTROPY_COEF,
            'PPO_EPOCHS': PPO_EPOCHS, 'ROLLOUT_SIZE': ROLLOUT_SIZE,
            'ALPHA': ALPHA, 'BETA_TRANS': BETA_TRANS, 'NORM_POWER': NORM_POWER,
            'ON_REWARD_COEF': ON_REWARD_COEF,
            'BC_COEF': BC_COEF, 'BCE_COEF': BCE_COEF,
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
        description='LNN-PPO v4 NILM on UKDALE — BC + BCE + sigmoid actions')
    p.add_argument('--dataset-dir',   default=DEFAULT_DATASET_DIR)
    p.add_argument('--hidden-size',   type=int,   default=128,
                   help='LNN hidden size (P9: default 128, up from 64)')
    p.add_argument('--dt',            type=float, default=0.1)
    p.add_argument('--win',           type=int,   default=WIN,
                   help='Window length (P10: try 200 or 300)')
    p.add_argument('--alpha',         type=float, default=ALPHA)
    p.add_argument('--beta-trans',    type=float, default=BETA_TRANS,
                   help='Transition penalty weight (P3: default 0.005)')
    p.add_argument('--norm-power',    type=float, default=NORM_POWER,
                   help='Reward normalisation (P4: default 300)')
    p.add_argument('--on-coef',       type=float, default=ON_REWARD_COEF,
                   help='ON-state reward coefficient (P8: default 0.2)')
    p.add_argument('--bc-coef',       type=float, default=BC_COEF,
                   help='Behaviour-cloning loss weight (P1: default 0.3)')
    p.add_argument('--bce-coef',      type=float, default=BCE_COEF,
                   help='Binary ON/OFF BCE loss weight (P2: default 0.2)')
    p.add_argument('--pretrain',      type=int,   default=PRETRAIN_EPOCHS)
    return p.parse_args()


if __name__ == '__main__':
    script_start = time.time()
    args = parse_args()

    for split in ('train', 'validation', 'test'):
        path = os.path.join(args.dataset_dir, f'UKDALE_HF_{split}.csv')
        if not os.path.exists(path):
            print(f"Error: {path} not found. Run preprocess_hf.py first.")
            sys.exit(1)

    ALPHA           = args.alpha
    BETA_TRANS      = args.beta_trans
    NORM_POWER      = args.norm_power
    ON_REWARD_COEF  = args.on_coef
    BC_COEF         = args.bc_coef
    BCE_COEF        = args.bce_coef
    PRETRAIN_EPOCHS = args.pretrain

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir  = f'models/lnn_rl_v4_{timestamp}'

    data_splits = load_data(args.dataset_dir)

    train_model(
        data_splits,
        save_dir    = save_dir,
        hidden_size = args.hidden_size,
        dt          = args.dt,
        win         = args.win,
    )
    total_time = time.time() - script_start
    print(f"\nAll outputs saved to {save_dir}/")
    print(f"Total wall-clock time: {total_time/60:.1f} min ({total_time:.0f}s)")
