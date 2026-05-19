"""
LNN + PPO Actor-Critic for NILM — UKDALE  (v3: impedance reward).

Extends v2 with three fixes for the always-OFF entropy-collapse observed in v2:

  Fix A — FP Dead-Zone Impedance
     Instead of triggering an FP penalty on any non-zero prediction, the agent
     gets a tolerance band up to FP_DEAD_ZONE (default 15 W).  Only predictions
     that exceed the dead-zone while the appliance is truly OFF (< TRUE_OFF_THR,
     default 5 W) incur a penalty, and only on the excess above the dead-zone:

         fp_impedance = FP_SCALE × max(0, pred − FP_DEAD_ZONE)

     This prevents the policy from becoming terrified of any non-zero emission:
     the agent can safely explore up to 15 W without punishment.

  Fix B — Entropy Floor
     ENTROPY_COEF is kept at 0.05 (raised from v1's 0.01) to slow entropy
     collapse.  The entropy term in the PPO loss also enforces a soft floor
     so the policy cannot contract into a deterministic always-OFF spike.

  Fix C — Balanced FN Penalty
     An explicit False-Negative penalty is added to counter the lazy-zero
     strategy.  When an appliance is truly ON (> FP_DEAD_ZONE W) but the
     agent predicts nearly zero (< TRUE_OFF_THR W), it pays:

         fn_penalty = FN_SCALE × (true_watts − pred_watts)

     With FN_SCALE (default 4) > FP_SCALE (default 2), the agent slightly
     prefers raising a false alarm over silently missing a real event.

All three fixes are tunable via CLI flags.  The F1-based primary reward and
conservation guardrail from v2 are retained.

Architecture : shared LNN encoder → Gaussian actor + critic V(s)
Training     : PPO with GAE, supervised MSE pre-training
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
WIN    = 100
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
ENTROPY_COEF = 0.05   # raised from v1's 0.01 to slow entropy collapse
ENTROPY_FLOOR = 0.1   # nats — soft lower bound on mean entropy per update
PPO_EPOCHS   = 4
ROLLOUT_SIZE = 512

# Reward shaping — base terms
ALPHA        = 1.0    # MAE weight
BETA_TRANS   = 0.05   # transition penalty weight
NORM_POWER   = 3000.0 # normalisation constant (W)
PRETRAIN_EPOCHS = 20  # supervised MSE warm-start

# Reward shaping — v3: impedance + FN balance + conservation
FP_DEAD_ZONE = 15.0  # W — predictions up to this are penalty-free (tolerance band)
TRUE_OFF_THR =  5.0  # W — true values below this = considered truly OFF for penalties
FP_SCALE     =  2.0  # multiplier on excess above dead-zone for FP impedance
FN_SCALE     =  4.0  # multiplier on missed wattage for FN penalty (> FP_SCALE)
BETA_CONSERVE = 1.0  # conservation guardrail weight

THRESHOLDS = {
    'dishwasher':      10.0,
    'fridge':          10.0,
    'microwave':       10.0,
    'washing_machine': 10.0,
}
THRESHOLD_ARR = np.array([THRESHOLDS[a] for a in APPLIANCES], dtype=np.float32)

DEFAULT_DATASET_DIR = 'dataset'


# ---------------------------------------------------------------------------
# Model  (unchanged from v1/v2)
# ---------------------------------------------------------------------------

class LNNActorCritic(nn.Module):
    """Shared LNN encoder → Gaussian actor + scalar critic."""

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
        return log_prob, entropy, value


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


def create_sequences(df):
    """
    Sequential sliding windows with midpoint aggregate for conservation check.

    Returns
    -------
    X   : (N, WIN, 1)  aggregate mains windows (before scaling)
    Y   : (N, n_apps)  mid-point appliance values (before scaling)
    Agg : (N,)         mid-point aggregate mains in raw Watts
    """
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
# Reward  (v3: F1-primary + FP impedance + FN balance + conservation)
# ---------------------------------------------------------------------------

def compute_rewards_batch(actions, y_true, prev_actions, y_mins, y_ranges, agg_watts):
    """
    Vectorised reward with v3 impedance and balance terms.

    Component breakdown
    -------------------
    Primary   : per-sample F1 averaged across appliances (from v2).
                Prevents collapse to always-ON or always-OFF regardless of
                class sparsity.

    Secondary : MAE + transition penalty / NORM_POWER (regression quality).

    FP impedance (Fix A):
                Only triggers when true_watts < TRUE_OFF_THR (5 W) AND
                pred_watts > FP_DEAD_ZONE (15 W).  Penalty = FP_SCALE × excess
                above the dead-zone.  The tolerance band [0, 15 W] lets the
                agent explore low-level predictions without punishment.

    FN penalty (Fix C):
                Only triggers when true_watts > FP_DEAD_ZONE (15 W) AND
                pred_watts < TRUE_OFF_THR (5 W).  Penalty = FN_SCALE × missed
                wattage.  With FN_SCALE > FP_SCALE the agent slightly prefers
                false alarms over silent misses.

    Conservation guardrail (from v2):
                Penalises Σpred > agg × 1.1 (Watts).

    Parameters
    ----------
    actions, y_true, prev_actions : (N, n_apps) scaled [0, 1]
    y_mins, y_ranges              : (n_apps,) inverse-scaling constants
    agg_watts                     : (N,) midpoint aggregate mains in raw Watts

    Returns
    -------
    rewards : (N,) float32
    """
    pred_raw = actions      * y_ranges + y_mins  # → Watts
    true_raw = y_true       * y_ranges + y_mins
    prev_raw = prev_actions * y_ranges + y_mins

    N = len(actions)

    # ── MAE + transition ─────────────────────────────────────────────────
    mae_term   = np.abs(pred_raw - true_raw).mean(axis=1)
    trans_term = np.abs(pred_raw - prev_raw).mean(axis=1)

    # ── 1. F1-based primary ───────────────────────────────────────────────
    pred_on = pred_raw > THRESHOLD_ARR
    true_on = true_raw > THRESHOLD_ARR
    tp = (pred_on & true_on).astype(np.float32)
    fp = (pred_on & ~true_on).astype(np.float32)
    fn = (~pred_on & true_on).astype(np.float32)
    f1_per_app = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
    f1_reward  = f1_per_app.mean(axis=1)          # (N,) ∈ [0, 1]

    # ── 2. FP impedance with dead-zone (Fix A) ───────────────────────────
    # Penalty activates only when: true < TRUE_OFF_THR AND pred > FP_DEAD_ZONE
    # Cost = FP_SCALE × (pred − FP_DEAD_ZONE)  — only excess above the band.
    fp_impedance = np.zeros(N, dtype=np.float32)
    fn_missed    = np.zeros(N, dtype=np.float32)
    for i in range(len(APPLIANCES)):
        truly_off = true_raw[:, i] < TRUE_OFF_THR
        pred_over = pred_raw[:, i] > FP_DEAD_ZONE
        excess    = np.maximum(0.0, pred_raw[:, i] - FP_DEAD_ZONE)
        fp_impedance += np.where(truly_off & pred_over, FP_SCALE * excess, 0.0)

        # ── 3. FN penalty — balanced FN/FP (Fix C) ───────────────────────
        # Penalty activates only when: true > FP_DEAD_ZONE AND pred < TRUE_OFF_THR
        truly_on  = true_raw[:, i] > FP_DEAD_ZONE
        pred_low  = pred_raw[:, i] < TRUE_OFF_THR
        missed_w  = true_raw[:, i] - pred_raw[:, i]
        fn_missed += np.where(truly_on & pred_low, FN_SCALE * missed_w, 0.0)

    # ── 4. Conservation guardrail (from v2) ──────────────────────────────
    conservation = np.maximum(0.0, pred_raw.sum(axis=1) - agg_watts * 1.1)

    reward = (f1_reward
              - (ALPHA * mae_term
                 + BETA_TRANS * trans_term
                 + fp_impedance
                 + fn_missed
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
# Supervised pre-training (MSE warm-start)
# ---------------------------------------------------------------------------

def pretrain_supervised(model, X_tr, Y_tr, device, epochs=PRETRAIN_EPOCHS):
    if epochs <= 0:
        return
    print(f"\n--- Supervised pre-training ({epochs} epochs) ---")
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
            loss = F.mse_loss(mean, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            total_loss += loss.item()
            n_batches  += 1
        print(f"  pretrain epoch {ep+1}/{epochs}  MSE={total_loss/n_batches:.6f}")
    print("--- Pre-training complete ---\n")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(data_splits, save_dir, hidden_size=64, dt=0.1):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}  hidden={hidden_size}  dt={dt}")
    print(f"PPO: γ={GAMMA}  λ={GAE_LAMBDA}  ε={PPO_CLIP}  "
          f"ent_coef={ENTROPY_COEF}  ent_floor={ENTROPY_FLOOR}")
    print(f"v3 reward: FP_dead={FP_DEAD_ZONE}W  true_off_thr={TRUE_OFF_THR}W  "
          f"FP_scale={FP_SCALE}  FN_scale={FN_SCALE}  β_conserve={BETA_CONSERVE}")

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
    print(f"Agg_tr range: [{Agg_tr.min():.1f}, {Agg_tr.max():.1f}] W")

    model     = LNNActorCritic(1, hidden_size, len(APPLIANCES), dt).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    pretrain_supervised(model, X_tr, Y_tr, device, epochs=PRETRAIN_EPOCHS)

    print("Starting LNN-PPO-v3 training...\n")

    train_start = time.time()
    history = {
        'policy_loss': [], 'value_loss': [], 'entropy': [],
        'mean_reward': [], 'val_metrics': [],
    }
    best_f1    = -1.0
    best_state = None
    counter    = 0

    ent_floor_t = torch.tensor(ENTROPY_FLOOR, dtype=torch.float32)

    for epoch in range(EPOCHS):
        ep_start   = time.time()
        model.train()
        ep_pi_loss = ep_v_loss = ep_ent = ep_reward = 0.0
        n_updates  = 0

        prev_action_np = np.zeros(len(APPLIANCES), dtype=np.float32)

        bar = tqdm(range(0, len(X_tr), ROLLOUT_SIZE),
                   desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        for rollout_start in bar:
            rollout_end = min(rollout_start + ROLLOUT_SIZE, len(X_tr))
            N = rollout_end - rollout_start

            x_chunk   = torch.FloatTensor(X_tr[rollout_start:rollout_end]).to(device)
            y_chunk   = Y_tr[rollout_start:rollout_end]
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

                    new_lp, ent, vals = model.evaluate_actions(xb, ab)

                    ratio   = (new_lp - olp).exp()
                    surr1   = ratio * adv
                    surr2   = ratio.clamp(1 - PPO_CLIP, 1 + PPO_CLIP) * adv
                    pi_loss = -torch.min(surr1, surr2).mean()
                    v_loss  = VALUE_COEF * (ret - vals).pow(2).mean()

                    # Entropy bonus with soft floor (Fix B):
                    # if ent drops below ENTROPY_FLOOR, use the floor value so
                    # the gradient still pushes entropy upward.
                    ent_eff = torch.max(ent, ent_floor_t.to(device))
                    e_loss  = -ENTROPY_COEF * ent_eff.mean()

                    loss = pi_loss + v_loss + e_loss
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    optimizer.step()

                    ep_pi_loss += pi_loss.item()
                    ep_v_loss  += v_loss.item()
                    ep_ent     += ent.mean().item()
                    n_updates  += 1

        if n_updates:
            ep_pi_loss /= n_updates
            ep_v_loss  /= n_updates
            ep_ent     /= n_updates

        rollouts_per_epoch = max(1, len(X_tr) // ROLLOUT_SIZE)
        ep_reward /= rollouts_per_epoch

        val_metrics = evaluate(model, X_va, Y_va, y_scalers, device)
        history['val_metrics'].append(val_metrics)
        history['policy_loss'].append(ep_pi_loss)
        history['value_loss'].append(ep_v_loss)
        history['entropy'].append(ep_ent)
        history['mean_reward'].append(ep_reward)

        avg_f1  = np.mean([val_metrics[a]['f1']  for a in APPLIANCES])
        avg_mae = np.mean([val_metrics[a]['mae'] for a in APPLIANCES])

        ep_time = time.time() - ep_start
        print(
            f"  Epoch {epoch+1:3d}/{EPOCHS}  "
            f"π={ep_pi_loss:.5f}  V={ep_v_loss:.5f}  "
            f"H={ep_ent:.3f}  R={ep_reward:.5f}  "
            f"avgF1={avg_f1:.4f}  avgMAE={avg_mae:.2f}  "
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
    _save_json(test_metrics, hidden_size, dt, save_dir)
    return test_metrics, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_results(history, test_metrics, save_dir):
    epochs_x = range(1, len(history['policy_loss']) + 1)

    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1)
    plt.plot(epochs_x, history['policy_loss'], color='blue')
    plt.title('Policy Loss (π)'); plt.xlabel('Epoch'); plt.grid(True, alpha=0.3)
    plt.subplot(1, 4, 2)
    plt.plot(epochs_x, history['value_loss'], color='red')
    plt.title('Value Loss (V)'); plt.xlabel('Epoch'); plt.grid(True, alpha=0.3)
    plt.subplot(1, 4, 3)
    plt.plot(epochs_x, history['entropy'], color='green')
    plt.axhline(ENTROPY_FLOOR, color='orange', linestyle='--', linewidth=0.8,
                label=f'floor={ENTROPY_FLOOR}')
    plt.title('Entropy (H)'); plt.xlabel('Epoch'); plt.legend(fontsize=7)
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 4, 4)
    plt.plot(epochs_x, history['mean_reward'], color='purple')
    plt.title('Mean Reward'); plt.xlabel('Epoch'); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rl_training_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(len(APPLIANCES), 2, figsize=(12, 4 * len(APPLIANCES)))
    fig.suptitle('LNN-PPO-v3 UKDALE — Per-Appliance Val Metrics', fontsize=13)
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

def _save_json(test_metrics, hidden_size, dt, save_dir):
    config = {
        'model': 'LNNActorCritic PPO-v3 (impedance reward)',
        'dataset': 'UKDALE-dataset',
        'hyperparams': {
            'WIN': WIN, 'STRIDE': STRIDE, 'BATCH': BATCH,
            'EPOCHS': EPOCHS, 'PATIENCE': PATIENCE, 'LR': LR,
            'hidden_size': hidden_size, 'dt': dt,
            'GAMMA': GAMMA, 'GAE_LAMBDA': GAE_LAMBDA, 'PPO_CLIP': PPO_CLIP,
            'VALUE_COEF': VALUE_COEF, 'ENTROPY_COEF': ENTROPY_COEF,
            'ENTROPY_FLOOR': ENTROPY_FLOOR,
            'PPO_EPOCHS': PPO_EPOCHS, 'ROLLOUT_SIZE': ROLLOUT_SIZE,
            'ALPHA': ALPHA, 'BETA_TRANS': BETA_TRANS, 'NORM_POWER': NORM_POWER,
            'FP_DEAD_ZONE': FP_DEAD_ZONE, 'TRUE_OFF_THR': TRUE_OFF_THR,
            'FP_SCALE': FP_SCALE, 'FN_SCALE': FN_SCALE,
            'BETA_CONSERVE': BETA_CONSERVE,
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
        description='LNN-PPO v3 NILM on UKDALE — FP impedance / FN balance / entropy floor')
    p.add_argument('--dataset-dir',   default=DEFAULT_DATASET_DIR,
                   help='Directory containing UKDALE_HF_*.csv files')
    p.add_argument('--hidden-size',   type=int,   default=64)
    p.add_argument('--dt',            type=float, default=0.1)
    p.add_argument('--alpha',         type=float, default=ALPHA,
                   help='MAE reward weight')
    p.add_argument('--beta-trans',    type=float, default=BETA_TRANS,
                   help='Transition penalty weight')
    p.add_argument('--fp-dead-zone',  type=float, default=FP_DEAD_ZONE,
                   help='Prediction dead-zone in W — FP penalty only above this (default 15)')
    p.add_argument('--true-off-thr',  type=float, default=TRUE_OFF_THR,
                   help='True-OFF threshold in W — FP/FN penalty uses this (default 5)')
    p.add_argument('--fp-scale',      type=float, default=FP_SCALE,
                   help='FP impedance multiplier on excess above dead-zone (default 2)')
    p.add_argument('--fn-scale',      type=float, default=FN_SCALE,
                   help='FN missed-activation multiplier (default 4, > fp-scale)')
    p.add_argument('--beta-conserve', type=float, default=BETA_CONSERVE,
                   help='Conservation guardrail weight (default 1.0)')
    p.add_argument('--ent-floor',     type=float, default=ENTROPY_FLOOR,
                   help='Soft entropy floor in nats (default 0.1)')
    p.add_argument('--pretrain',      type=int,   default=PRETRAIN_EPOCHS,
                   help='Supervised pre-train epochs (0 to skip)')
    return p.parse_args()


if __name__ == '__main__':
    script_start = time.time()
    args = parse_args()

    for split in ('train', 'validation', 'test'):
        p = os.path.join(args.dataset_dir, f'UKDALE_HF_{split}.csv')
        if not os.path.exists(p):
            print(f"Error: {p} not found. Run preprocess_hf.py first.")
            sys.exit(1)

    ALPHA          = args.alpha
    BETA_TRANS     = args.beta_trans
    FP_DEAD_ZONE   = args.fp_dead_zone
    TRUE_OFF_THR   = args.true_off_thr
    FP_SCALE       = args.fp_scale
    FN_SCALE       = args.fn_scale
    BETA_CONSERVE  = args.beta_conserve
    ENTROPY_FLOOR  = args.ent_floor
    PRETRAIN_EPOCHS = args.pretrain

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir  = f'models/lnn_rl_v3_{timestamp}'

    data_splits = load_data(args.dataset_dir)

    train_model(
        data_splits,
        save_dir    = save_dir,
        hidden_size = args.hidden_size,
        dt          = args.dt,
    )
    total_time = time.time() - script_start
    print(f"\nAll outputs saved to {save_dir}/")
    print(f"Total wall-clock time: {total_time/60:.1f} min ({total_time:.0f}s)")
