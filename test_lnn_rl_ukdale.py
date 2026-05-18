"""
LNN + Proximal Policy Optimisation (PPO) Actor-Critic for NILM — UKDALE.

Frames energy disaggregation as a Markov Decision Process (MDP):

  State  S_t : aggregate mains window (WIN timesteps) encoded by a shared LNN
  Action A_t : predicted scaled power for each appliance  ∈ [0, 1]
  Reward R_t : -α·MAE(p_true, p_pred) - β·transition_penalty(A_t, A_{t-1})
               (both terms normalised by NORM_POWER so reward ∈ [-1, 0])

The transition penalty naturally discourages rapid appliance flickering,
replacing the manual min-ON/OFF post-processing filters used in supervised scripts.

Architecture:
  Shared LNN encoder (LiquidCell — adaptive τ, input-dependent gate)
        ↓
  ┌─────────────────────────┬──────────────────┐
  │ Actor  (Gaussian policy)│ Critic  V(s)     │
  │ mean = sigmoid(linear)  │ linear → scalar  │
  │ std  = exp(log_std_param│                  │
  └─────────────────────────┴──────────────────┘

Training: PPO with GAE advantages, offline sequential rollouts over CSV data.
Dataset : medium_dataset/ (or --dataset-dir override)
"""

import sys
import os
import argparse
import json
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
GAMMA        = 0.99    # discount factor
GAE_LAMBDA   = 0.95    # GAE λ
PPO_CLIP     = 0.2     # clip ratio ε
VALUE_COEF   = 0.5     # critic loss weight
ENTROPY_COEF = 0.05    # entropy bonus weight — higher to prevent std collapse
PPO_EPOCHS   = 4       # update sweeps per rollout
ROLLOUT_SIZE = 512     # windows per rollout segment

# Reward shaping
ALPHA        = 1.0    # disaggregation MAE weight
BETA_TRANS   = 0.05   # state-transition penalty weight
GAMMA_SPARSE = 0.5    # sparsity penalty — discourages always-ON collapse
NORM_POWER   = 3000.0 # normalisation constant (W) — typical UK household max

THRESHOLDS = {
    'dishwasher':      10.0,
    'fridge':          10.0,
    'microwave':       10.0,
    'washing_machine': 10.0,
}

DEFAULT_DATASET_DIR = 'dataset'


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LNNActorCritic(nn.Module):
    """
    Shared LNN encoder → Gaussian actor + scalar critic.

    The encoder runs a LiquidCell over the aggregate mains window, producing a
    hidden state h that captures the temporal energy signature.  The actor maps
    h to a Gaussian distribution over scaled appliance power; the critic maps h
    to V(s) for advantage estimation.
    """

    def __init__(self, input_size, hidden_size, n_appliances, dt=0.1):
        super().__init__()
        self.hidden_size  = hidden_size
        self.n_appliances = n_appliances
        self.dt           = dt

        # ── Shared LiquidCell encoder ──────────────────────────────────────
        self.input_proj  = nn.Linear(input_size, hidden_size)
        self.tau_base    = nn.Parameter(torch.ones(hidden_size))
        self.tau_mod     = nn.Linear(input_size, hidden_size)
        self.rec_weights = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.rec_weights)
        self.gate        = nn.Linear(input_size + hidden_size, hidden_size)
        self.norm        = nn.LayerNorm(hidden_size)

        # ── Actor ──────────────────────────────────────────────────────────
        self.actor_mean    = nn.Linear(hidden_size, n_appliances)
        # Per-appliance log-std — learned but shared across the batch
        self.actor_log_std = nn.Parameter(torch.full((n_appliances,), 0.0))

        # ── Critic ─────────────────────────────────────────────────────────
        self.critic = nn.Linear(hidden_size, 1)

    # ------------------------------------------------------------------
    def _encode(self, x):
        """LiquidCell forward pass.  x: (B, WIN, 1) → h: (B, hidden)"""
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

    # ------------------------------------------------------------------
    def get_action(self, x, deterministic=False):
        """
        Sample action from the current policy.

        Returns
        -------
        action   : (B, n_apps)  scaled ∈ [0, 1]
        log_prob : (B,)
        entropy  : (B,)
        value    : (B,)
        """
        h    = self._encode(x)
        mean = torch.sigmoid(self.actor_mean(h))          # [0, 1]
        std  = self.actor_log_std.exp().clamp(0.01, 1.0)
        dist = Normal(mean, std)

        if deterministic:
            action = mean
        else:
            action = dist.rsample().clamp(0.0, 1.0)

        log_prob = dist.log_prob(action.clamp(1e-6, 1 - 1e-6)).sum(-1)
        entropy  = dist.entropy().sum(-1)
        value    = self.critic(h).squeeze(-1)
        return action, log_prob, entropy, value

    # ------------------------------------------------------------------
    def evaluate_actions(self, x, actions):
        """Recompute log_prob / entropy / value for stored (s, a) pairs."""
        h    = self._encode(x)
        mean = torch.sigmoid(self.actor_mean(h))
        std  = self.actor_log_std.exp().clamp(0.01, 1.0)
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
    """Sequential windows: window order is preserved for rollout transition penalty."""
    mains    = df['aggregate'].values.astype(np.float32)
    app_vals = {a: df[a].values.astype(np.float32) for a in APPLIANCES}
    X, Y = [], []
    for i in range(0, len(mains) - WIN + 1, STRIDE):
        X.append(mains[i: i + WIN])
        mid = i + WIN // 2
        Y.append([app_vals[a][mid] for a in APPLIANCES])
    return (
        np.array(X, dtype=np.float32).reshape(-1, WIN, 1),
        np.array(Y, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def compute_rewards_batch(actions, y_true, prev_actions, y_mins, y_ranges):
    """
    Vectorised reward: R = -(α·MAE + β·transition + γ·sparsity) / NORM_POWER

    All inputs are (N, n_apps) numpy arrays in scaled [0, 1] space.
    y_mins, y_ranges: (n_apps,) arrays for inverse scaling to Watts.
    """
    pred_raw = actions      * y_ranges + y_mins
    true_raw = y_true       * y_ranges + y_mins
    prev_raw = prev_actions * y_ranges + y_mins

    mae_term     = np.abs(pred_raw - true_raw).mean(axis=1)   # (N,)
    trans_term   = np.abs(pred_raw - prev_raw).mean(axis=1)   # (N,)
    sparse_term  = pred_raw.mean(axis=1)                       # (N,) — penalise high predictions
    return -(ALPHA * mae_term + BETA_TRANS * trans_term + GAMMA_SPARSE * sparse_term) / NORM_POWER


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------

def compute_gae(rewards, values, last_value):
    """Generalised Advantage Estimation → (advantages, returns) as lists."""
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
    """Deterministic policy → per-appliance metrics."""
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            xb     = torch.FloatTensor(X[i: i + BATCH]).to(device)
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
# Training
# ---------------------------------------------------------------------------

def train_model(data_splits, save_dir, hidden_size=64, dt=0.1):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}  hidden={hidden_size}  dt={dt}")
    print(f"PPO: γ={GAMMA}  λ={GAE_LAMBDA}  ε={PPO_CLIP}  "
          f"α_mae={ALPHA}  β_trans={BETA_TRANS}")

    tr_df = data_splits['train']
    va_df = data_splits['validation']
    te_df = data_splits['test']

    X_tr, Y_tr = create_sequences(tr_df)
    X_va, Y_va = create_sequences(va_df)
    X_te, Y_te = create_sequences(te_df)

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

    # Pre-compute inverse-scaling constants for vectorised reward
    y_mins   = np.array([float(ys.data_min_[0])   for ys in y_scalers])
    y_ranges = np.array([float(ys.data_range_[0]) for ys in y_scalers])

    print(f"\nTrain: {X_tr.shape}  Val: {X_va.shape}  Test: {X_te.shape}")

    model     = LNNActorCritic(1, hidden_size, len(APPLIANCES), dt).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    print("Starting LNN-PPO training...\n")

    history = {
        'policy_loss': [], 'value_loss': [], 'entropy': [],
        'mean_reward': [], 'val_metrics': [],
    }
    best_f1    = -1.0
    best_state = None
    counter    = 0

    for epoch in range(EPOCHS):
        model.train()
        ep_pi_loss = ep_v_loss = ep_ent = ep_reward = 0.0
        n_updates  = 0

        # prev_action tracks the last prediction across rollouts (for transition penalty)
        prev_action_np = np.zeros(len(APPLIANCES), dtype=np.float32)

        bar = tqdm(range(0, len(X_tr), ROLLOUT_SIZE),
                   desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        for rollout_start in bar:
            rollout_end = min(rollout_start + ROLLOUT_SIZE, len(X_tr))
            N = rollout_end - rollout_start

            x_chunk = torch.FloatTensor(X_tr[rollout_start:rollout_end]).to(device)
            y_chunk = Y_tr[rollout_start:rollout_end]   # (N, n_apps) numpy

            # ── Collect rollout (batched, no-grad) ──────────────────────────
            model.eval()
            with torch.no_grad():
                actions_t, logprobs_t, _, values_t = model.get_action(x_chunk)

            actions_np  = actions_t.cpu().numpy()   # (N, n_apps)
            logprobs_np = logprobs_t.cpu().numpy()  # (N,)
            values_np   = values_t.cpu().numpy().tolist()  # list of N floats

            # Build prev_actions array: prev[0] = prev from last rollout/epoch
            prev_np = np.vstack([
                prev_action_np[None, :],
                actions_np[:-1]
            ])                                       # (N, n_apps)

            rewards_np = compute_rewards_batch(
                actions_np, y_chunk, prev_np, y_mins, y_ranges)  # (N,)

            prev_action_np = actions_np[-1]  # carry forward

            # Bootstrap last value
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
            old_acts_t   = actions_t  # already on device

            # Normalise advantages
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

                    ratio  = (new_lp - olp).exp()
                    surr1  = ratio * adv
                    surr2  = ratio.clamp(1 - PPO_CLIP, 1 + PPO_CLIP) * adv
                    pi_loss = -torch.min(surr1, surr2).mean()
                    v_loss  = VALUE_COEF * (ret - vals).pow(2).mean()
                    e_loss  = -ENTROPY_COEF * ent.mean()

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

        # ── Validation ────────────────────────────────────────────────────
        val_metrics = evaluate(model, X_va, Y_va, y_scalers, device)
        history['val_metrics'].append(val_metrics)
        history['policy_loss'].append(ep_pi_loss)
        history['value_loss'].append(ep_v_loss)
        history['entropy'].append(ep_ent)
        history['mean_reward'].append(ep_reward)

        avg_f1  = np.mean([val_metrics[a]['f1']  for a in APPLIANCES])
        avg_mae = np.mean([val_metrics[a]['mae'] for a in APPLIANCES])

        print(
            f"  Epoch {epoch+1:3d}/{EPOCHS}  "
            f"π={ep_pi_loss:.5f}  V={ep_v_loss:.5f}  "
            f"H={ep_ent:.3f}  R={ep_reward:.5f}  "
            f"avgF1={avg_f1:.4f}  avgMAE={avg_mae:.2f}"
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

    print("Training complete.")

    if best_state:
        model.load_state_dict(best_state)

    # ── Test ──────────────────────────────────────────────────────────────
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
    plt.title('Entropy (H)'); plt.xlabel('Epoch'); plt.grid(True, alpha=0.3)

    plt.subplot(1, 4, 4)
    plt.plot(epochs_x, history['mean_reward'], color='purple')
    plt.title('Mean Reward'); plt.xlabel('Epoch'); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rl_training_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(len(APPLIANCES), 2, figsize=(12, 4 * len(APPLIANCES)))
    fig.suptitle('LNN-PPO UKDALE — Per-Appliance Val Metrics', fontsize=13)
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
        'model': 'LNNActorCritic (PPO)',
        'dataset': 'UKDALE-dataset',
        'hyperparams': {
            'WIN': WIN, 'STRIDE': STRIDE, 'BATCH': BATCH,
            'EPOCHS': EPOCHS, 'PATIENCE': PATIENCE, 'LR': LR,
            'hidden_size': hidden_size, 'dt': dt,
            'GAMMA': GAMMA, 'GAE_LAMBDA': GAE_LAMBDA, 'PPO_CLIP': PPO_CLIP,
            'VALUE_COEF': VALUE_COEF, 'ENTROPY_COEF': ENTROPY_COEF,
            'PPO_EPOCHS': PPO_EPOCHS, 'ROLLOUT_SIZE': ROLLOUT_SIZE,
            'ALPHA': ALPHA, 'BETA_TRANS': BETA_TRANS, 'NORM_POWER': NORM_POWER,
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
    p = argparse.ArgumentParser(description='LNN-PPO NILM on UKDALE CSV dataset')
    p.add_argument('--dataset-dir', default=DEFAULT_DATASET_DIR,
                   help='Directory containing UKDALE_HF_*.csv files')
    p.add_argument('--hidden-size', type=int,   default=64)
    p.add_argument('--dt',          type=float, default=0.1)
    p.add_argument('--alpha',       type=float, default=ALPHA,
                   help='MAE reward weight')
    p.add_argument('--beta-trans',  type=float, default=BETA_TRANS,
                   help='Transition penalty weight')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    for split in ('train', 'validation', 'test'):
        p = os.path.join(args.dataset_dir, f'UKDALE_HF_{split}.csv')
        if not os.path.exists(p):
            print(f"Error: {p} not found. Run preprocess_hf.py first.")
            import sys; sys.exit(1)

    # Override module-level constants from CLI
    ALPHA      = args.alpha
    BETA_TRANS = args.beta_trans

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir  = f'models/lnn_rl_{timestamp}'

    data_splits = load_data(args.dataset_dir)

    train_model(
        data_splits,
        save_dir    = save_dir,
        hidden_size = args.hidden_size,
        dt          = args.dt,
    )
    print(f"\nAll outputs saved to {save_dir}/")
