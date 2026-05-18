"""
LNN + Soft Actor-Critic (SAC) for NILM — UKDALE.

SAC is off-policy: a replay buffer stores all past (state, action, reward,
next_state) transitions and each sample is reused many times across epochs,
giving ~10× better sample efficiency than PPO.  This makes it well-suited for
single-day CSV datasets where on-policy methods collapse due to sparse rollouts.

Key differences from the PPO script
--------------------------------------
  Replay buffer       stores up to REPLAY_CAPACITY transitions; uniform random
                      mini-batch sampling decouples updates from collection
  Twin Q-networks     Q1, Q2 — take min to avoid over-estimation bias
  Polyak targets      slow-moving critic copies for stable TD bootstrapping
  Squashed Gaussian   tanh(u) → [0, 1] with Jacobian log-prob correction;
                      log_std predicted per state (not a single global param)
  Auto-temperature    α tuned to match target entropy = -n_appliances

Dataset strides
  STRIDE_COLLECT = 1  : maximise replay buffer density from limited data
  STRIDE_EVAL    = 5  : fast evaluation, consistent with the PPO script
"""

import sys
import os
import argparse
import json
import copy
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

WIN            = 100
STRIDE_COLLECT = 1     # stride=1 for dense replay buffer
STRIDE_EVAL    = 5     # stride=5 for fast evaluation
BATCH          = 256
EPOCHS         = 50
PATIENCE       = 15
LR             = 3e-4

# SAC
GAMMA              = 0.99
TAU                = 0.005   # Polyak averaging coefficient for target networks
ALPHA_INIT         = 0.2     # initial entropy temperature
TARGET_ENTROPY     = -float(len(APPLIANCES))   # heuristic: -n_actions
UPDATES_PER_EPOCH  = 400     # gradient updates per epoch (independent of data)
REPLAY_CAPACITY    = 50_000

# Reward
ALPHA_MAE  = 1.0
BETA_TRANS = 0.05
NORM_POWER = 3000.0

PRETRAIN_EPOCHS = 10

THRESHOLDS = {
    'dishwasher':      10.0,
    'fridge':          10.0,
    'microwave':       10.0,
    'washing_machine': 10.0,
}

DEFAULT_DATASET_DIR = 'dataset'


# ---------------------------------------------------------------------------
# LNN encoder (shared architecture, separate weights per module)
# ---------------------------------------------------------------------------

class LNNEncoder(nn.Module):
    """LiquidCell encoder: (B, WIN, 1) → (B, hidden_size)."""

    def __init__(self, input_size: int, hidden_size: int, dt: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt          = dt
        self.input_proj  = nn.Linear(input_size, hidden_size)
        self.tau_base    = nn.Parameter(torch.ones(hidden_size))
        self.tau_mod     = nn.Linear(input_size, hidden_size)
        self.rec_weights = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.rec_weights)
        self.gate        = nn.Linear(input_size + hidden_size, hidden_size)
        self.norm        = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.size()
        h = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(T):
            x_t      = x[:, t, :]
            tau_base = F.softplus(self.tau_base).unsqueeze(0)
            tau_mod  = torch.sigmoid(self.tau_mod(x_t))
            tau      = (tau_base * tau_mod).clamp(min=self.dt)
            gate     = torch.sigmoid(self.gate(torch.cat([x_t, h], dim=1)))
            f_t      = torch.tanh(self.input_proj(x_t) +
                                  torch.matmul(h, self.rec_weights))
            dh       = ((-h / tau) + gate * f_t) * self.dt
            h        = (h + dh).clamp(-10.0, 10.0)
        return self.norm(h)


# ---------------------------------------------------------------------------
# Actor — squashed Gaussian (tanh → [0, 1])
# ---------------------------------------------------------------------------

class SACActorLNN(nn.Module):
    """
    State-dependent Gaussian actor with tanh squashing.

    Outputs action ∈ (0, 1)^n_apps via:
        u ~ N(μ(s), σ(s))
        a = (tanh(u) + 1) / 2

    Log-prob includes the Jacobian correction for both tanh and the [0,1]
    rescaling so that automatic temperature tuning remains well-calibrated.
    """

    LOG_STD_MIN = -5
    LOG_STD_MAX =  2

    def __init__(self, input_size: int, hidden_size: int,
                 n_appliances: int, dt: float = 0.1):
        super().__init__()
        self.encoder       = LNNEncoder(input_size, hidden_size, dt)
        self.mean_layer    = nn.Linear(hidden_size, n_appliances)
        self.log_std_layer = nn.Linear(hidden_size, n_appliances)

    def _dist(self, x):
        h       = self.encoder(x)
        mean    = self.mean_layer(h)
        log_std = self.log_std_layer(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std.exp()

    def get_action(self, x, deterministic: bool = False):
        """
        Returns
        -------
        action   : (B, n_apps)  ∈ (0, 1)
        log_prob : (B,)  with Jacobian correction for tanh + [0,1] rescaling
        """
        mean, std = self._dist(x)
        u       = mean if deterministic else mean + std * torch.randn_like(mean)
        a_tanh  = torch.tanh(u)                      # ∈ (-1, 1)
        action  = (a_tanh + 1.0) / 2.0              # ∈ (0, 1)
        # Jacobian: log|da/du| = log(1-tanh²) + log(0.5) per dimension
        log_prob = (Normal(mean, std).log_prob(u)
                    - torch.log(1.0 - a_tanh.pow(2) + 1e-6)
                    - np.log(2.0)).sum(dim=-1)
        return action, log_prob


# ---------------------------------------------------------------------------
# Critic — twin Q-networks
# ---------------------------------------------------------------------------

class SACCriticLNN(nn.Module):
    """
    Two independent Q(s, a) networks to mitigate over-estimation bias.
    Each has its own LNN encoder so gradients are isolated.
    """

    def __init__(self, input_size: int, hidden_size: int,
                 n_appliances: int, dt: float = 0.1):
        super().__init__()
        h2 = max(hidden_size // 2, 16)
        self.enc1 = LNNEncoder(input_size, hidden_size, dt)
        self.q1   = nn.Sequential(
            nn.Linear(hidden_size + n_appliances, h2), nn.ReLU(),
            nn.Linear(h2, 1),
        )
        self.enc2 = LNNEncoder(input_size, hidden_size, dt)
        self.q2   = nn.Sequential(
            nn.Linear(hidden_size + n_appliances, h2), nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x, action):
        h1 = self.enc1(x)
        h2 = self.enc2(x)
        q1 = self.q1(torch.cat([h1, action], dim=-1)).squeeze(-1)
        q2 = self.q2(torch.cat([h2, action], dim=-1)).squeeze(-1)
        return q1, q2


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int, state_shape: tuple, action_dim: int):
        self.capacity    = capacity
        self.states      = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions     = np.zeros((capacity, action_dim),   dtype=np.float32)
        self.rewards     = np.zeros(capacity,                 dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.ptr  = 0
        self.size = 0

    def push(self, state, action, reward, next_state):
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device):
        idx = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.states[idx]).to(device),
            torch.FloatTensor(self.actions[idx]).to(device),
            torch.FloatTensor(self.rewards[idx]).to(device),
            torch.FloatTensor(self.next_states[idx]).to(device),
        )

    def __len__(self):
        return self.size


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


def create_sequences(df, stride: int = STRIDE_EVAL):
    """Windows with given stride → (X, Y). Used for evaluation."""
    mains    = df['aggregate'].values.astype(np.float32)
    app_vals = {a: df[a].values.astype(np.float32) for a in APPLIANCES}
    X, Y = [], []
    for i in range(0, len(mains) - WIN + 1, stride):
        X.append(mains[i: i + WIN])
        mid = i + WIN // 2
        Y.append([app_vals[a][mid] for a in APPLIANCES])
    return (
        np.array(X, dtype=np.float32).reshape(-1, WIN, 1),
        np.array(Y, dtype=np.float32),
    )


def create_transitions(df, stride: int = STRIDE_COLLECT):
    """
    (state, y_target, next_state) triples for the replay buffer.
    next_state = state window shifted forward by `stride` raw timesteps.
    """
    mains    = df['aggregate'].values.astype(np.float32)
    app_vals = {a: df[a].values.astype(np.float32) for a in APPLIANCES}
    X, Y, X_next = [], [], []
    for i in range(0, len(mains) - WIN - stride + 1, stride):
        X.append(mains[i: i + WIN])
        mid = i + WIN // 2
        Y.append([app_vals[a][mid] for a in APPLIANCES])
        X_next.append(mains[i + stride: i + stride + WIN])
    X      = np.array(X,      dtype=np.float32).reshape(-1, WIN, 1)
    Y      = np.array(Y,      dtype=np.float32)
    X_next = np.array(X_next, dtype=np.float32).reshape(-1, WIN, 1)
    return X, Y, X_next


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def compute_rewards(actions_np, y_true_np, prev_actions_np, y_mins, y_ranges):
    """
    Vectorised: (N, n_apps) arrays → (N,) rewards.
    No sparsity term: MAE already penalises both over- and under-prediction.
    """
    pred  = actions_np      * y_ranges + y_mins   # raw Watts
    true  = y_true_np       * y_ranges + y_mins
    prev  = prev_actions_np * y_ranges + y_mins
    mae   = np.abs(pred - true).mean(axis=1)
    trans = np.abs(pred - prev).mean(axis=1)
    return -(ALPHA_MAE * mae + BETA_TRANS * trans) / NORM_POWER


# ---------------------------------------------------------------------------
# Supervised pre-training
# ---------------------------------------------------------------------------

def pretrain_supervised(actor: SACActorLNN, X_tr, Y_tr, device,
                        epochs: int = PRETRAIN_EPOCHS):
    """MSE warm-start: train squashed actor mean toward scaled targets."""
    if epochs <= 0:
        return
    print(f"\n--- Supervised pre-training ({epochs} epochs) ---")
    actor.train()
    opt = torch.optim.Adam(actor.parameters(), lr=LR)
    N   = len(X_tr)
    for ep in range(epochs):
        perm  = np.random.permutation(N)
        total = 0.0; n = 0
        for s in range(0, N, BATCH):
            idx  = perm[s: s + BATCH]
            xb   = torch.FloatTensor(X_tr[idx]).to(device)
            yb   = torch.FloatTensor(Y_tr[idx]).to(device)
            mean, _ = actor._dist(xb)
            pred    = (torch.tanh(mean) + 1.0) / 2.0   # squashed ∈ (0,1)
            loss    = F.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            opt.step()
            total += loss.item(); n += 1
        print(f"  pretrain epoch {ep+1}/{epochs}  MSE={total/n:.6f}")
    print("--- Pre-training complete ---\n")


# ---------------------------------------------------------------------------
# SAC update step
# ---------------------------------------------------------------------------

def sac_update(actor, critic, critic_target, log_alpha,
               actor_opt, critic_opt, alpha_opt,
               states, actions, rewards, next_states):
    """One SAC gradient step. Returns scalar diagnostics."""
    alpha = log_alpha.exp().detach()

    # ── Critic update ─────────────────────────────────────────────────────
    with torch.no_grad():
        next_a, next_lp  = actor.get_action(next_states)
        q1_t, q2_t       = critic_target(next_states, next_a)
        q_next           = torch.min(q1_t, q2_t) - alpha * next_lp
        target_q         = rewards + GAMMA * q_next

    q1, q2      = critic(states, actions)
    critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
    critic_opt.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
    critic_opt.step()

    # ── Actor update ──────────────────────────────────────────────────────
    new_a, log_prob = actor.get_action(states)
    q1_new, q2_new  = critic(states, new_a)
    actor_loss      = (alpha * log_prob - torch.min(q1_new, q2_new)).mean()
    actor_opt.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
    actor_opt.step()

    # ── Temperature update ────────────────────────────────────────────────
    alpha_loss = -(log_alpha * (log_prob.detach() + TARGET_ENTROPY)).mean()
    alpha_opt.zero_grad()
    alpha_loss.backward()
    alpha_opt.step()

    # ── Polyak update for target critic ──────────────────────────────────
    with torch.no_grad():
        for p, pt in zip(critic.parameters(), critic_target.parameters()):
            pt.data.mul_(1.0 - TAU).add_(TAU * p.data)

    return (critic_loss.item(), actor_loss.item(),
            log_prob.mean().item(), log_alpha.exp().item())


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(actor, X, Y_scaled, y_scalers, device):
    """Deterministic policy → per-appliance metrics."""
    actor.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            xb = torch.FloatTensor(X[i: i + BATCH]).to(device)
            a, _ = actor.get_action(xb, deterministic=True)
            all_pred.append(a.cpu().numpy())
            all_true.append(Y_scaled[i: i + BATCH])
    pred_sc = np.concatenate(all_pred)
    true_sc = np.concatenate(all_true)
    results = {}
    for i, app in enumerate(APPLIANCES):
        rp  = y_scalers[i].inverse_transform(pred_sc[:, i:i+1]).flatten()
        rt  = y_scalers[i].inverse_transform(true_sc[:, i:i+1]).flatten()
        thr = THRESHOLDS[app]
        m   = calculate_nilm_metrics(rt, rp, threshold=thr)
        m['TP'] = int(((rt > thr) &  (rp > thr)).sum())
        m['FP'] = int(((rt <= thr) & (rp > thr)).sum())
        m['FN'] = int(((rt > thr) &  (rp <= thr)).sum())
        results[app] = m
    return results


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def collect_transitions(actor, X_tr, Y_tr, y_mins, y_ranges, buffer, device):
    """
    One pass through training data with the stochastic policy.
    Pushes (state, action, reward, next_state) into the replay buffer.
    Returns mean reward for logging.
    """
    actor.eval()
    all_acts = []
    with torch.no_grad():
        for i in range(0, len(X_tr), BATCH):
            xb = torch.FloatTensor(X_tr[i: i + BATCH]).to(device)
            a, _ = actor.get_action(xb, deterministic=False)
            all_acts.append(a.cpu().numpy())
    all_acts  = np.concatenate(all_acts)                   # (N, n_apps)
    prev_acts = np.vstack([np.zeros_like(all_acts[:1]),
                           all_acts[:-1]])                  # shift-by-1 prev
    rewards   = compute_rewards(all_acts, Y_tr, prev_acts, y_mins, y_ranges)
    # X_tr pairs with X_tr_next via index (pre-computed in train_model)
    return all_acts, rewards


def train_model(data_splits, save_dir, hidden_size=64, dt=0.1):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}  hidden={hidden_size}  dt={dt}")
    print(f"SAC: γ={GAMMA}  τ={TAU}  α_init={ALPHA_INIT}  "
          f"target_H={TARGET_ENTROPY}  updates/epoch={UPDATES_PER_EPOCH}")
    print(f"Reward: α_mae={ALPHA_MAE}  β_trans={BETA_TRANS}")

    tr_df = data_splits['train']
    va_df = data_splits['validation']
    te_df = data_splits['test']

    # Replay-buffer transitions (stride=1 for density)
    X_tr, Y_tr, X_tr_next = create_transitions(tr_df, stride=STRIDE_COLLECT)
    # Eval sequences (stride=5 for speed)
    X_va, Y_va = create_sequences(va_df, stride=STRIDE_EVAL)
    X_te, Y_te = create_sequences(te_df, stride=STRIDE_EVAL)

    # ── Scale inputs ─────────────────────────────────────────────────────
    x_scaler  = MinMaxScaler()
    X_tr      = x_scaler.fit_transform(X_tr.reshape(-1, 1)).reshape(X_tr.shape)
    X_tr_next = x_scaler.transform(X_tr_next.reshape(-1, 1)).reshape(X_tr_next.shape)
    X_va      = x_scaler.transform(X_va.reshape(-1, 1)).reshape(X_va.shape)
    X_te      = x_scaler.transform(X_te.reshape(-1, 1)).reshape(X_te.shape)

    # ── Scale outputs per appliance ───────────────────────────────────────
    y_scalers = []
    for i in range(len(APPLIANCES)):
        ys = MinMaxScaler()
        Y_tr[:, i:i+1] = ys.fit_transform(Y_tr[:, i:i+1])
        Y_va[:, i:i+1] = ys.transform(Y_va[:, i:i+1])
        Y_te[:, i:i+1] = ys.transform(Y_te[:, i:i+1])
        y_scalers.append(ys)

    y_mins   = np.array([float(ys.data_min_[0])   for ys in y_scalers])
    y_ranges = np.array([float(ys.data_range_[0]) for ys in y_scalers])

    print(f"\nTrain transitions: {X_tr.shape}  "
          f"Val: {X_va.shape}  Test: {X_te.shape}")

    # ── Models ────────────────────────────────────────────────────────────
    actor   = SACActorLNN(1, hidden_size, len(APPLIANCES), dt).to(device)
    critic  = SACCriticLNN(1, hidden_size, len(APPLIANCES), dt).to(device)
    critic_target = copy.deepcopy(critic).to(device)
    for p in critic_target.parameters():
        p.requires_grad_(False)

    log_alpha = torch.tensor(np.log(ALPHA_INIT), dtype=torch.float32,
                             device=device, requires_grad=True)

    actor_opt  = torch.optim.Adam(actor.parameters(),  lr=LR, eps=1e-5)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=LR, eps=1e-5)
    alpha_opt  = torch.optim.Adam([log_alpha],          lr=LR)

    buffer = ReplayBuffer(REPLAY_CAPACITY, (WIN, 1), len(APPLIANCES))

    n_actor  = sum(p.numel() for p in actor.parameters()  if p.requires_grad)
    n_critic = sum(p.numel() for p in critic.parameters() if p.requires_grad)
    print(f"Actor params: {n_actor:,}   Critic params: {n_critic:,}")

    # ── Supervised pre-training ───────────────────────────────────────────
    pretrain_supervised(actor, X_tr, Y_tr, device, epochs=PRETRAIN_EPOCHS)

    # ── Initial buffer population with pre-trained actor ─────────────────
    print("Populating replay buffer with pre-trained actor...")
    all_acts, rewards = collect_transitions(
        actor, X_tr, Y_tr, y_mins, y_ranges, buffer, device)
    for i in range(len(X_tr)):
        buffer.push(X_tr[i], all_acts[i], rewards[i], X_tr_next[i])
    print(f"  Buffer: {len(buffer):,} transitions  "
          f"mean reward = {rewards.mean():.5f}\n")

    history = {
        'critic_loss': [], 'actor_loss': [], 'alpha': [],
        'mean_reward': [], 'val_metrics': [],
    }
    best_f1    = -1.0
    best_state = None
    counter    = 0

    print("Starting SAC training...\n")

    for epoch in range(EPOCHS):
        # ── Collect new transitions with the current stochastic policy ────
        all_acts, rewards = collect_transitions(
            actor, X_tr, Y_tr, y_mins, y_ranges, buffer, device)
        for i in range(len(X_tr)):
            buffer.push(X_tr[i], all_acts[i], rewards[i], X_tr_next[i])

        # ── Gradient updates ──────────────────────────────────────────────
        actor.train(); critic.train()
        ep_cl = ep_al = ep_lp = ep_alpha = 0.0

        for _ in range(UPDATES_PER_EPOCH):
            states, acts, rews, nxt = buffer.sample(BATCH, device)
            cl, al, lp, alp = sac_update(
                actor, critic, critic_target, log_alpha,
                actor_opt, critic_opt, alpha_opt,
                states, acts, rews, nxt,
            )
            ep_cl += cl; ep_al += al; ep_lp += lp; ep_alpha += alp

        ep_cl    /= UPDATES_PER_EPOCH
        ep_al    /= UPDATES_PER_EPOCH
        ep_lp    /= UPDATES_PER_EPOCH
        ep_alpha /= UPDATES_PER_EPOCH

        # ── Validation ────────────────────────────────────────────────────
        val_metrics = evaluate(actor, X_va, Y_va, y_scalers, device)
        avg_f1  = np.mean([val_metrics[a]['f1']  for a in APPLIANCES])
        avg_mae = np.mean([val_metrics[a]['mae'] for a in APPLIANCES])

        history['critic_loss'].append(ep_cl)
        history['actor_loss'].append(ep_al)
        history['alpha'].append(ep_alpha)
        history['mean_reward'].append(float(rewards.mean()))
        history['val_metrics'].append(val_metrics)

        print(
            f"  Epoch {epoch+1:3d}/{EPOCHS}  "
            f"Qlos={ep_cl:.5f}  πlos={ep_al:.5f}  "
            f"α={ep_alpha:.4f}  R={rewards.mean():.5f}  "
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
            best_state = {k: v.clone() for k, v in actor.state_dict().items()}
            counter    = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print("Training complete.")
    if best_state:
        actor.load_state_dict(best_state)

    # ── Test ──────────────────────────────────────────────────────────────
    test_metrics = evaluate(actor, X_te, Y_te, y_scalers, device)
    print(f"\n{'Appliance':<22} {'F1':>6} {'Prec':>6} {'Rec':>6} "
          f"{'MAE':>7} {'SAE':>7} {'TP':>7} {'FP':>7} {'FN':>7}")
    print("-" * 80)
    for app in APPLIANCES:
        m = test_metrics[app]
        print(f"{app:<22} {m['f1']:>6.4f} {m['precision']:>6.4f} "
              f"{m['recall']:>6.4f} {m['mae']:>7.2f} {m['sae']:>7.4f} "
              f"{m['TP']:>7} {m['FP']:>7} {m['FN']:>7}")

    _plot_results(history, test_metrics, save_dir)
    _save_json(test_metrics, hidden_size, dt, save_dir)
    return test_metrics, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_results(history, test_metrics, save_dir):
    epochs_x = range(1, len(history['critic_loss']) + 1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('LNN-SAC Training Curves', fontsize=13)

    axes[0].plot(epochs_x, history['critic_loss'], color='red')
    axes[0].set_title('Q-network Loss'); axes[0].set_xlabel('Epoch')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_x, history['actor_loss'], color='blue')
    axes[1].set_title('Actor Loss (π)'); axes[1].set_xlabel('Epoch')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs_x, history['alpha'], color='green')
    axes[2].set_title('Temperature (α)'); axes[2].set_xlabel('Epoch')
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(epochs_x, history['mean_reward'], color='purple')
    axes[3].set_title('Mean Reward'); axes[3].set_xlabel('Epoch')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sac_training_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(len(APPLIANCES), 2,
                             figsize=(12, 4 * len(APPLIANCES)))
    fig.suptitle('LNN-SAC UKDALE — Per-Appliance Val Metrics', fontsize=13)
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
        'model': 'LNN-SAC (Soft Actor-Critic)',
        'dataset': f'UKDALE-{DEFAULT_DATASET_DIR}',
        'hyperparams': {
            'WIN': WIN,
            'STRIDE_COLLECT': STRIDE_COLLECT, 'STRIDE_EVAL': STRIDE_EVAL,
            'BATCH': BATCH, 'EPOCHS': EPOCHS, 'PATIENCE': PATIENCE, 'LR': LR,
            'hidden_size': hidden_size, 'dt': dt,
            'GAMMA': GAMMA, 'TAU': TAU,
            'ALPHA_INIT': ALPHA_INIT, 'TARGET_ENTROPY': TARGET_ENTROPY,
            'UPDATES_PER_EPOCH': UPDATES_PER_EPOCH,
            'REPLAY_CAPACITY': REPLAY_CAPACITY,
            'ALPHA_MAE': ALPHA_MAE, 'BETA_TRANS': BETA_TRANS,
            'NORM_POWER': NORM_POWER,
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
    p = argparse.ArgumentParser(description='LNN-SAC NILM on UKDALE CSV dataset')
    p.add_argument('--dataset-dir',  default=DEFAULT_DATASET_DIR,
                   help='Directory containing UKDALE_HF_*.csv files')
    p.add_argument('--hidden-size',  type=int,   default=64)
    p.add_argument('--dt',           type=float, default=0.1)
    p.add_argument('--alpha-mae',    type=float, default=ALPHA_MAE,
                   help='MAE reward weight')
    p.add_argument('--beta-trans',   type=float, default=BETA_TRANS,
                   help='Transition penalty weight')
    p.add_argument('--updates',      type=int,   default=UPDATES_PER_EPOCH,
                   help='Gradient updates per epoch')
    p.add_argument('--pretrain',     type=int,   default=PRETRAIN_EPOCHS,
                   help='Supervised pre-training epochs')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    for split in ('train', 'validation', 'test'):
        p = os.path.join(args.dataset_dir, f'UKDALE_HF_{split}.csv')
        if not os.path.exists(p):
            print(f"Error: {p} not found. Run preprocess_hf.py first.")
            sys.exit(1)

    ALPHA_MAE         = args.alpha_mae
    BETA_TRANS        = args.beta_trans
    UPDATES_PER_EPOCH = args.updates
    PRETRAIN_EPOCHS   = args.pretrain

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir  = f'models/lnn_sac_{timestamp}'

    data_splits = load_data(args.dataset_dir)
    train_model(data_splits, save_dir=save_dir,
                hidden_size=args.hidden_size, dt=args.dt)
    print(f"\nAll outputs saved to {save_dir}/")
