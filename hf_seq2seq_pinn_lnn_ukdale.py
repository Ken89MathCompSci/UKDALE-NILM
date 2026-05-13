"""
HF Seq2Seq PINN-LNN for UKDALE NILM
======================================

Addresses the HF exploitation gaps in hf_pinn_lnn_ukdale.py:

GAP 1 — Seq2Point wastes label density
  Fix: Seq2Seq output — predict ALL WIN appliance power values per window,
       not just the midpoint.  At 6s, one 30-min window contains 299 labelled
       targets per appliance.  The model must learn the full shape of each
       appliance cycle, not just a single point estimate.

GAP 2 — Single-scale dP/dt misses multi-speed events
  Fix: Three rate-of-change channels:
       dP1  (1-step  ~6 s)  — instantaneous switching transients
       dP10 (10-step ~60 s) — appliance phase transitions (wash → heat)
       dP30 (30-step ~3 min)— slow thermal ramps (oven / wash heating)

GAP 3 — Rolling stats only at one scale
  Fix: Short (10-step, 60 s) + long (50-step, 5 min) rolling mean and std,
       giving the model both fast oscillations (fridge compressor) and
       slow background trends.

GAP 4 — CNN compresses sequence, losing transient timing
  Fix: Length-preserving CNN with growing dilations (1→4→16→64).
       No MaxPool — every 6-second timestep remains in the output.
       Receptive field covers ~183 steps (~18 min) without losing resolution.

GAP 5 — Unidirectional LNN: only past context
  Fix: Bidirectional LNN — forward cell (left→right) and backward cell
       (right→left) run in parallel; their hidden states are concatenated
       at every timestep before the appliance heads, giving each prediction
       both past and future context within the window.

GAP 6 — Physics loss only at midpoint
  Fix: Physics constraint applied at every timestep in the window:
       L_phys = mean_t( ReLU( Σ_i p̂_i(t) − P_agg(t) − ε ) )
       This is ~299x more constraints per window than the original.

Architecture:
    Input  (batch, WIN, 8)
         |
    LengthPreservingCNN  →  (batch, WIN, hidden)   [4 dilated layers, no pooling]
         |
    BiLNN: forward cell + backward cell
         |
    LayerNorm(hidden * 2)  at each timestep
         |
    4 × Linear(hidden*2, 1)  applied at each of WIN timesteps
         |
    Output (batch, WIN, 4)

Input channels (8):
    0  aggregate          raw mains power (W)
    1  dP/dt  1-step      ~6 s rate of change   (switching edges)
    2  dP/dt 10-step      ~60 s rate of change  (phase transitions)
    3  dP/dt 30-step      ~3 min rate of change (slow thermal ramps)
    4  rolling mean 10    60 s local baseline
    5  rolling std  10    60 s local variability
    6  rolling mean 50    5 min trend
    7  rolling std  50    5 min variability

Evaluation:
    Overlapping Seq2Seq predictions are averaged into a full-day reconstructed
    trace, which is then evaluated against ground truth.  This avoids double-
    counting timesteps covered by multiple windows.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Source Code'))
from utils import calculate_nilm_metrics


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPOCHS        = 80
PATIENCE      = 20
LR            = 1e-3
BATCH         = 32
WIN           = 299       # ~30-min window (299 × 6 s)
STRIDE        = 10        # stride between windows (training)
STRIDE_TEST   = WIN       # non-overlapping windows for test reconstruction

ROLL_S        = 10        # short rolling window: 10 steps = 60 s
ROLL_L        = 50        # long  rolling window: 50 steps =  5 min

LAMBDA_PHYS   = 0.01
EPSILON_W     = 50.0      # background-load tolerance (W)
WARMUP_EPOCHS = 20        # epochs before physics + BCE losses are added

APPLIANCES = ['dishwasher', 'fridge', 'microwave', 'washing_machine']

THRESHOLDS = {
    'dishwasher':      10.0,
    'fridge':          10.0,
    'microwave':       10.0,
    'washing_machine': 10.0,
}

BCE_LAMBDA = {'dishwasher': 0.5, 'fridge': 0.3, 'microwave': 0.0, 'washing_machine': 0.0}
BCE_ALPHA  = {'dishwasher': 2.0, 'fridge': 2.0, 'microwave': 1.0, 'washing_machine': 1.0}

DATA_DIR = os.path.join(os.path.dirname(__file__), 'dataset')


# ---------------------------------------------------------------------------
# High-frequency feature engineering  (8 channels)
# ---------------------------------------------------------------------------

def _n_step_diff(arr: np.ndarray, n: int) -> np.ndarray:
    """arr[t] - arr[t-n], zero-padded at the start."""
    d       = np.zeros_like(arr)
    d[n:]   = arr[n:] - arr[:-n]
    return d


def compute_hf_features(df: pd.DataFrame) -> np.ndarray:
    """
    Build (N, 8) feature matrix from the 6-second aggregate column.

    Three dP/dt scales capture events at different speeds:
      dP1  — instantaneous switch-on transient (single 6s spike)
      dP10 — appliance phase change over ~60s (dishwasher heat phase)
      dP30 — slow thermal ramp over ~3 min (oven / water heater)

    Two rolling-stat scales separate fast cycles from slow trends:
      Short (60s)  — fridge compressor on/off, microwave burst
      Long  (5min) — washing machine cycle stage, background drift
    """
    agg  = df['aggregate'].values.astype(np.float32)
    d1   = _n_step_diff(agg,  1).astype(np.float32)
    d10  = _n_step_diff(agg, 10).astype(np.float32)
    d30  = _n_step_diff(agg, 30).astype(np.float32)

    s      = pd.Series(agg)
    rm_s   = s.rolling(ROLL_S, min_periods=1).mean().values.astype(np.float32)
    rs_s   = s.rolling(ROLL_S, min_periods=1).std().fillna(0).values.astype(np.float32)
    rm_l   = s.rolling(ROLL_L, min_periods=1).mean().values.astype(np.float32)
    rs_l   = s.rolling(ROLL_L, min_periods=1).std().fillna(0).values.astype(np.float32)

    return np.stack([agg, d1, d10, d30, rm_s, rs_s, rm_l, rs_l], axis=1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(split: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f'UKDALE_HF_{split}.csv')
    df   = pd.read_csv(path, index_col='timestamp', parse_dates=True)
    print(f"  {split:12s}: {len(df):6d} rows")
    return df


def create_sequences(feat: np.ndarray, targets: np.ndarray,
                     stride: int = STRIDE):
    """
    Seq2Seq sliding window.

    Args:
        feat:    (N, 8)       HF feature matrix
        targets: (N, n_apps)  appliance ground truth
        stride:  step between consecutive windows

    Returns:
        X: (M, WIN, 8)
        Y: (M, WIN, n_apps)   — full window of targets, not just midpoint
    """
    X, Y = [], []
    for i in range(0, len(feat) - WIN, stride):
        X.append(feat[i : i + WIN])
        Y.append(targets[i : i + WIN])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


# ---------------------------------------------------------------------------
# Physics Consistency Loss — applied at every timestep
# ---------------------------------------------------------------------------

class PhysicsConsistencyLoss(nn.Module):
    """
    At every timestep t in the window:
        violation(t) = ReLU( Σ_i p̂_i_raw(t) − P_agg_raw(t) − ε )

    L_phys = mean over (batch, timesteps) of violation

    This gives ~299x more physics constraints per window compared to
    the midpoint-only version, strongly enforcing energy conservation
    throughout the appliance cycles.
    """

    def __init__(self, agg_scaler, y_scalers, epsilon_w: float = EPSILON_W):
        super().__init__()
        self.epsilon = epsilon_w
        self.register_buffer('x_min',
            torch.tensor(float(agg_scaler.data_min_[0]),   dtype=torch.float32))
        self.register_buffer('x_range',
            torch.tensor(float(agg_scaler.data_range_[0]), dtype=torch.float32))
        self.register_buffer('y_mins',
            torch.tensor([float(s.data_min_[0])   for s in y_scalers],
                         dtype=torch.float32))
        self.register_buffer('y_ranges',
            torch.tensor([float(s.data_range_[0]) for s in y_scalers],
                         dtype=torch.float32))

    def forward(self, x_scaled: torch.Tensor,
                pred_scaled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_scaled:    (batch, WIN, 8) — channel 0 is scaled aggregate
            pred_scaled: (batch, WIN, n_apps)
        """
        x_raw = x_scaled[:, :, 0] * self.x_range + self.x_min      # (batch, WIN)
        p_raw = pred_scaled * self.y_ranges + self.y_mins            # (batch, WIN, n)
        return F.relu(p_raw.sum(dim=-1) - x_raw - self.epsilon).mean()


# ---------------------------------------------------------------------------
# Length-preserving dilated CNN encoder
# ---------------------------------------------------------------------------

class LengthPreservingCNN(nn.Module):
    """
    Four dilated Conv1d layers, all with 'same' padding (output length = input).
    No MaxPool — every 6-second tick is preserved in the output representation.

    Dilation schedule: 1 → 4 → 16 → 64
    Receptive field:   7 → 23 → 55 → 183 timesteps  (~18 min at 6s)

    This lets the CNN detect both instantaneous transients (d=1) and
    patterns spanning several minutes (d=64) without losing temporal resolution.
    """

    def __init__(self, in_ch: int, hidden: int):
        super().__init__()
        # 'same' padding: p = dilation * (kernel - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch,  hidden, kernel_size=7, padding=3,  dilation=1),
            nn.BatchNorm1d(hidden), nn.GELU(),

            nn.Conv1d(hidden, hidden, kernel_size=5, padding=8,  dilation=4),
            nn.BatchNorm1d(hidden), nn.GELU(),

            nn.Conv1d(hidden, hidden, kernel_size=3, padding=16, dilation=16),
            nn.BatchNorm1d(hidden), nn.GELU(),

            nn.Conv1d(hidden, hidden, kernel_size=3, padding=64, dilation=64),
            nn.BatchNorm1d(hidden), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, WIN, in_ch)
        return self.net(x.permute(0, 2, 1)).permute(0, 2, 1)
        # → (batch, WIN, hidden)   length unchanged


# ---------------------------------------------------------------------------
# Advanced Liquid Cell  (one-step)
# ---------------------------------------------------------------------------

class AdvancedLiquidCell(nn.Module):
    """
    Single-step AdvancedLiquidTimeLayer (same ODE formulation as the original
    PINN-LNN, factored out so both the forward and backward cells can share
    the same class).
    """

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

    def forward(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: (batch, input_size)
            h:   (batch, hidden_size)
        Returns:
            new_h: (batch, hidden_size)
        """
        ip  = self.input_proj(x_t)
        rp  = h @ self.rec_weights
        tb  = F.softplus(self.tau_base).unsqueeze(0)
        tm  = torch.sigmoid(self.tau_mod(x_t))
        tau = (tb * tm).clamp(min=self.dt)
        g   = torch.sigmoid(self.gate(torch.cat([x_t, h], dim=1)))
        dh  = ((-h / tau) + g * torch.tanh(ip + rp)) * self.dt
        return (h + dh).clamp(-10.0, 10.0)


# ---------------------------------------------------------------------------
# Seq2Seq HF Physics-Informed Liquid Network
# ---------------------------------------------------------------------------

class Seq2SeqHFLiquidNet(nn.Module):
    """
    LengthPreservingCNN  +  Bidirectional LNN  +  per-timestep heads.

    Forward cell  (t = 0 … WIN-1): causal context — what load pattern
                                   led up to this moment.
    Backward cell (t = WIN-1 … 0): anti-causal context — how the aggregate
                                   changes after this moment (e.g. appliance
                                   ramping down signals it was near end of cycle).

    Both hidden states are concatenated at each t and normalised before the
    per-appliance linear heads, giving a (batch, WIN, n_apps) prediction.
    """

    def __init__(self, in_ch: int, hidden: int, n_apps: int, dt: float = 0.1):
        super().__init__()
        self.hidden = hidden
        self.n_apps = n_apps

        self.cnn      = LengthPreservingCNN(in_ch, hidden)
        self.fwd_cell = AdvancedLiquidCell(hidden, hidden, dt)
        self.bwd_cell = AdvancedLiquidCell(hidden, hidden, dt)

        self.norm  = nn.LayerNorm(hidden * 2)
        self.heads = nn.ModuleList([
            nn.Linear(hidden * 2, 1) for _ in range(n_apps)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, WIN, in_ch)
        Returns:
            (batch, WIN, n_apps)
        """
        feat  = self.cnn(x)                # (batch, WIN, hidden)
        batch = feat.shape[0]
        T     = feat.shape[1]

        # Forward pass — store hidden state at every step
        h_f  = torch.zeros(batch, self.hidden, device=x.device)
        fwd  = []
        for t in range(T):
            h_f = self.fwd_cell(feat[:, t, :], h_f)
            fwd.append(h_f)

        # Backward pass — iterate in reverse, store per-step state
        h_b  = torch.zeros(batch, self.hidden, device=x.device)
        bwd  = [None] * T
        for t in reversed(range(T)):
            h_b    = self.bwd_cell(feat[:, t, :], h_b)
            bwd[t] = h_b

        # Concatenate bidirectional states, normalise, apply heads per timestep
        out = []
        for t in range(T):
            h_t   = self.norm(torch.cat([fwd[t], bwd[t]], dim=1))  # (batch, hidden*2)
            preds = torch.cat([head(h_t) for head in self.heads], dim=1)  # (batch, n_apps)
            out.append(preds)

        return torch.stack(out, dim=1)     # (batch, WIN, n_apps)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Seq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)   # (M, WIN, 8)
        self.Y = torch.FloatTensor(Y)   # (M, WIN, n_apps)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


# ---------------------------------------------------------------------------
# Trace reconstruction from overlapping windows
# ---------------------------------------------------------------------------

def reconstruct_trace(window_preds: list, n_total: int,
                      stride: int, win: int) -> np.ndarray:
    """
    Average overlapping Seq2Seq window predictions into a single full-day trace.

    Args:
        window_preds: list of (WIN, n_apps) arrays, ordered by window start
        n_total:      total number of timesteps in the day (14400)
        stride, win:  same values used when creating sequences

    Returns:
        (n_total, n_apps) reconstructed trace
    """
    n_apps = window_preds[0].shape[1]
    acc    = np.zeros((n_total, n_apps), dtype=np.float64)
    count  = np.zeros((n_total, 1),     dtype=np.float64)
    for i, pw in enumerate(window_preds):
        s = i * stride
        e = s + win
        if e > n_total:
            break
        acc[s:e]   += pw
        count[s:e] += 1
    count = np.maximum(count, 1)
    return (acc / count).astype(np.float32)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def per_app_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_scalers: list) -> dict:
    """
    Args:
        y_true, y_pred: (N, n_apps) — scaled, from reconstructed trace
        y_scalers: one MinMaxScaler per appliance
    """
    out = {}
    for i, app in enumerate(APPLIANCES):
        raw_t = y_scalers[i].inverse_transform(y_true[:, i:i+1]).flatten()
        raw_p = y_scalers[i].inverse_transform(y_pred[:, i:i+1]).flatten()
        out[app] = calculate_nilm_metrics(raw_t, raw_p,
                                          threshold=THRESHOLDS[app])
    return out


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def train_seq2seq(save_dir: str,
                  hidden:       int   = 64,
                  dt:           float = 0.1,
                  lambda_phys:  float = LAMBDA_PHYS,
                  epsilon_w:    float = EPSILON_W) -> tuple:

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}  |  WIN={WIN}  hidden={hidden}  "
          f"dt={dt}  lambda_phys={lambda_phys}  epsilon={epsilon_w} W")

    # ── Load CSVs ────────────────────────────────────────────────────────────
    print("\nLoading HF CSV data...")
    df_tr = load_csv('train')
    df_va = load_csv('validation')
    df_te = load_csv('test')
    n_tr  = len(df_tr)
    n_va  = len(df_va)
    n_te  = len(df_te)

    # ── 8-channel HF feature matrices (N, 8) ─────────────────────────────────
    feat_tr = compute_hf_features(df_tr)
    feat_va = compute_hf_features(df_va)
    feat_te = compute_hf_features(df_te)

    # ── Appliance targets (N, n_apps) ─────────────────────────────────────────
    tgt_tr = df_tr[APPLIANCES].values.astype(np.float32)
    tgt_va = df_va[APPLIANCES].values.astype(np.float32)
    tgt_te = df_te[APPLIANCES].values.astype(np.float32)

    # ── Create Seq2Seq windows ───────────────────────────────────────────────
    # Training / validation: overlapping (STRIDE=10) for more gradient updates
    # Test: non-overlapping (STRIDE=WIN) for clean reconstruction
    X_tr, Y_tr = create_sequences(feat_tr, tgt_tr, STRIDE)
    X_va, Y_va = create_sequences(feat_va, tgt_va, STRIDE)
    X_te, Y_te = create_sequences(feat_te, tgt_te, WIN)   # non-overlapping

    print(f"\nTrain : {X_tr.shape} -> {Y_tr.shape}  "
          f"({X_tr.shape[0] * WIN:,} total predictions)")
    print(f"Val   : {X_va.shape} -> {Y_va.shape}  "
          f"({X_va.shape[0] * WIN:,} total predictions)")
    print(f"Test  : {X_te.shape} -> {Y_te.shape}  "
          f"[non-overlapping, reconstructed]")

    n_feat = X_tr.shape[-1]   # 8

    # ── Scale each input channel independently ───────────────────────────────
    feat_scalers = []
    for ch in range(n_feat):
        sc = MinMaxScaler()
        X_tr[:, :, ch] = sc.fit_transform(
            X_tr[:, :, ch].reshape(-1, 1)).reshape(-1, WIN)
        X_va[:, :, ch] = sc.transform(
            X_va[:, :, ch].reshape(-1, 1)).reshape(-1, WIN)
        X_te[:, :, ch] = sc.transform(
            X_te[:, :, ch].reshape(-1, 1)).reshape(-1, WIN)
        feat_scalers.append(sc)
    agg_scaler = feat_scalers[0]

    # ── Scale appliance targets  (N, WIN, n_apps) ────────────────────────────
    y_scalers = []
    for i in range(len(APPLIANCES)):
        ys = MinMaxScaler()
        Y_tr[:, :, i] = ys.fit_transform(
            Y_tr[:, :, i].reshape(-1, 1)).reshape(-1, WIN)
        Y_va[:, :, i] = ys.transform(
            Y_va[:, :, i].reshape(-1, 1)).reshape(-1, WIN)
        Y_te[:, :, i] = ys.transform(
            Y_te[:, :, i].reshape(-1, 1)).reshape(-1, WIN)
        y_scalers.append(ys)

    # Scaled on/off thresholds for BCE
    thresholds_scaled = [
        (THRESHOLDS[app] - float(y_scalers[i].data_min_[0]))
        / float(y_scalers[i].data_range_[0])
        for i, app in enumerate(APPLIANCES)
    ]

    # ── DataLoaders ──────────────────────────────────────────────────────────
    tr_ld = torch.utils.data.DataLoader(
        Seq2SeqDataset(X_tr, Y_tr), batch_size=BATCH, shuffle=True,  drop_last=False)
    va_ld = torch.utils.data.DataLoader(
        Seq2SeqDataset(X_va, Y_va), batch_size=BATCH, shuffle=False, drop_last=False)
    te_ld = torch.utils.data.DataLoader(
        Seq2SeqDataset(X_te, Y_te), batch_size=BATCH, shuffle=False, drop_last=False)

    # ── Model + losses ───────────────────────────────────────────────────────
    model = Seq2SeqHFLiquidNet(
        in_ch=n_feat, hidden=hidden, n_apps=len(APPLIANCES), dt=dt
    ).to(device)

    mse_crit  = nn.MSELoss()
    phys_crit = PhysicsConsistencyLoss(agg_scaler, y_scalers, epsilon_w).to(device)

    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode='min', factor=0.5, patience=8, min_lr=1e-5)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    history = {k: [] for k in [
        'train_loss', 'train_mse', 'train_phys',
        'val_loss',   'val_mse',   'val_phys', 'val_metrics',
    ]}
    best_val_mse = float('inf')
    best_state   = None
    counter      = 0

    print(f"\nStarting HF Seq2Seq PINN-LNN training ({len(APPLIANCES)} appliances)...")

    for epoch in range(EPOCHS):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        ep_mse = ep_phys = ep_tot = 0.0
        pbar = tqdm(tr_ld, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)   # (B,WIN,8), (B,WIN,n)
            opt.zero_grad()
            pred = model(xb)                          # (B, WIN, n_apps)

            l_mse  = mse_crit(pred, yb)
            l_phys = phys_crit(xb, pred)              # applied at all timesteps

            if epoch < WARMUP_EPOCHS:
                loss = l_mse
            else:
                l_bce = torch.tensor(0.0, device=device)
                for i, app in enumerate(APPLIANCES):
                    if BCE_LAMBDA[app] > 0:
                        # pred[:, :, i]: (B, WIN) — predictions at all timesteps
                        p_i   = pred[:, :, i].reshape(-1).clamp(1e-7, 1 - 1e-7)
                        thr_s = thresholds_scaled[i]
                        y_bin = (yb[:, :, i].reshape(-1) > thr_s).float()
                        w     = torch.where(y_bin == 1,
                                            torch.full_like(y_bin, BCE_ALPHA[app]),
                                            torch.ones_like(y_bin))
                        l_bce = l_bce + BCE_LAMBDA[app] * F.binary_cross_entropy(
                                    p_i, y_bin, weight=w)
                loss = l_mse + lambda_phys * l_phys + l_bce

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_mse  += l_mse.item()
            ep_phys += l_phys.item()
            ep_tot  += loss.item()
            pbar.set_postfix({'mse': f'{l_mse.item():.5f}',
                              'phys': f'{l_phys.item():.5f}'})

        history['train_mse'].append(ep_mse  / len(tr_ld))
        history['train_phys'].append(ep_phys / len(tr_ld))
        history['train_loss'].append(ep_tot  / len(tr_ld))

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        vl_mse = vl_phys = vl_tot = 0.0
        va_window_preds = []
        va_window_trues = []

        with torch.no_grad():
            for xb, yb in va_ld:
                xb, yb = xb.to(device), yb.to(device)
                pred   = model(xb)
                l_mse  = mse_crit(pred, yb)
                l_phys = phys_crit(xb, pred)
                vl_mse  += l_mse.item()
                vl_phys += l_phys.item()
                vl_tot  += (l_mse + lambda_phys * l_phys).item()
                # Collect per-window predictions for reconstruction
                for b in range(pred.shape[0]):
                    va_window_preds.append(pred[b].cpu().numpy())    # (WIN, n)
                    va_window_trues.append(yb[b].cpu().numpy())

        avg_va_mse  = vl_mse  / len(va_ld)
        avg_va_phys = vl_phys / len(va_ld)
        avg_va_tot  = vl_tot  / len(va_ld)
        history['val_mse'].append(avg_va_mse)
        history['val_phys'].append(avg_va_phys)
        history['val_loss'].append(avg_va_tot)
        sched.step(avg_va_mse)

        # Reconstruct full-day trace and compute metrics
        pred_trace = reconstruct_trace(va_window_preds, n_va, STRIDE, WIN)
        true_trace = reconstruct_trace(va_window_trues, n_va, STRIDE, WIN)

        vm      = per_app_metrics(true_trace, pred_trace, y_scalers)
        history['val_metrics'].append(vm)

        avg_f1  = np.mean([vm[a]['f1']  for a in APPLIANCES])
        avg_mae = np.mean([vm[a]['mae'] for a in APPLIANCES])

        print(
            f"  Epoch {epoch+1:3d}/{EPOCHS}  "
            f"train={history['train_loss'][-1]:.5f} "
            f"(mse={history['train_mse'][-1]:.5f} "
            f"phys={history['train_phys'][-1]:.5f})  "
            f"val={avg_va_tot:.5f} "
            f"(mse={avg_va_mse:.5f} phys={avg_va_phys:.5f})  "
            f"avgF1={avg_f1:.4f}  avgMAE={avg_mae:.2f}  "
            f"lr={opt.param_groups[0]['lr']:.2e}"
        )
        for app in APPLIANCES:
            m = vm[app]
            print(f"    {app:<22}  F1={m['f1']:.4f}  P={m['precision']:.4f}  "
                  f"R={m['recall']:.4f}  MAE={m['mae']:.2f}  SAE={m['sae']:.4f}")

        if avg_va_mse < best_val_mse:
            best_val_mse = avg_va_mse
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            counter      = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print("\nTraining complete.")

    # ── Test evaluation ───────────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    te_window_preds = []
    te_window_trues = []
    with torch.no_grad():
        for xb, yb in te_ld:
            pred = model(xb.to(device))
            for b in range(pred.shape[0]):
                te_window_preds.append(pred[b].cpu().numpy())
                te_window_trues.append(yb[b].cpu().numpy())

    # Non-overlapping reconstruction for clean test evaluation
    pred_trace_te = reconstruct_trace(te_window_preds, n_te, WIN, WIN)
    true_trace_te = reconstruct_trace(te_window_trues, n_te, WIN, WIN)
    test_metrics  = per_app_metrics(true_trace_te, pred_trace_te, y_scalers)

    print(f"\n{'Appliance':<24} {'F1':>8} {'Precision':>10} "
          f"{'Recall':>8} {'MAE':>8} {'SAE':>8}")
    print("-" * 72)
    for app in APPLIANCES:
        m = test_metrics[app]
        print(f"{app:<24} {m['f1']:>8.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['mae']:>8.2f} {m['sae']:>8.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_loss(history, save_dir)
    _plot_per_appliance(history, test_metrics, save_dir)
    _plot_trace(true_trace_te, pred_trace_te, y_scalers, save_dir)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    cfg = {
        'dataset':      'UKDALE HF 6s',
        'model':        'Seq2SeqHFLiquidNet',
        'description':  'LengthPreservingCNN + BiLNN + Seq2Seq + Physics(all t)',
        'hf_features': [
            'aggregate',
            'dP_dt_1step (6s)',
            'dP_dt_10step (60s)',
            'dP_dt_30step (3min)',
            'rolling_mean_10 (60s)',
            'rolling_std_10 (60s)',
            'rolling_mean_50 (5min)',
            'rolling_std_50 (5min)',
        ],
        'architecture': {
            'cnn_dilations':  [1, 4, 16, 64],
            'cnn_receptive_field_steps': 183,
            'cnn_receptive_field_seconds': 183 * 6,
            'lnn_direction': 'bidirectional',
            'output': 'seq2seq (all WIN timesteps)',
            'physics_constraint': 'every timestep',
        },
        'window_size': WIN,
        'stride_train': STRIDE,
        'stride_test':  WIN,
        'model_params': {
            'in_ch': n_feat, 'hidden': hidden,
            'n_apps': len(APPLIANCES), 'dt': dt,
        },
        'train_params': {
            'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE,
            'lambda_phys': lambda_phys, 'epsilon_w': epsilon_w,
            'warmup_epochs': WARMUP_EPOCHS,
        },
        'test_metrics': {
            app: {k: float(v) for k, v in m.items()}
            for app, m in test_metrics.items()
        },
    }
    with open(os.path.join(save_dir, 'hf_seq2seq_results.json'),
              'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=4)
    print(f"\nResults saved to {save_dir}")

    return test_metrics, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_loss(history: dict, save_dir: str) -> None:
    ep = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(15, 4))
    for idx, (tr_k, va_k, title) in enumerate([
        ('train_loss',  'val_loss',  'Total Loss'),
        ('train_mse',   'val_mse',   'MSE Loss'),
        ('train_phys',  'val_phys',  'Physics Loss (all t)'),
    ]):
        plt.subplot(1, 3, idx + 1)
        plt.plot(ep, history[tr_k], label='Train', color='steelblue')
        plt.plot(ep, history[va_k], label='Val',   color='tomato')
        plt.title(title); plt.xlabel('Epoch')
        plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'hf_seq2seq_loss.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def _plot_per_appliance(history: dict, test_metrics: dict,
                        save_dir: str) -> None:
    ep  = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(len(APPLIANCES), 2,
                             figsize=(12, 4 * len(APPLIANCES)))
    fig.suptitle('HF Seq2Seq PINN-LNN — Per-Appliance Validation Metrics',
                 fontsize=13)
    for row, app in enumerate(APPLIANCES):
        f1s  = [m[app]['f1']  for m in history['val_metrics']]
        maes = [m[app]['mae'] for m in history['val_metrics']]
        axes[row][0].plot(ep, f1s, color='steelblue', linewidth=1.5)
        axes[row][0].axhline(test_metrics[app]['f1'], color='green',
                             linestyle='--', linewidth=1.2, label='Test F1')
        axes[row][0].set_title(f'{app} — F1')
        axes[row][0].set_xlabel('Epoch'); axes[row][0].legend()
        axes[row][0].grid(alpha=0.3)
        axes[row][1].plot(ep, maes, color='tomato', linewidth=1.5)
        axes[row][1].axhline(test_metrics[app]['mae'], color='green',
                             linestyle='--', linewidth=1.2, label='Test MAE')
        axes[row][1].set_title(f'{app} — MAE (W)')
        axes[row][1].set_xlabel('Epoch'); axes[row][1].legend()
        axes[row][1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'hf_seq2seq_per_appliance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def _plot_trace(true_trace: np.ndarray, pred_trace: np.ndarray,
                y_scalers: list, save_dir: str,
                n_steps: int = 600) -> None:
    """Plot a 600-step (~1h) segment of the reconstructed test trace."""
    fig, axes = plt.subplots(len(APPLIANCES), 1,
                             figsize=(14, 3 * len(APPLIANCES)))
    fig.suptitle('HF Seq2Seq — Reconstructed Test Trace (first 1 hour)', fontsize=12)
    t = np.arange(n_steps) * 6 / 60   # minutes
    for row, (app, ax) in enumerate(zip(APPLIANCES, axes)):
        i      = APPLIANCES.index(app)
        raw_t  = y_scalers[i].inverse_transform(
                     true_trace[:n_steps, i:i+1]).flatten()
        raw_p  = y_scalers[i].inverse_transform(
                     pred_trace[:n_steps, i:i+1]).flatten()
        ax.plot(t, raw_t, label='Ground truth', color='steelblue',
                linewidth=1.0, alpha=0.8)
        ax.plot(t, raw_p, label='Prediction',   color='tomato',
                linewidth=1.0, alpha=0.8)
        ax.set_title(app); ax.set_ylabel('Power (W)')
        ax.legend(loc='upper right'); ax.grid(alpha=0.3)
    axes[-1].set_xlabel('Time (min)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'hf_seq2seq_trace.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    for split in ['train', 'validation', 'test']:
        p = os.path.join(DATA_DIR, f'UKDALE_HF_{split}.csv')
        if not os.path.exists(p):
            print(f"Error: {p} not found. Run preprocess_hf.py first.")
            sys.exit(1)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir  = f'models/hf_seq2seq_pinn_lnn_{timestamp}'

    train_seq2seq(
        save_dir     = save_dir,
        hidden       = 64,
        dt           = 0.1,
        lambda_phys  = LAMBDA_PHYS,
        epsilon_w    = EPSILON_W,
    )
