"""
Fixed PINN-LNN for UKDALE  (drop-in replacement for test_pinn_lnn_ukdale_specific_splits.py)
==============================================================================================

Fixes applied (see analysis for full rationale):

  FIX 1  Output sigmoid on power heads
         Original: raw Linear output clamped to (1e-7, 1-1e-7) before BCE.
         Fixed:    sigmoid applied to each head; physics inverse-scale is correct.

  FIX 2  Seq2Seq output — predict all WIN timesteps, not just the midpoint
         Original: Y shape (N, n_apps) — one prediction per 100-step window.
         Fixed:    Y shape (N, WIN, n_apps) — 100x more gradient signal per batch.

  FIX 3  Physics loss over every timestep
         Original: x_mid = xb[:, WIN//2, 0] — one sample per batch.
         Fixed:    mean over (batch, t) matching Seq2Seq output.

  FIX 4  WIN increased to 299 steps (~30 min at 6 s)
         Original: WIN=100 (10 min) — shorter than a dishwasher cycle (13.7 min avg).
         Fixed:    WIN=299 covers fridge compressor cycles (24.6 min avg).

  FIX 5  Multi-channel input — 8 features instead of 1
         Added: median filter, EMA smooth, residual, delta(1), delta_smooth(1),
                rolling mean(10), rolling std(10).
         Captures switching edges and noise separately from baseline drift.

  FIX 6  StandardScaler (Z-score) for inputs instead of MinMaxScaler
         Original: MinMaxScaler — 6257W spike compresses all other variation to ~7% of range.
         Fixed:    StandardScaler — outliers expressed in sigma units, not clipped to [0,1].

  FIX 7  Bidirectional LNN — forward + backward LiquidCell
         Original: forward-only; midpoint prediction ignores the right half of the window.
         Fixed:    both directions concatenated at each timestep.

  FIX 8  BCE fix — BCEWithLogits on raw logits with pos_weight
         Original: binary_cross_entropy on clamped output, microwave/washer dryer disabled.
         Fixed:    binary_cross_entropy_with_logits on separate event heads,
                   pos_weight per appliance from training ON fraction, all appliances enabled.

  FIX 9  Washer dryer threshold: 0.5 W -> adaptive per-split
         Original: 0.5 W effectively labels everything as ON.
         Fixed:    standby + 20 W floor, computed from each split's nonzero p5.

Architecture after fixes:

    Input (batch, WIN, 8)
         |
    LengthPreservingCNN  (dilations 1, 4, 16, 64  RF = 183 steps)
         |
    Bidirectional LiquidCell  (fwd + bwd, concat -> hidden*2)
         |
    LayerNorm(hidden*2)
         |
    +-----------------+-----------------+
    Power heads                   Event heads
    n_apps x Linear(h*2,1)+sig   n_apps x Linear(h*2,1)
    (batch, WIN, n_apps)          (batch, WIN, n_apps)

Loss:  L = L_MSE + lambda_phys * L_phys_time + lambda_event * L_event
Warmup: MSE-only for first WARMUP_EPOCHS epochs, then add physics + event losses.
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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Source Code'))
from utils import calculate_nilm_metrics


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPOCHS        = 80
PATIENCE      = 20
LR            = 1e-3
BATCH         = 32
WIN           = 299     # FIX 4: was 100 (10 min); 299 = ~30 min at 6 s
STRIDE        = 10      # training stride; test uses WIN (non-overlapping)

MEDFILT_K     = 5       # median filter kernel (steps)
EMA_SPAN      = 10      # EMA span (steps)
ROLL_K        = 10      # rolling stats window (steps)

LAMBDA_PHYS   = 0.01
LAMBDA_EVENT  = 0.05
EPSILON_W     = 50.0
WARMUP_EPOCHS = 20

# FIX 9: replaced hardcoded THRESHOLDS with adaptive per-split computation
THRESHOLD_DELTA   = 20.0   # W above standby = definitely ON
THRESHOLD_LOW_PCT = 0.05   # p5 of nonzero readings = standby estimate
THRESHOLD_MIN     = 10.0   # floor (W)
# Bimodal fix: 2-means on nonzero readings.  If the high cluster center is
# more than BIMODAL_RATIO x the low cluster center, the appliance has a true
# standby+operating separation → use p5+delta.  Otherwise it is a cycling
# appliance (fridge: 0W↔110W, single cluster) → use THRESHOLD_MIN.
BIMODAL_RATIO     = 3.0   # high_center > low_center * (1 + ratio) → bimodal
KMEANS_MIN_N      = 50    # fall back to p5+delta if fewer nonzero samples

POS_WEIGHT_CLAMP = (1.0, 50.0)

# appliance names match the CSV column names
APPLIANCES  = ['dishwasher', 'fridge', 'microwave', 'washing_machine']
AGG_COL     = 'aggregate'

DATA_DIR = os.path.join(os.path.dirname(__file__), 'dataset')


# ---------------------------------------------------------------------------
# FIX 9: Adaptive per-split ON/OFF thresholds
# ---------------------------------------------------------------------------

def compute_adaptive_thresholds(df: pd.DataFrame) -> dict:
    """
    2-means bimodal-aware adaptive threshold per appliance.

    Fit KMeans(k=2) on nonzero readings.  Sort cluster centers low/high.
    If high_center > low_center * (1 + BIMODAL_RATIO), the distribution is
    bimodal (standby cluster + operating cluster) → threshold = p5 + delta
    sits between them.

    If the two centers are close (unimodal), the appliance cycles between
    exactly 0W and one operating level (fridge) — any nonzero draw is ON →
    threshold = THRESHOLD_MIN.

    Examples:
      Fridge H5  : nonzero all ~106-200W → centers [120W, 165W]
                   165 < 120*(1+3)=480 → unimodal → 10W ✓
      Dishwasher H5: nonzero 57W phases + 1500W cycle
                   centers ≈ [70W, 1400W], 1400 > 70*4=280 → bimodal → 77W ✓
      Microwave H5 : standby 25W + cooking 800W+
                   centers ≈ [25W, 820W], 820 > 25*4=100 → bimodal → 68W ✓
    """
    thresholds = {}
    for app in APPLIANCES:
        col     = df[app]
        nonzero = col[col > 0].values
        if len(nonzero) == 0:
            thresholds[app] = THRESHOLD_MIN
            continue

        p5 = float(np.percentile(nonzero, 5))

        if len(nonzero) >= KMEANS_MIN_N:
            km  = KMeans(n_clusters=2, n_init=10, random_state=42)
            km.fit(nonzero.reshape(-1, 1))
            low, high = sorted(km.cluster_centers_.flatten())
            bimodal = high > low * (1.0 + BIMODAL_RATIO)
        else:
            bimodal = True  # too few samples to cluster; trust p5+delta

        if bimodal:
            thresholds[app] = max(p5 + THRESHOLD_DELTA, THRESHOLD_MIN)
        else:
            # Unimodal cycling appliance: ON = any nonzero draw
            thresholds[app] = THRESHOLD_MIN
    return thresholds


# ---------------------------------------------------------------------------
# FIX 5 + FIX 6: Multi-channel feature extraction
# ---------------------------------------------------------------------------

def _median_filter(arr: np.ndarray, k: int) -> np.ndarray:
    return (pd.Series(arr)
            .rolling(k, center=True, min_periods=1)
            .median()
            .values.astype(np.float32))


def _ema_filter(arr: np.ndarray, span: int) -> np.ndarray:
    return (pd.Series(arr)
            .ewm(span=span, adjust=False)
            .mean()
            .values.astype(np.float32))


def _n_step_diff(arr: np.ndarray, n: int) -> np.ndarray:
    d     = np.zeros_like(arr)
    d[n:] = arr[n:] - arr[:-n]
    return d


def compute_features(mains: np.ndarray) -> np.ndarray:
    """
    8-channel feature matrix from raw mains signal.

      0  raw mains
      1  median filtered        (impulse noise removed)
      2  EMA smoothed           (slow drift removed)
      3  residual               (raw - smooth, edge energy)
      4  delta raw 1-step       (raw switching edge)
      5  delta smooth 1-step    (denoised switching edge)
      6  rolling mean 10        (local baseline on smooth)
      7  rolling std  10        (local variability on smooth)
    """
    raw    = mains.astype(np.float32)
    med    = _median_filter(raw, MEDFILT_K)
    smooth = _ema_filter(med, EMA_SPAN)
    resid  = (raw - smooth).astype(np.float32)

    d_raw_1    = _n_step_diff(raw,    1)
    d_smooth_1 = _n_step_diff(smooth, 1)

    s    = pd.Series(smooth)
    rm   = s.rolling(ROLL_K, min_periods=1).mean().values.astype(np.float32)
    rs   = s.rolling(ROLL_K, min_periods=1).std().fillna(0).values.astype(np.float32)

    return np.stack([raw, med, smooth, resid,
                     d_raw_1, d_smooth_1, rm, rs], axis=1)  # (N, 8)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> dict:
    print("Loading UKDALE CSV data ...")
    file_map = {
        'train': 'UKDALE_HF_train.csv',
        'val':   'UKDALE_HF_validation.csv',
        'test':  'UKDALE_HF_test.csv',
    }
    splits = {}
    for name, fname in file_map.items():
        path = os.path.join(DATA_DIR, fname)
        splits[name] = pd.read_csv(path, index_col='timestamp', parse_dates=True)
    for name, df in splits.items():
        print(f"  {name:6s}: {len(df):6d} rows  "
              f"{df.index.min().date()} -> {df.index.max().date()}")
    print(f"  Columns: {list(splits['train'].columns)}")
    return splits


# FIX 2: Seq2Seq — targets at every timestep, not just midpoint
def create_sequences(df: pd.DataFrame, thresholds: dict, stride: int):
    """
    Returns:
        X   (M, WIN, 8)         — multi-channel feature windows
        Y   (M, WIN, n_apps)    — appliance values at every timestep  [FIX 2]
    """
    feat = compute_features(df[AGG_COL].values)
    app_arrs = {app: df[app].values.astype(np.float32) for app in APPLIANCES}
    N = len(feat)

    X, Y = [], []
    for i in range(0, N - WIN, stride):
        X.append(feat[i : i + WIN])
        Y.append(np.stack([app_arrs[app][i : i + WIN]
                           for app in APPLIANCES], axis=1))  # (WIN, n_apps)
    return (np.array(X, dtype=np.float32),   # (M, WIN, 8)
            np.array(Y, dtype=np.float32))   # (M, WIN, n_apps)


# ---------------------------------------------------------------------------
# Length-preserving dilated CNN  (same as hf_seq2seq script)
# ---------------------------------------------------------------------------

class LengthPreservingCNN(nn.Module):
    """
    4 dilated Conv1d layers with 'same' padding — output length = input length.
    Dilations 1, 4, 16, 64 give RF = 183 steps (~18 min at 6 s).
    No MaxPool so temporal alignment is preserved for Seq2Seq targets.
    """

    def __init__(self, in_ch: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch,   hidden, kernel_size=7, padding=3,  dilation=1),
            nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Conv1d(hidden,  hidden, kernel_size=5, padding=8,  dilation=4),
            nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Conv1d(hidden,  hidden, kernel_size=3, padding=16, dilation=16),
            nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Conv1d(hidden,  hidden, kernel_size=3, padding=64, dilation=64),
            nn.BatchNorm1d(hidden), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.permute(0, 2, 1)).permute(0, 2, 1)


# ---------------------------------------------------------------------------
# FIX 7: Bidirectional Liquid Cell
# ---------------------------------------------------------------------------

class LiquidCell(nn.Module):
    """Single-step LNN cell — same AdvancedLiquidCell as original, now used
    in both forward and backward directions."""

    def __init__(self, input_size: int, hidden_size: int, dt: float = 0.1):
        super().__init__()
        self.dt          = dt
        self.hidden_size = hidden_size
        self.input_proj  = nn.Linear(input_size, hidden_size)
        self.tau_base    = nn.Parameter(torch.ones(hidden_size))
        self.tau_mod     = nn.Linear(input_size, hidden_size)
        self.rec_weights = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.rec_weights)
        self.gate        = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        ip  = self.input_proj(x_t)
        rp  = h @ self.rec_weights
        tb  = F.softplus(self.tau_base).unsqueeze(0)
        tm  = torch.sigmoid(self.tau_mod(x_t))
        tau = (tb * tm).clamp(min=self.dt)
        g   = torch.sigmoid(self.gate(torch.cat([x_t, h], dim=1)))
        dh  = ((-h / tau) + g * torch.tanh(ip + rp)) * self.dt
        return (h + dh).clamp(-10.0, 10.0)


# ---------------------------------------------------------------------------
# Fixed model
# ---------------------------------------------------------------------------

class FixedPINNLNN(nn.Module):
    """
    LengthPreservingCNN  ->  Bidirectional LNN  ->  power + event heads.

    Power heads:  sigmoid(Linear(hidden*2, 1))   [FIX 1 — was raw Linear]
    Event heads:  Linear(hidden*2, 1)             [logits for BCEWithLogits, FIX 8]
    Output:       (power_pred, event_logits), both (batch, WIN, n_apps)  [FIX 2]
    """

    def __init__(self, in_ch: int, hidden: int, n_apps: int, dt: float = 0.1):
        super().__init__()
        self.hidden = hidden
        self.n_apps = n_apps

        self.cnn      = LengthPreservingCNN(in_ch, hidden)
        self.fwd_cell = LiquidCell(hidden, hidden, dt)
        self.bwd_cell = LiquidCell(hidden, hidden, dt)   # FIX 7

        self.norm        = nn.LayerNorm(hidden * 2)
        self.power_heads = nn.ModuleList([nn.Linear(hidden * 2, 1) for _ in range(n_apps)])
        self.event_heads = nn.ModuleList([nn.Linear(hidden * 2, 1) for _ in range(n_apps)])

    def forward(self, x: torch.Tensor):
        """
        x: (batch, WIN, in_ch)
        Returns:
            power_pred:   (batch, WIN, n_apps)  values in [0,1]  [FIX 1]
            event_logits: (batch, WIN, n_apps)  raw logits       [FIX 8]
        """
        feat  = self.cnn(x)
        batch, T, _ = feat.shape

        # Forward pass
        h_f = torch.zeros(batch, self.hidden, device=x.device)
        fwd = []
        for t in range(T):
            h_f = self.fwd_cell(feat[:, t, :], h_f)
            fwd.append(h_f)

        # Backward pass  [FIX 7]
        h_b  = torch.zeros(batch, self.hidden, device=x.device)
        bwd  = [None] * T
        for t in reversed(range(T)):
            h_b    = self.bwd_cell(feat[:, t, :], h_b)
            bwd[t] = h_b

        power_list, event_list = [], []
        for t in range(T):
            h_t = self.norm(torch.cat([fwd[t], bwd[t]], dim=1))
            # FIX 1: sigmoid on power heads
            power_list.append(
                torch.cat([torch.sigmoid(head(h_t)) for head in self.power_heads], dim=1))
            event_list.append(
                torch.cat([head(h_t) for head in self.event_heads], dim=1))

        return torch.stack(power_list, dim=1), torch.stack(event_list, dim=1)


# ---------------------------------------------------------------------------
# FIX 3: Physics loss over all timesteps
# ---------------------------------------------------------------------------

class PhysicsConsistencyLoss(nn.Module):
    """
    One-sided energy conservation at every timestep  [FIX 3 — was midpoint only].
    L_phys = mean_{batch,t}( ReLU( sum_i(p_hat_raw_i(t)) - P_agg_raw(t) - eps ) )

    Input channel 0 = raw mains, Z-score standardised  [FIX 6].
    Inverse transform: x_raw = x_z * scale + mean
    """

    def __init__(self, agg_scaler: StandardScaler,
                 y_scalers: list, epsilon_w: float = EPSILON_W):
        super().__init__()
        self.epsilon = epsilon_w
        self.register_buffer('x_mean',
            torch.tensor(float(agg_scaler.mean_[0]),  dtype=torch.float32))
        self.register_buffer('x_scale',
            torch.tensor(float(agg_scaler.scale_[0]), dtype=torch.float32))
        self.register_buffer('y_mins',
            torch.tensor([float(s.data_min_[0])   for s in y_scalers], dtype=torch.float32))
        self.register_buffer('y_ranges',
            torch.tensor([float(s.data_range_[0]) for s in y_scalers], dtype=torch.float32))

    def forward(self, x_z: torch.Tensor, power_pred: torch.Tensor) -> torch.Tensor:
        x_raw = x_z[:, :, 0] * self.x_scale + self.x_mean   # (batch, WIN)
        p_raw = power_pred * self.y_ranges + self.y_mins      # (batch, WIN, n_apps)
        return F.relu(p_raw.sum(dim=-1) - x_raw - self.epsilon).mean()


# ---------------------------------------------------------------------------
# FIX 8: Event detection loss — BCEWithLogits + pos_weight for all appliances
# ---------------------------------------------------------------------------

class EventDetectionLoss(nn.Module):
    """
    Original: BCE on clamped output, disabled for microwave and washer dryer.
    Fixed:    BCEWithLogits on raw logits, per-appliance pos_weight,
              all appliances enabled.
    """

    def __init__(self, thresholds_scaled: list, pos_weights: list):
        super().__init__()
        self.n_apps = len(thresholds_scaled)
        self.register_buffer('thresholds',
            torch.tensor(thresholds_scaled, dtype=torch.float32))
        self.register_buffer('pos_weights',
            torch.tensor(pos_weights, dtype=torch.float32))

    def forward(self, event_logits: torch.Tensor,
                y_scaled: torch.Tensor) -> torch.Tensor:
        total = torch.zeros(1, device=event_logits.device)
        for i in range(self.n_apps):
            logit_i = event_logits[:, :, i]
            y_on_i  = (y_scaled[:, :, i] > self.thresholds[i]).float()
            total   = total + F.binary_cross_entropy_with_logits(
                logit_i, y_on_i, pos_weight=self.pos_weights[i:i+1])
        return total / self.n_apps


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Seq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


# ---------------------------------------------------------------------------
# Trace reconstruction (needed because windows overlap)
# ---------------------------------------------------------------------------

def reconstruct_trace(window_preds: list, n_total: int,
                      stride: int, win: int) -> np.ndarray:
    n_apps = window_preds[0].shape[1]
    acc    = np.zeros((n_total, n_apps), dtype=np.float64)
    count  = np.zeros((n_total, 1),     dtype=np.float64)
    for idx, pw in enumerate(window_preds):
        s, e = idx * stride, idx * stride + win
        if e > n_total:
            break
        acc[s:e]   += pw
        count[s:e] += 1
    return (acc / np.maximum(count, 1)).astype(np.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def per_app_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_scalers: list, thresholds: dict) -> dict:
    out = {}
    for i, app in enumerate(APPLIANCES):
        raw_t = y_scalers[i].inverse_transform(y_true[:, i:i+1]).flatten()
        raw_p = y_scalers[i].inverse_transform(y_pred[:, i:i+1]).flatten()
        out[app] = calculate_nilm_metrics(raw_t, raw_p, threshold=thresholds[app])
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(data_dict: dict, save_dir: str,
          hidden:       int   = 64,
          dt:           float = 0.1,
          lambda_phys:  float = LAMBDA_PHYS,
          lambda_event: float = LAMBDA_EVENT,
          epsilon_w:    float = EPSILON_W) -> tuple:

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}  |  WIN={WIN}  hidden={hidden}  dt={dt}")
    print(f"Fixes applied: sigmoid heads, Seq2Seq, BiLNN, physics@all-t, "
          f"Z-score, 8-ch features, BCEWithLogits, adaptive thresholds\n")

    df_tr = data_dict['train']
    df_va = data_dict['val']
    df_te = data_dict['test']
    n_tr, n_va, n_te = len(df_tr), len(df_va), len(df_te)

    # FIX 9: adaptive thresholds per split
    tr_thresholds = compute_adaptive_thresholds(df_tr)
    va_thresholds = compute_adaptive_thresholds(df_va)
    te_thresholds = compute_adaptive_thresholds(df_te)
    print("  Adaptive thresholds (W):")
    print(f"  {'Appliance':<16} {'Train':>8} {'Val':>8} {'Test':>8}")
    for app in APPLIANCES:
        print(f"  {app:<16} {tr_thresholds[app]:>8.1f} "
              f"{va_thresholds[app]:>8.1f} {te_thresholds[app]:>8.1f}")

    # Sequences  [FIX 2]
    print("\nCreating sequences ...")
    X_tr, Y_tr = create_sequences(df_tr, tr_thresholds, STRIDE)
    X_va, Y_va = create_sequences(df_va, va_thresholds, STRIDE)
    X_te, Y_te = create_sequences(df_te, te_thresholds, WIN)
    n_feat = X_tr.shape[2]   # 8
    print(f"  Train : {X_tr.shape} -> {Y_tr.shape}  "
          f"({X_tr.shape[0]*WIN:,} predictions)")
    print(f"  Val   : {X_va.shape} -> {Y_va.shape}")
    print(f"  Test  : {X_te.shape} -> {Y_te.shape}  [non-overlapping]")

    # FIX 6: StandardScaler per input channel
    feat_scalers = []
    for ch in range(n_feat):
        sc = StandardScaler()
        X_tr[:, :, ch] = sc.fit_transform(X_tr[:, :, ch].reshape(-1, 1)).reshape(-1, WIN)
        X_va[:, :, ch] = sc.transform(    X_va[:, :, ch].reshape(-1, 1)).reshape(-1, WIN)
        X_te[:, :, ch] = sc.transform(    X_te[:, :, ch].reshape(-1, 1)).reshape(-1, WIN)
        feat_scalers.append(sc)
    agg_scaler = feat_scalers[0]
    print(f"\n  Agg Z-score: mean={agg_scaler.mean_[0]:.1f} W  "
          f"std={agg_scaler.scale_[0]:.1f} W")

    # MinMaxScaler for targets
    y_scalers = []
    for i in range(len(APPLIANCES)):
        ys = MinMaxScaler()
        Y_tr[:, :, i] = ys.fit_transform(Y_tr[:, :, i].reshape(-1, 1)).reshape(-1, WIN)
        Y_va[:, :, i] = ys.transform(    Y_va[:, :, i].reshape(-1, 1)).reshape(-1, WIN)
        Y_te[:, :, i] = ys.transform(    Y_te[:, :, i].reshape(-1, 1)).reshape(-1, WIN)
        y_scalers.append(ys)

    # Scaled thresholds (training split) for event BCE targets
    thresholds_scaled = [
        (tr_thresholds[app] - float(y_scalers[i].data_min_[0]))
        / float(y_scalers[i].data_range_[0])
        for i, app in enumerate(APPLIANCES)
    ]

    # FIX 8: pos_weight from training ON fraction — all appliances enabled
    pos_weights = []
    print("\n  Event pos_weight per appliance:")
    for i, app in enumerate(APPLIANCES):
        flat  = Y_tr[:, :, i].flatten()
        n_on  = float((flat > thresholds_scaled[i]).sum())
        n_off = float((flat <= thresholds_scaled[i]).sum())
        pw    = float(np.clip(n_off / max(n_on, 1.0), *POS_WEIGHT_CLAMP))
        pos_weights.append(pw)
        print(f"    {app:<16}  on={100*n_on/(n_on+n_off):5.1f}%  pos_weight={pw:.1f}")

    # DataLoaders
    tr_ld = torch.utils.data.DataLoader(
        Seq2SeqDataset(X_tr, Y_tr), batch_size=BATCH, shuffle=True,  drop_last=False)
    va_ld = torch.utils.data.DataLoader(
        Seq2SeqDataset(X_va, Y_va), batch_size=BATCH, shuffle=False, drop_last=False)
    te_ld = torch.utils.data.DataLoader(
        Seq2SeqDataset(X_te, Y_te), batch_size=BATCH, shuffle=False, drop_last=False)

    # Model + losses
    model       = FixedPINNLNN(n_feat, hidden, len(APPLIANCES), dt).to(device)
    mse_crit    = nn.MSELoss()
    phys_crit   = PhysicsConsistencyLoss(agg_scaler, y_scalers, epsilon_w).to(device)
    event_crit  = EventDetectionLoss(thresholds_scaled, pos_weights).to(device)
    opt         = torch.optim.Adam(model.parameters(), lr=LR)
    sched       = torch.optim.lr_scheduler.ReduceLROnPlateau(
                      opt, mode='min', factor=0.5, patience=8, min_lr=1e-5)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")
    print(f"Warmup: MSE-only for {WARMUP_EPOCHS} epochs, then + physics + event.\n")

    history = {k: [] for k in [
        'train_loss', 'train_mse', 'train_phys', 'train_event',
        'val_loss',   'val_mse',   'val_phys',   'val_metrics',
    ]}
    best_val_mse = float('inf')
    best_state   = None
    counter      = 0

    for epoch in range(EPOCHS):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        ep_tot = ep_mse = ep_phys = ep_ev = 0.0
        pbar = tqdm(tr_ld, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()

            power_pred, event_logits = model(xb)

            l_mse  = mse_crit(power_pred, yb)
            l_phys = phys_crit(xb, power_pred)    # FIX 3: all timesteps
            l_ev   = event_crit(event_logits, yb)  # FIX 8: BCEWithLogits

            if epoch < WARMUP_EPOCHS:
                loss = l_mse
            else:
                loss = l_mse + lambda_phys * l_phys + lambda_event * l_ev

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_tot  += loss.item();   ep_mse  += l_mse.item()
            ep_phys += l_phys.item(); ep_ev   += l_ev.item()
            pbar.set_postfix({'mse': f'{l_mse.item():.4f}',
                              'phys': f'{l_phys.item():.4f}',
                              'ev': f'{l_ev.item():.4f}'})

        nb = len(tr_ld)
        history['train_loss'].append(ep_tot  / nb)
        history['train_mse'].append(ep_mse   / nb)
        history['train_phys'].append(ep_phys / nb)
        history['train_event'].append(ep_ev  / nb)

        # ── Validate ────────────────────────────────────────────────────────
        model.eval()
        vl_tot = vl_mse = vl_phys = 0.0
        va_preds, va_trues = [], []

        with torch.no_grad():
            for xb, yb in va_ld:
                xb, yb = xb.to(device), yb.to(device)
                power_pred, _ = model(xb)
                l_mse  = mse_crit(power_pred, yb)
                l_phys = phys_crit(xb, power_pred)
                vl_mse  += l_mse.item()
                vl_phys += l_phys.item()
                vl_tot  += (l_mse + lambda_phys * l_phys).item()
                for b in range(power_pred.shape[0]):
                    va_preds.append(power_pred[b].cpu().numpy())
                    va_trues.append(yb[b].cpu().numpy())

        nv = len(va_ld)
        avg_va_mse = vl_mse / nv
        history['val_loss'].append(vl_tot  / nv)
        history['val_mse'].append(avg_va_mse)
        history['val_phys'].append(vl_phys / nv)
        sched.step(avg_va_mse)

        pred_trace = reconstruct_trace(va_preds, n_va, STRIDE, WIN)
        true_trace = reconstruct_trace(va_trues, n_va, STRIDE, WIN)
        vm         = per_app_metrics(true_trace, pred_trace, y_scalers, va_thresholds)
        history['val_metrics'].append(vm)

        avg_f1  = np.mean([vm[a]['f1']  for a in APPLIANCES])
        avg_mae = np.mean([vm[a]['mae'] for a in APPLIANCES])
        status  = "WARMUP" if epoch < WARMUP_EPOCHS else "FULL  "
        print(
            f"  [{status}] Epoch {epoch+1:3d}/{EPOCHS}  "
            f"train={history['train_loss'][-1]:.5f} "
            f"(mse={history['train_mse'][-1]:.5f} "
            f"phys={history['train_phys'][-1]:.5f} "
            f"ev={history['train_event'][-1]:.5f})  "
            f"val_mse={avg_va_mse:.5f}  "
            f"avgF1={avg_f1:.4f}  avgMAE={avg_mae:.2f}  "
            f"lr={opt.param_groups[0]['lr']:.2e}"
        )
        for app in APPLIANCES:
            m = vm[app]
            print(f"    {app:<16}  F1={m['f1']:.4f}  P={m['precision']:.4f}  "
                  f"R={m['recall']:.4f}  MAE={m['mae']:.2f}  SAE={m['sae']:.4f}")

        if avg_va_mse < best_val_mse:
            best_val_mse = avg_va_mse
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            counter      = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

    print("\nTraining complete.")

    # ── Test ──────────────────────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    te_preds, te_trues = [], []
    with torch.no_grad():
        for xb, yb in te_ld:
            power_pred, _ = model(xb.to(device))
            for b in range(power_pred.shape[0]):
                te_preds.append(power_pred[b].cpu().numpy())
                te_trues.append(yb[b].cpu().numpy())

    pred_trace_te = reconstruct_trace(te_preds, n_te, WIN, WIN)
    true_trace_te = reconstruct_trace(te_trues, n_te, WIN, WIN)
    test_metrics  = per_app_metrics(true_trace_te, pred_trace_te, y_scalers, te_thresholds)

    print(f"\n{'Appliance':<18} {'F1':>8} {'Precision':>10} "
          f"{'Recall':>8} {'MAE':>8} {'SAE':>8}")
    print("-" * 65)
    for app in APPLIANCES:
        m = test_metrics[app]
        print(f"{app:<18} {m['f1']:>8.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['mae']:>8.2f} {m['sae']:>8.4f}")

    _plot_loss(history, save_dir)
    _plot_metrics(history, test_metrics, save_dir)

    cfg = {
        'dataset': 'UKDALE (CSV splits)',
        'model':   'FixedPINNLNN',
        'fixes_applied': [
            'FIX1: sigmoid on power heads',
            'FIX2: Seq2Seq (all WIN timesteps)',
            'FIX3: physics loss at every timestep',
            'FIX4: WIN=299',
            'FIX5: 8-channel input (median+EMA+residual+delta+rolling)',
            'FIX6: StandardScaler (Z-score) for inputs',
            'FIX7: bidirectional LNN',
            'FIX8: BCEWithLogits + pos_weight for all appliances',
            'FIX9: adaptive per-split thresholds',
        ],
        'thresholds': {
            'train': tr_thresholds, 'val': va_thresholds, 'test': te_thresholds},
        'pos_weights': dict(zip(APPLIANCES, [float(w) for w in pos_weights])),
        'window': {'win': WIN, 'stride_train': STRIDE, 'stride_test': WIN},
        'model_params': {'in_ch': n_feat, 'hidden': hidden,
                         'n_apps': len(APPLIANCES), 'dt': dt},
        'train_params': {'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE,
                         'lambda_phys': lambda_phys, 'lambda_event': lambda_event,
                         'epsilon_w': epsilon_w, 'warmup_epochs': WARMUP_EPOCHS},
        'test_metrics': {
            app: {k: float(v) for k, v in m.items()}
            for app, m in test_metrics.items()
        },
    }
    with open(os.path.join(save_dir, 'fixed_pinn_lnn_results.json'),
              'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=4)
    print(f"\nResults saved to: {save_dir}")
    return test_metrics, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_loss(history: dict, save_dir: str) -> None:
    ep = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle('Fixed PINN-LNN -- Loss Curves')
    pairs = [
        ('train_loss',  'val_loss',  'Total Loss'),
        ('train_mse',   'val_mse',   'MSE Loss'),
        ('train_phys',  'val_phys',  'Physics Loss'),
        ('train_event', None,        'Event BCE (train)'),
    ]
    for ax, (tr_k, va_k, title) in zip(axes, pairs):
        ax.plot(ep, history[tr_k], label='Train', color='steelblue')
        if va_k:
            ax.plot(ep, history[va_k], label='Val', color='tomato')
        ax.set_title(title); ax.set_xlabel('Epoch')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fixed_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _plot_metrics(history: dict, test_metrics: dict, save_dir: str) -> None:
    ep = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(len(APPLIANCES), 2,
                             figsize=(12, 4 * len(APPLIANCES)))
    fig.suptitle('Fixed PINN-LNN -- Per-Appliance Validation Metrics')
    for row, app in enumerate(APPLIANCES):
        f1s  = [m[app]['f1']  for m in history['val_metrics']]
        maes = [m[app]['mae'] for m in history['val_metrics']]
        axes[row][0].plot(ep, f1s, color='steelblue')
        axes[row][0].axhline(test_metrics[app]['f1'], color='green',
                              linestyle='--', label='Test')
        axes[row][0].set_title(f'{app} -- F1')
        axes[row][0].legend(); axes[row][0].grid(alpha=0.3)
        axes[row][1].plot(ep, maes, color='tomato')
        axes[row][1].axhline(test_metrics[app]['mae'], color='green',
                              linestyle='--', label='Test')
        axes[row][1].set_title(f'{app} -- MAE (W)')
        axes[row][1].legend(); axes[row][1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fixed_per_appliance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    for f in ['UKDALE_HF_train.csv', 'UKDALE_HF_validation.csv', 'UKDALE_HF_test.csv']:
        p = os.path.join(DATA_DIR, f)
        if not os.path.exists(p):
            print(f"Error: {p} not found.")
            sys.exit(1)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir  = f'models/fixed_pinn_lnn_{timestamp}'

    data_dict = load_data()

    train(
        data_dict,
        save_dir     = save_dir,
        hidden       = 64,
        dt           = 0.1,
        lambda_phys  = LAMBDA_PHYS,
        lambda_event = LAMBDA_EVENT,
        epsilon_w    = EPSILON_W,
    )
