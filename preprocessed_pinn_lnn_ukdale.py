"""
Preprocessed PINN-LNN for UKDALE NILM
======================================

Implements fundamental preprocessing + time-domain transformations for NILM:

─────────────────────────────────────────────────────────────────────────────
1.  Fundamental Preprocessing
─────────────────────────────────────────────────────────────────────────────

A.  Resampling & Interpolation
    Already handled by preprocess_hf.py (6-second regular grid, forward-fill).
    Loaded here directly from the preprocessed CSVs.

B.  Denoising
    Raw power meters contain two types of noise:
      • Impulse noise — short-duration spikes from electrical interference,
        not corresponding to any real appliance event.
      • Slow drift — low-amplitude baseline wander from grid voltage variation.

    Two complementary filters are applied in sequence:

    1.  Median Filter (window = MEDFILT_K steps = 30 s)
        Removes impulse spikes while preserving sharp step edges.
        Unlike a mean filter, it does not blur real switching transients.
        Implemented via a rolling median with center alignment.

    2.  EMA Low-pass Filter (span = EMA_SPAN steps = 60 s)
        Exponential Moving Average smooths the median-filtered signal,
        removing residual high-frequency noise while tracking trends.
        α = 2/(span+1) controls the cutoff frequency.

    Residual = raw − EMA_smooth
        The difference between the raw and smoothed signals isolates the
        high-frequency content — this is where appliance switching edges live.

C.  Normalization
    Two strategies are used for different parts of the pipeline:

    Z-score (StandardScaler) for input features:
        More robust than Min-Max when features contain residuals or delta-power
        values with heavy-tailed distributions (appliance spikes create outliers
        that would saturate a [0,1] scale and collapse all non-spike variation).
        Inverse formula:  x_raw = x_z × σ + μ

    Min-Max for appliance targets:
        Keeps predictions in [0,1], required for the physics loss to remain
        in a consistent scale relative to the inverse-scaled aggregate.

─────────────────────────────────────────────────────────────────────────────
2.  Time-Domain Transformations
─────────────────────────────────────────────────────────────────────────────

A.  Sliding Windows  (Seq2Seq, WIN=299 steps, ~30 min)
    Same as hf_seq2seq_pinn_lnn_ukdale.py — all WIN targets predicted.

B.  Delta Power (ΔP) at three levels:
    ΔP_raw_1    = raw[t]    − raw[t-1]     (raw 6s edge)
    ΔP_smooth_1 = smooth[t] − smooth[t-1]  (denoised 6s edge — cleaner,
                                             fewer false positives)
    ΔP_smooth_6 = smooth[t] − smooth[t-6]  (36s edge — phase transitions)

    Delta of the smoothed signal is specifically recommended: after removing
    impulse noise, the remaining steps correspond more reliably to real
    appliance events.

─────────────────────────────────────────────────────────────────────────────
Feature channels  (N_FEAT = 12)
─────────────────────────────────────────────────────────────────────────────
    0   raw aggregate           original mains power (W)
    1   median-filtered         impulse noise removed
    2   EMA-smoothed            slow-drift removed
    3   residual                raw − smooth (high-freq content, edge energy)
    4   ΔP raw 1-step           raw switching edge
    5   ΔP smooth 1-step        denoised switching edge (fewer false positives)
    6   ΔP smooth 6-step        denoised 36s phase transition
    7   rolling mean 10         60s local baseline (on smoothed signal)
    8   rolling std  10         60s local variability (on smoothed signal)
    9   rolling mean 50         5min trend (on smoothed signal)
   10   rolling std  50         5min variability (on smoothed signal)
   11   residual energy         rolling mean of residual² over 10 steps
                                (high when many spikes/edges are present)

Architecture:  same LengthPreservingCNN + Bidirectional LNN + Seq2Seq + Physics
               as hf_seq2seq_pinn_lnn_ukdale.py, adapted for Z-score inputs.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Source Code'))
from utils import calculate_nilm_metrics


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPOCHS        = 80
PATIENCE      = 20
LR            = 1e-3
BATCH         = 32
WIN           = 299
STRIDE        = 10
STRIDE_TEST   = WIN

MEDFILT_K     = 5        # median filter kernel: 5 steps = 30 s
EMA_SPAN      = 10       # EMA span: 10 steps = 60 s  (α ≈ 0.18)
ROLL_S        = 10       # short rolling window: 60 s
ROLL_L        = 50       # long  rolling window:  5 min

LAMBDA_PHYS   = 0.01
EPSILON_W     = 50.0
WARMUP_EPOCHS = 20

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
# Preprocessing pipeline
# ---------------------------------------------------------------------------

def _median_filter(arr: np.ndarray, k: int = MEDFILT_K) -> np.ndarray:
    """
    Center-aligned rolling median.
    Removes impulse spikes without blurring real step edges.
    Edge NaNs are filled with the nearest valid value.
    """
    s = pd.Series(arr)
    med = s.rolling(k, center=True, min_periods=1).median()
    return med.values.astype(np.float32)


def _ema_filter(arr: np.ndarray, span: int = EMA_SPAN) -> np.ndarray:
    """
    Exponential Moving Average low-pass filter.
    α = 2/(span+1).  Smooth baseline; removes high-frequency drift.
    Applied to the already median-filtered signal.
    """
    return pd.Series(arr).ewm(span=span, adjust=False).mean().values.astype(np.float32)


def _n_step_diff(arr: np.ndarray, n: int) -> np.ndarray:
    """arr[t] - arr[t-n], zero-padded at the head."""
    d     = np.zeros_like(arr)
    d[n:] = arr[n:] - arr[:-n]
    return d


def compute_features(df: pd.DataFrame) -> np.ndarray:
    """
    Full preprocessing pipeline → (N, 12) feature matrix.

    Pipeline per sample:
        raw  →  median filter  →  EMA low-pass  →  smoothed
        residual = raw - smoothed
        delta features on raw and smoothed
        rolling stats on smoothed
        residual energy (rolling mean of residual²)
    """
    raw   = df['aggregate'].values.astype(np.float32)

    # ── Denoising ─────────────────────────────────────────────────────────────
    med    = _median_filter(raw, MEDFILT_K)
    smooth = _ema_filter(med, EMA_SPAN)
    resid  = (raw - smooth).astype(np.float32)     # high-frequency component

    # ── Delta features ────────────────────────────────────────────────────────
    d_raw_1    = _n_step_diff(raw,    1)   # raw 6s edge
    d_smooth_1 = _n_step_diff(smooth, 1)  # denoised 6s edge
    d_smooth_6 = _n_step_diff(smooth, 6)  # denoised 36s phase transition

    # ── Rolling stats on smoothed signal ─────────────────────────────────────
    s      = pd.Series(smooth)
    rm_s   = s.rolling(ROLL_S, min_periods=1).mean().values.astype(np.float32)
    rs_s   = s.rolling(ROLL_S, min_periods=1).std().fillna(0).values.astype(np.float32)
    rm_l   = s.rolling(ROLL_L, min_periods=1).mean().values.astype(np.float32)
    rs_l   = s.rolling(ROLL_L, min_periods=1).std().fillna(0).values.astype(np.float32)

    # ── Residual energy — rolling mean of residual² ───────────────────────────
    resid_energy = (pd.Series(resid ** 2)
                    .rolling(ROLL_S, min_periods=1)
                    .mean()
                    .values.astype(np.float32))

    return np.stack([
        raw, med, smooth, resid,
        d_raw_1, d_smooth_1, d_smooth_6,
        rm_s, rs_s, rm_l, rs_l,
        resid_energy,
    ], axis=1)   # (N, 12)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(split: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f'UKDALE_HF_{split}.csv')
    df   = pd.read_csv(path, index_col='timestamp', parse_dates=True)
    print(f"  {split:12s}: {len(df):6d} rows")
    return df


def create_sequences(feat: np.ndarray, targets: np.ndarray, stride: int):
    """Seq2Seq: X (M, WIN, N_FEAT), Y (M, WIN, n_apps)."""
    X, Y = [], []
    for i in range(0, len(feat) - WIN, stride):
        X.append(feat[i : i + WIN])
        Y.append(targets[i : i + WIN])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


# ---------------------------------------------------------------------------
# Physics Consistency Loss  —  adapted for Z-score input
# ---------------------------------------------------------------------------

class PhysicsConsistencyLoss(nn.Module):
    """
    One-sided energy conservation penalty at every timestep:
        L_phys = mean_{batch,t}( ReLU( Σ_i p̂_i_raw(t) − P_agg_raw(t) − ε ) )

    Input feature channel 0 is the raw aggregate, Z-score standardised.
    Inverse transform:  x_raw = x_z × x_scale + x_mean

    Appliance targets use MinMaxScaler (compatible with previous scripts).
    """

    def __init__(self, agg_scaler: StandardScaler,
                 y_scalers: list,
                 epsilon_w: float = EPSILON_W):
        super().__init__()
        self.epsilon = epsilon_w
        # StandardScaler: mean_ and scale_ (std dev)
        self.register_buffer('x_mean',
            torch.tensor(float(agg_scaler.mean_[0]),  dtype=torch.float32))
        self.register_buffer('x_scale',
            torch.tensor(float(agg_scaler.scale_[0]), dtype=torch.float32))
        self.register_buffer('y_mins',
            torch.tensor([float(s.data_min_[0])   for s in y_scalers],
                         dtype=torch.float32))
        self.register_buffer('y_ranges',
            torch.tensor([float(s.data_range_[0]) for s in y_scalers],
                         dtype=torch.float32))

    def forward(self, x_z: torch.Tensor,
                pred_scaled: torch.Tensor) -> torch.Tensor:
        """
        x_z:         (batch, WIN, N_FEAT)  channel 0 = Z-score aggregate
        pred_scaled: (batch, WIN, n_apps)
        """
        # Inverse Z-score: recover raw Watts
        x_raw = x_z[:, :, 0] * self.x_scale + self.x_mean
        p_raw = pred_scaled * self.y_ranges + self.y_mins
        return F.relu(p_raw.sum(dim=-1) - x_raw - self.epsilon).mean()


# ---------------------------------------------------------------------------
# Length-preserving dilated CNN encoder  (identical to seq2seq script)
# ---------------------------------------------------------------------------

class LengthPreservingCNN(nn.Module):
    """
    4 dilated Conv1d layers; 'same' padding; output length = input length.
    Dilation 1 → 4 → 16 → 64; receptive field = 183 steps (~18 min).
    """

    def __init__(self, in_ch: int, hidden: int):
        super().__init__()
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
        return self.net(x.permute(0, 2, 1)).permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Bidirectional Liquid Cell
# ---------------------------------------------------------------------------

class LiquidCell(nn.Module):
    """AdvancedLiquidTimeLayer — one step, fixed dt."""

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
# Full Seq2Seq model
# ---------------------------------------------------------------------------

class PreprocessedSeq2SeqNet(nn.Module):
    """
    LengthPreservingCNN  +  Bidirectional LNN  +  per-timestep heads.

    Identical architecture to hf_seq2seq_pinn_lnn_ukdale.py; the improvement
    here is entirely in the preprocessing pipeline that feeds into it.
    This isolates the contribution of denoising and Z-score standardization
    from architectural changes, making comparison straightforward.
    """

    def __init__(self, in_ch: int, hidden: int, n_apps: int, dt: float = 0.1):
        super().__init__()
        self.hidden = hidden
        self.n_apps = n_apps

        self.cnn      = LengthPreservingCNN(in_ch, hidden)
        self.fwd_cell = LiquidCell(hidden, hidden, dt)
        self.bwd_cell = LiquidCell(hidden, hidden, dt)

        self.norm  = nn.LayerNorm(hidden * 2)
        self.heads = nn.ModuleList([
            nn.Linear(hidden * 2, 1) for _ in range(n_apps)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat  = self.cnn(x)
        batch = feat.shape[0]
        T     = feat.shape[1]

        h_f = torch.zeros(batch, self.hidden, device=x.device)
        fwd = []
        for t in range(T):
            h_f = self.fwd_cell(feat[:, t, :], h_f)
            fwd.append(h_f)

        h_b = torch.zeros(batch, self.hidden, device=x.device)
        bwd = [None] * T
        for t in reversed(range(T)):
            h_b    = self.bwd_cell(feat[:, t, :], h_b)
            bwd[t] = h_b

        out = []
        for t in range(T):
            h_t = self.norm(torch.cat([fwd[t], bwd[t]], dim=1))
            out.append(torch.cat([head(h_t) for head in self.heads], dim=1))
        return torch.stack(out, dim=1)


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
# Trace reconstruction
# ---------------------------------------------------------------------------

def reconstruct_trace(window_preds: list, n_total: int,
                      stride: int, win: int) -> np.ndarray:
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
    return (acc / np.maximum(count, 1)).astype(np.float32)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def per_app_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_scalers: list) -> dict:
    out = {}
    for i, app in enumerate(APPLIANCES):
        raw_t = y_scalers[i].inverse_transform(y_true[:, i:i+1]).flatten()
        raw_p = y_scalers[i].inverse_transform(y_pred[:, i:i+1]).flatten()
        out[app] = calculate_nilm_metrics(raw_t, raw_p, threshold=THRESHOLDS[app])
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(save_dir: str,
          hidden:      int   = 64,
          dt:          float = 0.1,
          lambda_phys: float = LAMBDA_PHYS,
          epsilon_w:   float = EPSILON_W) -> tuple:

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}  |  WIN={WIN}  hidden={hidden}  dt={dt}")

    # ── Load CSVs ─────────────────────────────────────────────────────────────
    print("\nLoading data and applying preprocessing pipeline...")
    df_tr = load_csv('train')
    df_va = load_csv('validation')
    df_te = load_csv('test')
    n_tr, n_va, n_te = len(df_tr), len(df_va), len(df_te)

    # ── Preprocessing: denoising + feature extraction ─────────────────────────
    feat_tr = compute_features(df_tr)
    feat_va = compute_features(df_va)
    feat_te = compute_features(df_te)
    n_feat  = feat_tr.shape[1]   # 12

    print(f"  Feature channels: {n_feat}")
    print(f"  Median filter   : window = {MEDFILT_K} steps ({MEDFILT_K*6} s)")
    print(f"  EMA low-pass    : span   = {EMA_SPAN} steps  (α ≈ {2/(EMA_SPAN+1):.3f})")
    print(f"  Normalization   : Z-score for inputs, Min-Max for targets")

    # Show denoising effect on training data
    raw_std   = df_tr['aggregate'].std()
    smooth_ch = feat_tr[:, 2]   # channel 2 = EMA-smoothed
    resid_ch  = feat_tr[:, 3]   # channel 3 = residual
    print(f"\n  Aggregate std (raw)     : {raw_std:.2f} W")
    print(f"  EMA-smoothed std        : {smooth_ch.std():.2f} W")
    print(f"  Residual std            : {resid_ch.std():.2f} W  "
          f"({resid_ch.std()/raw_std*100:.1f}% of raw variance)")

    # ── Appliance targets ─────────────────────────────────────────────────────
    tgt_tr = df_tr[APPLIANCES].values.astype(np.float32)
    tgt_va = df_va[APPLIANCES].values.astype(np.float32)
    tgt_te = df_te[APPLIANCES].values.astype(np.float32)

    # ── Sliding-window sequences ───────────────────────────────────────────────
    X_tr, Y_tr = create_sequences(feat_tr, tgt_tr, STRIDE)
    X_va, Y_va = create_sequences(feat_va, tgt_va, STRIDE)
    X_te, Y_te = create_sequences(feat_te, tgt_te, WIN)
    print(f"\nTrain : {X_tr.shape} -> {Y_tr.shape}  "
          f"({X_tr.shape[0] * WIN:,} predictions)")
    print(f"Val   : {X_va.shape} -> {Y_va.shape}")
    print(f"Test  : {X_te.shape} -> {Y_te.shape}  [non-overlapping]")

    # ── Z-score standardisation for input features ────────────────────────────
    # Fit on all values of each channel across all training windows.
    # Heavy-tailed features (residual, delta-power spikes) stay interpretable
    # — outliers are expressed in units of std deviation, not clipped to [0,1].
    feat_scalers = []
    for ch in range(n_feat):
        sc = StandardScaler()
        X_tr[:, :, ch] = sc.fit_transform(
            X_tr[:, :, ch].reshape(-1, 1)).reshape(-1, WIN)
        X_va[:, :, ch] = sc.transform(
            X_va[:, :, ch].reshape(-1, 1)).reshape(-1, WIN)
        X_te[:, :, ch] = sc.transform(
            X_te[:, :, ch].reshape(-1, 1)).reshape(-1, WIN)
        feat_scalers.append(sc)

    agg_scaler = feat_scalers[0]   # StandardScaler for raw aggregate (ch 0)
    print(f"\n  Agg Z-score params:  mean={agg_scaler.mean_[0]:.1f} W  "
          f"std={agg_scaler.scale_[0]:.1f} W")

    # ── Min-Max for appliance targets ─────────────────────────────────────────
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

    thresholds_scaled = [
        (THRESHOLDS[app] - float(y_scalers[i].data_min_[0]))
        / float(y_scalers[i].data_range_[0])
        for i, app in enumerate(APPLIANCES)
    ]

    # ── DataLoaders ───────────────────────────────────────────────────────────
    tr_ld = torch.utils.data.DataLoader(
        Seq2SeqDataset(X_tr, Y_tr), batch_size=BATCH, shuffle=True,  drop_last=False)
    va_ld = torch.utils.data.DataLoader(
        Seq2SeqDataset(X_va, Y_va), batch_size=BATCH, shuffle=False, drop_last=False)
    te_ld = torch.utils.data.DataLoader(
        Seq2SeqDataset(X_te, Y_te), batch_size=BATCH, shuffle=False, drop_last=False)

    # ── Model + losses ────────────────────────────────────────────────────────
    model     = PreprocessedSeq2SeqNet(
                    in_ch=n_feat, hidden=hidden,
                    n_apps=len(APPLIANCES), dt=dt).to(device)
    mse_crit  = nn.MSELoss()
    phys_crit = PhysicsConsistencyLoss(
                    agg_scaler, y_scalers, epsilon_w).to(device)
    opt       = torch.optim.Adam(model.parameters(), lr=LR)
    sched     = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode='min', factor=0.5, patience=8, min_lr=1e-5)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")
    print("Starting Preprocessed PINN-LNN training...")

    history = {k: [] for k in [
        'train_loss', 'train_mse', 'train_phys',
        'val_loss',   'val_mse',   'val_phys', 'val_metrics',
    ]}
    best_val_mse = float('inf')
    best_state   = None
    counter      = 0

    for epoch in range(EPOCHS):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        ep_mse = ep_phys = ep_tot = 0.0
        pbar = tqdm(tr_ld, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)

            l_mse  = mse_crit(pred, yb)
            l_phys = phys_crit(xb, pred)

            if epoch < WARMUP_EPOCHS:
                loss = l_mse
            else:
                l_bce = torch.tensor(0.0, device=device)
                for i, app in enumerate(APPLIANCES):
                    if BCE_LAMBDA[app] > 0:
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

        # ── Validate ────────────────────────────────────────────────────────
        model.eval()
        vl_mse = vl_phys = vl_tot = 0.0
        va_preds, va_trues = [], []

        with torch.no_grad():
            for xb, yb in va_ld:
                xb, yb = xb.to(device), yb.to(device)
                pred   = model(xb)
                l_mse  = mse_crit(pred, yb)
                l_phys = phys_crit(xb, pred)
                vl_mse  += l_mse.item()
                vl_phys += l_phys.item()
                vl_tot  += (l_mse + lambda_phys * l_phys).item()
                for b in range(pred.shape[0]):
                    va_preds.append(pred[b].cpu().numpy())
                    va_trues.append(yb[b].cpu().numpy())

        avg_va_mse  = vl_mse  / len(va_ld)
        avg_va_phys = vl_phys / len(va_ld)
        avg_va_tot  = vl_tot  / len(va_ld)
        history['val_mse'].append(avg_va_mse)
        history['val_phys'].append(avg_va_phys)
        history['val_loss'].append(avg_va_tot)
        sched.step(avg_va_mse)

        pred_trace = reconstruct_trace(va_preds, n_va, STRIDE, WIN)
        true_trace = reconstruct_trace(va_trues, n_va, STRIDE, WIN)
        vm         = per_app_metrics(true_trace, pred_trace, y_scalers)
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

    # ── Test ──────────────────────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    te_preds, te_trues = [], []
    with torch.no_grad():
        for xb, yb in te_ld:
            pred = model(xb.to(device))
            for b in range(pred.shape[0]):
                te_preds.append(pred[b].cpu().numpy())
                te_trues.append(yb[b].cpu().numpy())

    pred_trace_te = reconstruct_trace(te_preds, n_te, WIN, WIN)
    true_trace_te = reconstruct_trace(te_trues, n_te, WIN, WIN)
    test_metrics  = per_app_metrics(true_trace_te, pred_trace_te, y_scalers)

    print(f"\n{'Appliance':<24} {'F1':>8} {'Precision':>10} "
          f"{'Recall':>8} {'MAE':>8} {'SAE':>8}")
    print("-" * 72)
    for app in APPLIANCES:
        m = test_metrics[app]
        print(f"{app:<24} {m['f1']:>8.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['mae']:>8.2f} {m['sae']:>8.4f}")

    _plot_preprocessing(df_tr, save_dir)
    _plot_loss(history, save_dir)
    _plot_metrics(history, test_metrics, save_dir)
    _plot_trace(true_trace_te, pred_trace_te, y_scalers, save_dir)

    cfg = {
        'dataset':  'UKDALE HF 6s',
        'model':    'PreprocessedSeq2SeqNet',
        'preprocessing': {
            'denoising': [
                f'Median filter: window={MEDFILT_K} steps ({MEDFILT_K*6}s)',
                f'EMA low-pass: span={EMA_SPAN} steps (alpha={2/(EMA_SPAN+1):.3f})',
            ],
            'normalization': {
                'inputs':  'Z-score (StandardScaler) per channel',
                'targets': 'Min-Max (MinMaxScaler) per appliance',
            },
            'delta_features': [
                'ΔP raw 1-step (6s)',
                'ΔP smooth 1-step (6s, denoised)',
                'ΔP smooth 6-step (36s, denoised)',
            ],
        },
        'features': {
            'channels': [
                'raw_aggregate', 'median_filtered', 'ema_smoothed', 'residual',
                'delta_raw_1', 'delta_smooth_1', 'delta_smooth_6',
                'roll_mean_10', 'roll_std_10', 'roll_mean_50', 'roll_std_50',
                'residual_energy',
            ],
            'total': n_feat,
        },
        'architecture': {
            'cnn_dilations':  [1, 4, 16, 64],
            'lnn_direction':  'bidirectional',
            'output':         'seq2seq (all WIN timesteps)',
            'physics_constraint': 'every timestep (Z-score inverse for aggregate)',
        },
        'window': {'win': WIN, 'stride_train': STRIDE, 'stride_test': WIN},
        'model_params': {'in_ch': n_feat, 'hidden': hidden,
                         'n_apps': len(APPLIANCES), 'dt': dt},
        'train_params': {'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE,
                         'lambda_phys': lambda_phys, 'epsilon_w': epsilon_w,
                         'warmup_epochs': WARMUP_EPOCHS},
        'test_metrics': {
            app: {k: float(v) for k, v in m.items()}
            for app, m in test_metrics.items()
        },
    }
    with open(os.path.join(save_dir, 'preprocessed_results.json'),
              'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=4)
    print(f"\nResults saved to {save_dir}")
    return test_metrics, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_preprocessing(df: pd.DataFrame, save_dir: str,
                         n_steps: int = 600) -> None:
    """
    Show the effect of the denoising pipeline on 1 hour of training data.
    Plots: raw vs median-filtered vs EMA-smoothed, and the residual.
    """
    agg    = df['aggregate'].values[:n_steps].astype(np.float32)
    med    = _median_filter(agg, MEDFILT_K)
    smooth = _ema_filter(med, EMA_SPAN)
    resid  = agg - smooth
    t      = np.arange(n_steps) * 6 / 60   # minutes

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle('Denoising Pipeline — Training Data (first 1 hour)', fontsize=12)

    ax1.plot(t, agg,    label='Raw aggregate',    color='steelblue', alpha=0.5,
             linewidth=0.8)
    ax1.plot(t, med,    label=f'Median (k={MEDFILT_K})', color='orange',
             linewidth=1.0)
    ax1.plot(t, smooth, label=f'EMA (span={EMA_SPAN})',  color='tomato',
             linewidth=1.2)
    ax1.set_ylabel('Power (W)'); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2.plot(t, resid, color='purple', linewidth=0.8, label='Residual (raw − smooth)')
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.set_ylabel('Residual (W)'); ax2.set_xlabel('Time (min)')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'preprocessing_pipeline.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def _plot_loss(history: dict, save_dir: str) -> None:
    ep = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(15, 4))
    for idx, (tr_k, va_k, title) in enumerate([
        ('train_loss',  'val_loss',  'Total Loss'),
        ('train_mse',   'val_mse',   'MSE Loss'),
        ('train_phys',  'val_phys',  'Physics Loss'),
    ]):
        plt.subplot(1, 3, idx + 1)
        plt.plot(ep, history[tr_k], label='Train', color='steelblue')
        plt.plot(ep, history[va_k], label='Val',   color='tomato')
        plt.title(title); plt.xlabel('Epoch')
        plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'preprocessed_loss.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def _plot_metrics(history: dict, test_metrics: dict, save_dir: str) -> None:
    ep  = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(len(APPLIANCES), 2,
                             figsize=(12, 4 * len(APPLIANCES)))
    fig.suptitle('Preprocessed PINN-LNN — Per-Appliance Validation Metrics')
    for row, app in enumerate(APPLIANCES):
        f1s  = [m[app]['f1']  for m in history['val_metrics']]
        maes = [m[app]['mae'] for m in history['val_metrics']]
        axes[row][0].plot(ep, f1s, color='steelblue')
        axes[row][0].axhline(test_metrics[app]['f1'], color='green',
                             linestyle='--', label='Test')
        axes[row][0].set_title(f'{app} — F1')
        axes[row][0].legend(); axes[row][0].grid(alpha=0.3)
        axes[row][1].plot(ep, maes, color='tomato')
        axes[row][1].axhline(test_metrics[app]['mae'], color='green',
                             linestyle='--', label='Test')
        axes[row][1].set_title(f'{app} — MAE (W)')
        axes[row][1].legend(); axes[row][1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'preprocessed_per_appliance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def _plot_trace(true_trace, pred_trace, y_scalers, save_dir,
                n_steps: int = 600) -> None:
    fig, axes = plt.subplots(len(APPLIANCES), 1,
                             figsize=(14, 3 * len(APPLIANCES)))
    fig.suptitle('Preprocessed PINN-LNN — Reconstructed Test Trace (1 hour)')
    t = np.arange(n_steps) * 6 / 60
    for row, app in enumerate(APPLIANCES):
        i     = APPLIANCES.index(app)
        raw_t = y_scalers[i].inverse_transform(
                    true_trace[:n_steps, i:i+1]).flatten()
        raw_p = y_scalers[i].inverse_transform(
                    pred_trace[:n_steps, i:i+1]).flatten()
        axes[row].plot(t, raw_t, label='Ground truth', color='steelblue',
                       linewidth=1.0, alpha=0.8)
        axes[row].plot(t, raw_p, label='Prediction',   color='tomato',
                       linewidth=1.0, alpha=0.8)
        axes[row].set_title(app); axes[row].set_ylabel('Power (W)')
        axes[row].legend(loc='upper right'); axes[row].grid(alpha=0.3)
    axes[-1].set_xlabel('Time (min)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'preprocessed_trace.png'),
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
    save_dir  = f'models/preprocessed_pinn_lnn_{timestamp}'

    train(
        save_dir     = save_dir,
        hidden       = 64,
        dt           = 0.1,
        lambda_phys  = LAMBDA_PHYS,
        epsilon_w    = EPSILON_W,
    )
