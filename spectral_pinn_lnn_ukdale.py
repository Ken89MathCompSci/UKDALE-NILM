"""
Spectral PINN-LNN for UKDALE NILM
===================================

Extends multiresolution_pinn_lnn_ukdale.py with frequency-domain preprocessing.
The 12-channel denoised feature set is augmented with 13 spectral channels:

────────────────────────────────────────────────────────────────────────────────
Feature channels  (N_FEAT = 25)
────────────────────────────────────────────────────────────────────────────────
  0    raw aggregate           original mains power (W)
  1    median filtered         impulse noise removed
  2    EMA smoothed            slow-drift removed
  3    residual                raw - smooth (edge energy)
  4    delta raw 1-step        raw 6-s edge
  5    delta smooth 1-step     denoised 6-s edge
  6    delta smooth 6-step     denoised 36-s transition
  7    rolling mean 10         60-s local baseline
  8    rolling std  10         60-s variability
  9    rolling mean 50         5-min trend
  10   rolling std  50         5-min variability
  11   residual energy         rolling mean(residual^2, 10 steps)

  12-19  STFT log-magnitude bins (causal, win=60 steps = 6 min, 8 log-spaced bins)
         Computed on the raw aggregate.  Each bin covers a log-spaced band of
         temporal frequencies from ~2.8e-3 Hz (6 min cycles) to ~0.08 Hz (12 s).
         Captures appliance harmonic signatures, fridge compressor oscillations,
         and dishwasher/washing-machine fill-heat-spin cycle structure.

  20-23  Wavelet detail coefficients — SWT db4, levels 1-4
         Stationary Wavelet Transform preserves signal length at every level.
         Detail coeff at level k = band-pass signal at 2^k * 6 s scale:
           Level 1 (ch 20): 12-24 s   — single-switch transients
           Level 2 (ch 21): 24-48 s   — double-switch patterns
           Level 3 (ch 22): 48-96 s   — appliance state cycles
           Level 4 (ch 23): 96-192 s  — fridge / WM cycle periods
         Requires PyWavelets (pip install PyWavelets).
         Falls back to zeros if not installed.

  24   Spectral flux
       L2 norm of the frame-to-frame STFT bin difference.
       Large during appliance ON/OFF transitions; near-zero during steady state.
       Complements delta-power for event detection.

────────────────────────────────────────────────────────────────────────────────
Architecture (identical to multiresolution_pinn_lnn_ukdale.py)
────────────────────────────────────────────────────────────────────────────────
  DualBranchCNN (in_ch=25)
    Fast branch: dilations 1, 2   (RF = 15 steps =  90 s)
    Slow branch: dilations 1, 16, 64 (RF = 199 steps = ~20 min)
    -> (batch, WIN, hidden)
  Bidirectional LNN (forward + backward LiquidCell)
    -> (batch, WIN, hidden*2)
  Power heads:  n_apps x Linear(hidden*2, 1) + sigmoid
  Event heads:  n_apps x Linear(hidden*2, 1)  [logits]

Losses:
  L = L_MSE + lambda_t * L_phys_time + lambda_f * L_phys_freq
          + lambda_e * L_event
  Warmup: MSE-only for first WARMUP_EPOCHS epochs.

Also includes fix #1: adaptive per-split ON/OFF thresholds calibrated from
the 5th percentile of each split's own nonzero readings + 20 W.
"""

import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    warnings.warn(
        "PyWavelets not found — wavelet features will be zeros.\n"
        "Install with:  pip install PyWavelets",
        RuntimeWarning, stacklevel=1,
    )

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

MEDFILT_K     = 5
EMA_SPAN      = 10
ROLL_S        = 10
ROLL_L        = 50

STFT_WIN      = 60          # causal STFT window: 60 steps = 6 min
STFT_BINS     = 8           # log-spaced frequency bands
N_WAVELET_LVL = 4           # SWT detail levels (db4 wavelet)

# N_FEAT = 12 existing + 8 STFT + 4 wavelet + 1 flux = 25
N_FEAT_BASE   = 12
N_FEAT        = N_FEAT_BASE + STFT_BINS + N_WAVELET_LVL + 1   # 25

LAMBDA_PHYS_T = 0.01
LAMBDA_PHYS_F = 0.005
LAMBDA_EVENT  = 0.05
EPSILON_W     = 50.0
WARMUP_EPOCHS = 20

THRESHOLD_DELTA   = 20.0    # W above standby = definitely ON
THRESHOLD_LOW_PCT = 0.05    # low percentile of nonzero = standby estimate
THRESHOLD_MIN     = 10.0    # floor

POS_WEIGHT_CLAMP = (1.0, 50.0)

APPLIANCES = ['dishwasher', 'fridge', 'microwave', 'washing_machine']

DATA_DIR = os.path.join(os.path.dirname(__file__), 'dataset')


# ---------------------------------------------------------------------------
# Spectral feature functions
# ---------------------------------------------------------------------------

def _causal_stft_bins(signal: np.ndarray,
                      win: int = STFT_WIN,
                      n_bins: int = STFT_BINS) -> np.ndarray:
    """
    Causal sliding-window FFT -> log-magnitude in 8 log-spaced frequency bands.

    For each timestep t the FFT is computed over signal[t-win+1 : t+1]
    (strictly causal: no future samples used).  The first `win` steps are
    edge-padded with signal[0].

    Frequency interpretation at 6-second sampling:
      Bin 0 (lowest):  period ~180-360 s  (3-6 min)  — fridge compressor
      Bin 7 (highest): period ~12-18 s               — switching transients
    """
    N      = len(signal)
    hann   = np.hanning(win).astype(np.float32)
    # Causal padding: prepend (win-1) copies of signal[0]
    padded = np.concatenate([[signal[0]] * (win - 1), signal])

    # Build (N, win) view without copy using stride tricks
    s      = padded.strides[0]
    wins   = np.lib.stride_tricks.as_strided(
        padded, shape=(N, win), strides=(s, s)
    ).copy() * hann                             # (N, win)

    mag    = np.abs(np.fft.rfft(wins, axis=1)).astype(np.float32)  # (N, win//2+1)

    # Log-spaced bin edges over frequency indices 1 .. win//2
    n_freq = win // 2       # 30 for win=60
    edges  = np.unique(
        np.round(np.logspace(0, np.log10(n_freq), n_bins + 1)).astype(int)
    )
    edges  = np.clip(edges, 1, n_freq)

    out    = np.zeros((N, n_bins), dtype=np.float32)
    n_valid = len(edges) - 1
    for b in range(min(n_bins, n_valid)):
        lo = int(edges[b])
        hi = int(min(edges[b + 1], n_freq + 1))
        if lo < hi:
            out[:, b] = np.log1p(mag[:, lo:hi].mean(axis=1))
        else:
            out[:, b] = np.log1p(mag[:, lo])
    # If fewer unique edges than n_bins, repeat the last valid bin
    if n_valid < n_bins:
        out[:, n_valid:] = out[:, n_valid - 1:n_valid]
    return out   # (N, n_bins)


def _wavelet_detail_coeffs(signal: np.ndarray,
                            wavelet: str = 'db4',
                            n_levels: int = N_WAVELET_LVL) -> np.ndarray:
    """
    Stationary Wavelet Transform (SWT) detail coefficients.

    SWT outputs the same length as the input at every level, avoiding the
    downsampling that makes standard DWT hard to align with the input.

    Coefficients are ordered finest to coarsest:
      col 0 (level 1): 12-24 s oscillations  — single-switch transients
      col 1 (level 2): 24-48 s
      col 2 (level 3): 48-96 s               — appliance state changes
      col 3 (level 4): 96-192 s              — fridge / WM cycle periods

    Returns zeros if PyWavelets is not installed.
    """
    N = len(signal)
    if not HAS_PYWT:
        return np.zeros((N, n_levels), dtype=np.float32)

    # SWT requires length divisible by 2^n_levels
    req     = 2 ** n_levels         # 16
    pad_len = (req - N % req) % req
    padded  = np.pad(signal.astype(np.float64), (0, pad_len), mode='edge')

    # swt returns [(cA_n, cD_n), ..., (cA_1, cD_1)] — coarsest first
    coeffs  = pywt.swt(padded, wavelet, level=n_levels, trim_approx=False)

    # Reverse so column 0 = finest (level 1) scale
    details = np.stack([c[1][:N] for c in reversed(coeffs)], axis=1)
    return details.astype(np.float32)   # (N, n_levels)


def _spectral_flux(stft_bins: np.ndarray) -> np.ndarray:
    """
    L2 norm of the frame-to-frame STFT bin difference.
    Large during appliance switching events; near-zero in steady state.
    """
    flux    = np.zeros(len(stft_bins), dtype=np.float32)
    flux[1:] = np.linalg.norm(stft_bins[1:] - stft_bins[:-1], axis=1)
    return flux


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

def _median_filter(arr: np.ndarray, k: int = MEDFILT_K) -> np.ndarray:
    return (pd.Series(arr)
            .rolling(k, center=True, min_periods=1)
            .median()
            .values.astype(np.float32))


def _ema_filter(arr: np.ndarray, span: int = EMA_SPAN) -> np.ndarray:
    return (pd.Series(arr)
            .ewm(span=span, adjust=False)
            .mean()
            .values.astype(np.float32))


def _n_step_diff(arr: np.ndarray, n: int) -> np.ndarray:
    d     = np.zeros_like(arr)
    d[n:] = arr[n:] - arr[:-n]
    return d


def compute_features(df: pd.DataFrame) -> np.ndarray:
    """
    Full 25-channel feature matrix.

    12 denoising/delta/rolling channels  (same as preprocessed script)
     + 8 STFT log-magnitude bins         (causal, 6-min window)
     + 4 wavelet detail coefficients     (SWT db4, levels 1-4)
     + 1 spectral flux
    = 25 channels
    """
    raw    = df['aggregate'].values.astype(np.float32)

    # ── Denoising ─────────────────────────────────────────────────────────────
    med    = _median_filter(raw)
    smooth = _ema_filter(med)
    resid  = (raw - smooth).astype(np.float32)

    # ── Delta features ────────────────────────────────────────────────────────
    d_raw_1    = _n_step_diff(raw,    1)
    d_smooth_1 = _n_step_diff(smooth, 1)
    d_smooth_6 = _n_step_diff(smooth, 6)

    # ── Rolling stats ─────────────────────────────────────────────────────────
    s    = pd.Series(smooth)
    rm_s = s.rolling(ROLL_S, min_periods=1).mean().values.astype(np.float32)
    rs_s = s.rolling(ROLL_S, min_periods=1).std().fillna(0).values.astype(np.float32)
    rm_l = s.rolling(ROLL_L, min_periods=1).mean().values.astype(np.float32)
    rs_l = s.rolling(ROLL_L, min_periods=1).std().fillna(0).values.astype(np.float32)

    resid_energy = (pd.Series(resid ** 2)
                    .rolling(ROLL_S, min_periods=1)
                    .mean()
                    .values.astype(np.float32))

    base = np.stack([
        raw, med, smooth, resid,
        d_raw_1, d_smooth_1, d_smooth_6,
        rm_s, rs_s, rm_l, rs_l,
        resid_energy,
    ], axis=1)   # (N, 12)

    # ── Spectral features ─────────────────────────────────────────────────────
    stft_bins = _causal_stft_bins(raw)             # (N, 8)
    wvlt_det  = _wavelet_detail_coeffs(raw)        # (N, 4)
    flux      = _spectral_flux(stft_bins)[:, None] # (N, 1)

    return np.concatenate([base, stft_bins, wvlt_det, flux], axis=1)  # (N, 25)


# ---------------------------------------------------------------------------
# Adaptive per-split thresholds  (fix #1)
# ---------------------------------------------------------------------------

def compute_adaptive_thresholds(df: pd.DataFrame) -> dict:
    """
    Per-appliance ON/OFF threshold = standby + THRESHOLD_DELTA (W).
    Standby estimated as the low_pct percentile of nonzero readings.
    Handles House 5 persistent standby draw (microwave ~25 W, WM ~14 W).
    """
    thresholds = {}
    for app in APPLIANCES:
        col     = df[app]
        nonzero = col[col > 0]
        if len(nonzero) == 0:
            thresholds[app] = THRESHOLD_MIN
        else:
            standby         = float(nonzero.quantile(THRESHOLD_LOW_PCT))
            thresholds[app] = max(standby + THRESHOLD_DELTA, THRESHOLD_MIN)
    return thresholds


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(split: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f'UKDALE_HF_{split}.csv')
    df   = pd.read_csv(path, index_col='timestamp', parse_dates=True)
    print(f"  {split:12s}: {len(df):6d} rows")
    return df


def create_sequences(feat: np.ndarray, targets: np.ndarray, stride: int):
    X, Y = [], []
    for i in range(0, len(feat) - WIN, stride):
        X.append(feat[i : i + WIN])
        Y.append(targets[i : i + WIN])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


# ---------------------------------------------------------------------------
# Physics losses
# ---------------------------------------------------------------------------

class PhysicsConsistencyLoss(nn.Module):
    """Time-domain one-sided energy conservation at every timestep."""

    def __init__(self, agg_scaler, y_scalers, epsilon_w=EPSILON_W):
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

    def forward(self, x_z, power_pred):
        x_raw = x_z[:, :, 0] * self.x_scale + self.x_mean
        p_raw = power_pred * self.y_ranges + self.y_mins
        return F.relu(p_raw.sum(dim=-1) - x_raw - self.epsilon).mean()


class FrequencyPhysicsLoss(nn.Module):
    """Spectral energy conservation via differentiable torch.fft.rfft."""

    def __init__(self, agg_scaler, y_scalers):
        super().__init__()
        self.register_buffer('x_mean',
            torch.tensor(float(agg_scaler.mean_[0]),  dtype=torch.float32))
        self.register_buffer('x_scale',
            torch.tensor(float(agg_scaler.scale_[0]), dtype=torch.float32))
        self.register_buffer('y_mins',
            torch.tensor([float(s.data_min_[0])   for s in y_scalers], dtype=torch.float32))
        self.register_buffer('y_ranges',
            torch.tensor([float(s.data_range_[0]) for s in y_scalers], dtype=torch.float32))

    def forward(self, x_z, power_pred):
        x_raw    = x_z[:, :, 0] * self.x_scale + self.x_mean
        p_raw    = power_pred * self.y_ranges + self.y_mins
        pred_sum = p_raw.sum(dim=-1)
        n        = float(pred_sum.shape[-1])
        pred_fft = (torch.fft.rfft(pred_sum, dim=-1).abs() / n).log1p()
        agg_fft  = (torch.fft.rfft(x_raw,    dim=-1).abs() / n).log1p()
        return F.mse_loss(pred_fft, agg_fft)


class EventDetectionLoss(nn.Module):
    """Per-appliance BCE-with-logits + class-weighted pos_weight."""

    def __init__(self, thresholds_scaled, pos_weights):
        super().__init__()
        self.n_apps = len(thresholds_scaled)
        self.register_buffer('thresholds',
            torch.tensor(thresholds_scaled, dtype=torch.float32))
        self.register_buffer('pos_weights',
            torch.tensor(pos_weights, dtype=torch.float32))

    def forward(self, event_logits, y_scaled):
        total = torch.zeros(1, device=event_logits.device)
        for i in range(self.n_apps):
            logit_i = event_logits[:, :, i]
            y_on_i  = (y_scaled[:, :, i] > self.thresholds[i]).float()
            pw      = self.pos_weights[i:i+1]
            total   = total + F.binary_cross_entropy_with_logits(
                logit_i, y_on_i, pos_weight=pw)
        return total / self.n_apps


# ---------------------------------------------------------------------------
# Dual-Branch CNN
# ---------------------------------------------------------------------------

class DualBranchCNN(nn.Module):
    """
    Fast branch: dilations 1, 2   (RF = 15 steps = ~90 s)
    Slow branch: dilations 1, 16, 64 (RF = 199 steps = ~20 min)
    Output: (batch, WIN, h_fast + h_slow)
    """

    def __init__(self, in_ch, h_fast, h_slow):
        super().__init__()
        self.fast = nn.Sequential(
            nn.Conv1d(in_ch,   h_fast, 7, padding=3,  dilation=1),
            nn.BatchNorm1d(h_fast), nn.GELU(),
            nn.Conv1d(h_fast,  h_fast, 5, padding=4,  dilation=2),
            nn.BatchNorm1d(h_fast), nn.GELU(),
        )
        self.slow = nn.Sequential(
            nn.Conv1d(in_ch,   h_slow, 7, padding=3,  dilation=1),
            nn.BatchNorm1d(h_slow), nn.GELU(),
            nn.Conv1d(h_slow,  h_slow, 5, padding=32, dilation=16),
            nn.BatchNorm1d(h_slow), nn.GELU(),
            nn.Conv1d(h_slow,  h_slow, 3, padding=64, dilation=64),
            nn.BatchNorm1d(h_slow), nn.GELU(),
        )

    def forward(self, x):
        xp = x.permute(0, 2, 1)
        return torch.cat([
            self.fast(xp).permute(0, 2, 1),
            self.slow(xp).permute(0, 2, 1),
        ], dim=-1)


# ---------------------------------------------------------------------------
# Liquid Cell
# ---------------------------------------------------------------------------

class LiquidCell(nn.Module):
    def __init__(self, input_size, hidden_size, dt=0.1):
        super().__init__()
        self.dt          = dt
        self.hidden_size = hidden_size
        self.input_proj  = nn.Linear(input_size, hidden_size)
        self.tau_base    = nn.Parameter(torch.ones(hidden_size))
        self.tau_mod     = nn.Linear(input_size, hidden_size)
        self.rec_weights = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.rec_weights)
        self.gate        = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x_t, h):
        ip  = self.input_proj(x_t)
        rp  = h @ self.rec_weights
        tb  = F.softplus(self.tau_base).unsqueeze(0)
        tm  = torch.sigmoid(self.tau_mod(x_t))
        tau = (tb * tm).clamp(min=self.dt)
        g   = torch.sigmoid(self.gate(torch.cat([x_t, h], dim=1)))
        dh  = ((-h / tau) + g * torch.tanh(ip + rp)) * self.dt
        return (h + dh).clamp(-10.0, 10.0)


# ---------------------------------------------------------------------------
# Multi-Resolution Seq2Seq model
# ---------------------------------------------------------------------------

class SpectralSeq2SeqNet(nn.Module):
    """
    DualBranchCNN (in_ch=25) -> BiLNN -> power heads + event heads.
    Identical to MultiResolutionSeq2SeqNet; renamed to reflect the richer input.
    """

    def __init__(self, in_ch, hidden, n_apps, dt=0.1):
        super().__init__()
        h_fast = hidden // 2
        h_slow = hidden - h_fast

        self.cnn         = DualBranchCNN(in_ch, h_fast, h_slow)
        self.fwd_cell    = LiquidCell(hidden, hidden, dt)
        self.bwd_cell    = LiquidCell(hidden, hidden, dt)
        self.norm        = nn.LayerNorm(hidden * 2)
        self.power_heads = nn.ModuleList([nn.Linear(hidden * 2, 1) for _ in range(n_apps)])
        self.event_heads = nn.ModuleList([nn.Linear(hidden * 2, 1) for _ in range(n_apps)])
        self.hidden      = hidden
        self.n_apps      = n_apps

    def forward(self, x):
        feat  = self.cnn(x)
        batch, T, _ = feat.shape

        h_f = torch.zeros(batch, self.hidden, device=x.device)
        fwd = []
        for t in range(T):
            h_f = self.fwd_cell(feat[:, t, :], h_f)
            fwd.append(h_f)

        h_b  = torch.zeros(batch, self.hidden, device=x.device)
        bwd  = [None] * T
        for t in reversed(range(T)):
            h_b    = self.bwd_cell(feat[:, t, :], h_b)
            bwd[t] = h_b

        power_list, event_list = [], []
        for t in range(T):
            h_t = self.norm(torch.cat([fwd[t], bwd[t]], dim=1))
            power_list.append(
                torch.cat([torch.sigmoid(head(h_t)) for head in self.power_heads], dim=1))
            event_list.append(
                torch.cat([head(h_t) for head in self.event_heads], dim=1))

        return torch.stack(power_list, dim=1), torch.stack(event_list, dim=1)


# ---------------------------------------------------------------------------
# Dataset / trace helpers
# ---------------------------------------------------------------------------

class Seq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


def reconstruct_trace(window_preds, n_total, stride, win):
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


def per_app_metrics(y_true, y_pred, y_scalers, thresholds):
    out = {}
    for i, app in enumerate(APPLIANCES):
        raw_t = y_scalers[i].inverse_transform(y_true[:, i:i+1]).flatten()
        raw_p = y_scalers[i].inverse_transform(y_pred[:, i:i+1]).flatten()
        out[app] = calculate_nilm_metrics(raw_t, raw_p, threshold=thresholds[app])
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(save_dir,
          hidden        = 64,
          dt            = 0.1,
          lambda_phys_t = LAMBDA_PHYS_T,
          lambda_phys_f = LAMBDA_PHYS_F,
          lambda_event  = LAMBDA_EVENT,
          epsilon_w     = EPSILON_W):

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}  |  WIN={WIN}  hidden={hidden}  N_FEAT={N_FEAT}")
    print(f"  PyWavelets available: {HAS_PYWT}")

    # ── Load ─────────────────────────────────────────────────────────────────
    print("\nLoading data ...")
    df_tr = load_csv('train')
    df_va = load_csv('validation')
    df_te = load_csv('test')
    n_tr, n_va, n_te = len(df_tr), len(df_va), len(df_te)

    # ── Adaptive thresholds ──────────────────────────────────────────────────
    tr_thresholds = compute_adaptive_thresholds(df_tr)
    va_thresholds = compute_adaptive_thresholds(df_va)
    te_thresholds = compute_adaptive_thresholds(df_te)
    print("\n  Adaptive ON/OFF thresholds (W):")
    print(f"  {'Appliance':<22} {'Train':>8} {'Val':>8} {'Test':>8}")
    for app in APPLIANCES:
        print(f"  {app:<22} {tr_thresholds[app]:>8.1f} "
              f"{va_thresholds[app]:>8.1f} {te_thresholds[app]:>8.1f}")

    # ── Feature extraction ───────────────────────────────────────────────────
    print("\nComputing spectral features (STFT + wavelet + flux) ...")
    feat_tr = compute_features(df_tr)
    feat_va = compute_features(df_va)
    feat_te = compute_features(df_te)
    print(f"  Feature shape per split: {feat_tr.shape}  (N x {N_FEAT})")
    print(f"  STFT bins      : {STFT_BINS}  (causal win={STFT_WIN} steps = {STFT_WIN*6}s)")
    print(f"  Wavelet levels : {N_WAVELET_LVL} (db4 SWT)"
          if HAS_PYWT else
          f"  Wavelet        : DISABLED (pip install PyWavelets)")

    # ── Targets ──────────────────────────────────────────────────────────────
    tgt_tr = df_tr[APPLIANCES].values.astype(np.float32)
    tgt_va = df_va[APPLIANCES].values.astype(np.float32)
    tgt_te = df_te[APPLIANCES].values.astype(np.float32)

    # ── Sequences ────────────────────────────────────────────────────────────
    X_tr, Y_tr = create_sequences(feat_tr, tgt_tr, STRIDE)
    X_va, Y_va = create_sequences(feat_va, tgt_va, STRIDE)
    X_te, Y_te = create_sequences(feat_te, tgt_te, WIN)
    print(f"\nTrain : {X_tr.shape}  ({X_tr.shape[0]*WIN:,} predictions)")
    print(f"Val   : {X_va.shape}")
    print(f"Test  : {X_te.shape}  [non-overlapping]")

    # ── Z-score for inputs ───────────────────────────────────────────────────
    feat_scalers = []
    for ch in range(N_FEAT):
        sc = StandardScaler()
        X_tr[:, :, ch] = sc.fit_transform(X_tr[:, :, ch].reshape(-1, 1)).reshape(-1, WIN)
        X_va[:, :, ch] = sc.transform(    X_va[:, :, ch].reshape(-1, 1)).reshape(-1, WIN)
        X_te[:, :, ch] = sc.transform(    X_te[:, :, ch].reshape(-1, 1)).reshape(-1, WIN)
        feat_scalers.append(sc)
    agg_scaler = feat_scalers[0]
    print(f"\n  Agg Z-score: mean={agg_scaler.mean_[0]:.1f} W  std={agg_scaler.scale_[0]:.1f} W")

    # ── MinMax for targets ───────────────────────────────────────────────────
    y_scalers = []
    for i in range(len(APPLIANCES)):
        ys = MinMaxScaler()
        Y_tr[:, :, i] = ys.fit_transform(Y_tr[:, :, i].reshape(-1, 1)).reshape(-1, WIN)
        Y_va[:, :, i] = ys.transform(    Y_va[:, :, i].reshape(-1, 1)).reshape(-1, WIN)
        Y_te[:, :, i] = ys.transform(    Y_te[:, :, i].reshape(-1, 1)).reshape(-1, WIN)
        y_scalers.append(ys)

    thresholds_scaled = [
        (tr_thresholds[app] - float(y_scalers[i].data_min_[0]))
        / float(y_scalers[i].data_range_[0])
        for i, app in enumerate(APPLIANCES)
    ]

    # ── Event pos_weights from training ON fraction ──────────────────────────
    pos_weights = []
    print("\n  Event pos_weight per appliance:")
    for i, app in enumerate(APPLIANCES):
        flat  = Y_tr[:, :, i].flatten()
        n_on  = float((flat > thresholds_scaled[i]).sum())
        n_off = float((flat <= thresholds_scaled[i]).sum())
        pw    = float(np.clip(n_off / max(n_on, 1.0), *POS_WEIGHT_CLAMP))
        pos_weights.append(pw)
        print(f"    {app:<22}  on={100*n_on/(n_on+n_off):5.1f}%  pos_weight={pw:6.1f}")

    # ── DataLoaders ──────────────────────────────────────────────────────────
    tr_ld = torch.utils.data.DataLoader(
        Seq2SeqDataset(X_tr, Y_tr), batch_size=BATCH, shuffle=True,  drop_last=False)
    va_ld = torch.utils.data.DataLoader(
        Seq2SeqDataset(X_va, Y_va), batch_size=BATCH, shuffle=False, drop_last=False)
    te_ld = torch.utils.data.DataLoader(
        Seq2SeqDataset(X_te, Y_te), batch_size=BATCH, shuffle=False, drop_last=False)

    # ── Model + losses ───────────────────────────────────────────────────────
    model       = SpectralSeq2SeqNet(N_FEAT, hidden, len(APPLIANCES), dt).to(device)
    mse_crit    = nn.MSELoss()
    phys_t_crit = PhysicsConsistencyLoss(agg_scaler, y_scalers, epsilon_w).to(device)
    phys_f_crit = FrequencyPhysicsLoss(agg_scaler, y_scalers).to(device)
    event_crit  = EventDetectionLoss(thresholds_scaled, pos_weights).to(device)
    opt         = torch.optim.Adam(model.parameters(), lr=LR)
    sched       = torch.optim.lr_scheduler.ReduceLROnPlateau(
                      opt, mode='min', factor=0.5, patience=8, min_lr=1e-5)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")
    print(f"Warmup: MSE-only for first {WARMUP_EPOCHS} epochs.\n")

    history = {k: [] for k in [
        'train_loss', 'train_mse', 'train_phys_t', 'train_phys_f', 'train_event',
        'val_loss',   'val_mse',   'val_phys_t',   'val_phys_f',   'val_metrics',
    ]}
    best_val_mse = float('inf')
    best_state   = None
    counter      = 0

    for epoch in range(EPOCHS):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        ep_tot = ep_mse = ep_pt = ep_pf = ep_ev = 0.0
        pbar = tqdm(tr_ld, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            power_pred, event_logits = model(xb)

            l_mse    = mse_crit(power_pred, yb)
            l_phys_t = phys_t_crit(xb, power_pred)
            l_phys_f = phys_f_crit(xb, power_pred)
            l_event  = event_crit(event_logits, yb)

            if epoch < WARMUP_EPOCHS:
                loss = l_mse
            else:
                loss = (l_mse
                        + lambda_phys_t * l_phys_t
                        + lambda_phys_f * l_phys_f
                        + lambda_event  * l_event)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_tot += loss.item();  ep_mse += l_mse.item()
            ep_pt  += l_phys_t.item(); ep_pf += l_phys_f.item()
            ep_ev  += l_event.item()
            pbar.set_postfix({'mse': f'{l_mse.item():.4f}',
                              'pt': f'{l_phys_t.item():.4f}',
                              'pf': f'{l_phys_f.item():.4f}',
                              'ev': f'{l_event.item():.4f}'})

        nb = len(tr_ld)
        history['train_loss'].append(ep_tot / nb)
        history['train_mse'].append(ep_mse / nb)
        history['train_phys_t'].append(ep_pt / nb)
        history['train_phys_f'].append(ep_pf / nb)
        history['train_event'].append(ep_ev / nb)

        # ── Validate ────────────────────────────────────────────────────────
        model.eval()
        vl_tot = vl_mse = vl_pt = vl_pf = 0.0
        va_preds, va_trues = [], []

        with torch.no_grad():
            for xb, yb in va_ld:
                xb, yb = xb.to(device), yb.to(device)
                power_pred, _ = model(xb)
                l_mse    = mse_crit(power_pred, yb)
                l_phys_t = phys_t_crit(xb, power_pred)
                l_phys_f = phys_f_crit(xb, power_pred)
                vl_mse  += l_mse.item()
                vl_pt   += l_phys_t.item()
                vl_pf   += l_phys_f.item()
                vl_tot  += (l_mse + lambda_phys_t * l_phys_t + lambda_phys_f * l_phys_f).item()
                for b in range(power_pred.shape[0]):
                    va_preds.append(power_pred[b].cpu().numpy())
                    va_trues.append(yb[b].cpu().numpy())

        nv = len(va_ld)
        avg_va_mse = vl_mse / nv
        history['val_loss'].append(vl_tot / nv)
        history['val_mse'].append(avg_va_mse)
        history['val_phys_t'].append(vl_pt / nv)
        history['val_phys_f'].append(vl_pf / nv)
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
            f"pt={history['train_phys_t'][-1]:.5f} "
            f"pf={history['train_phys_f'][-1]:.5f} "
            f"ev={history['train_event'][-1]:.5f})  "
            f"val_mse={avg_va_mse:.5f}  "
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

    print(f"\n{'Appliance':<24} {'F1':>8} {'Precision':>10} "
          f"{'Recall':>8} {'MAE':>8} {'SAE':>8}")
    print("-" * 72)
    for app in APPLIANCES:
        m = test_metrics[app]
        print(f"{app:<24} {m['f1']:>8.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['mae']:>8.2f} {m['sae']:>8.4f}")

    _plot_spectral_features(df_tr, save_dir)
    _plot_loss(history, save_dir)
    _plot_metrics(history, test_metrics, save_dir)
    _plot_trace(true_trace_te, pred_trace_te, y_scalers, save_dir)

    cfg = {
        'dataset':  'UKDALE HF 6s',
        'model':    'SpectralSeq2SeqNet',
        'features': {
            'n_feat': N_FEAT,
            'base_channels': N_FEAT_BASE,
            'stft_bins':  STFT_BINS,
            'stft_win_steps': STFT_WIN,
            'stft_win_seconds': STFT_WIN * 6,
            'wavelet_levels': N_WAVELET_LVL if HAS_PYWT else 0,
            'wavelet': 'db4 SWT' if HAS_PYWT else 'disabled',
            'spectral_flux': True,
        },
        'thresholds': {'train': tr_thresholds, 'val': va_thresholds, 'test': te_thresholds},
        'pos_weights': dict(zip(APPLIANCES, [float(w) for w in pos_weights])),
        'architecture': {
            'cnn': 'DualBranchCNN fast=[d1,d2] RF=90s / slow=[d1,d16,d64] RF=~20min',
            'lnn': 'Bidirectional LiquidCell',
            'output': 'Seq2Seq power heads (sigmoid) + event heads (logits)',
        },
        'window': {'win': WIN, 'stride_train': STRIDE, 'stride_test': WIN},
        'model_params': {'in_ch': N_FEAT, 'hidden': hidden,
                         'n_apps': len(APPLIANCES), 'dt': dt},
        'test_metrics': {
            app: {k: float(v) for k, v in m.items()}
            for app, m in test_metrics.items()
        },
    }
    with open(os.path.join(save_dir, 'spectral_results.json'), 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=4)
    print(f"\nResults saved to: {save_dir}")
    return test_metrics, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_spectral_features(df: pd.DataFrame, save_dir: str,
                             n_steps: int = 600) -> None:
    """Visualise the three spectral feature groups on 1 hour of training data."""
    raw   = df['aggregate'].values[:n_steps].astype(np.float32)
    stft  = _causal_stft_bins(raw)       # (n_steps, 8)
    wvlt  = _wavelet_detail_coeffs(raw)  # (n_steps, 4)
    flux  = _spectral_flux(stft)          # (n_steps,)
    t     = np.arange(n_steps) * 6 / 60  # minutes

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Spectral Features -- Training Data (first 1 hour)', fontsize=11)

    axes[0].plot(t, raw, color='steelblue', linewidth=0.8, label='Raw aggregate')
    axes[0].set_ylabel('Power (W)'); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    im = axes[1].imshow(stft.T, aspect='auto', origin='lower',
                         extent=[0, t[-1], 0, STFT_BINS - 1], cmap='viridis')
    axes[1].set_ylabel('STFT bin\n(low=slow, high=fast)')
    plt.colorbar(im, ax=axes[1], label='log1p(mag)')

    if HAS_PYWT:
        for lvl in range(N_WAVELET_LVL):
            axes[2].plot(t, wvlt[:, lvl],
                         label=f'Level {lvl+1} ({12*(2**lvl)}-{24*(2**lvl)} s)',
                         linewidth=0.8)
        axes[2].set_ylabel('Wavelet detail')
        axes[2].legend(fontsize=7, ncol=2); axes[2].grid(alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'PyWavelets not installed',
                     ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_ylabel('Wavelet detail')

    axes[3].plot(t, flux, color='tomato', linewidth=0.8, label='Spectral flux')
    axes[3].set_ylabel('Flux (L2)'); axes[3].set_xlabel('Time (min)')
    axes[3].legend(fontsize=8); axes[3].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'spectral_features.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def _plot_loss(history, save_dir):
    ep   = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle('Spectral PINN-LNN -- Loss Components')
    pairs = [
        ('train_loss',   'val_loss',   'Total Loss'),
        ('train_mse',    'val_mse',    'MSE'),
        ('train_phys_t', 'val_phys_t', 'Physics Time'),
        ('train_phys_f', 'val_phys_f', 'Physics Freq'),
        ('train_event',  None,         'Event BCE (train)'),
    ]
    for ax, (tr_k, va_k, title) in zip(axes, pairs):
        ax.plot(ep, history[tr_k], label='Train', color='steelblue')
        if va_k:
            ax.plot(ep, history[va_k], label='Val', color='tomato')
        ax.set_title(title); ax.set_xlabel('Epoch')
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'spectral_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _plot_metrics(history, test_metrics, save_dir):
    ep  = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(len(APPLIANCES), 2, figsize=(12, 4 * len(APPLIANCES)))
    fig.suptitle('Spectral PINN-LNN -- Per-Appliance Validation Metrics')
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
    plt.savefig(os.path.join(save_dir, 'spectral_per_appliance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def _plot_trace(true_trace, pred_trace, y_scalers, save_dir, n_steps=600):
    fig, axes = plt.subplots(len(APPLIANCES), 1, figsize=(14, 3 * len(APPLIANCES)))
    fig.suptitle('Spectral PINN-LNN -- Reconstructed Test Trace (1 hour)')
    t = np.arange(n_steps) * 6 / 60
    for row, app in enumerate(APPLIANCES):
        i     = APPLIANCES.index(app)
        raw_t = y_scalers[i].inverse_transform(true_trace[:n_steps, i:i+1]).flatten()
        raw_p = y_scalers[i].inverse_transform(pred_trace[:n_steps, i:i+1]).flatten()
        axes[row].plot(t, raw_t, label='Ground truth', color='steelblue', linewidth=1.0, alpha=0.8)
        axes[row].plot(t, raw_p, label='Prediction',   color='tomato',    linewidth=1.0, alpha=0.8)
        axes[row].set_title(app); axes[row].set_ylabel('Power (W)')
        axes[row].legend(loc='upper right'); axes[row].grid(alpha=0.3)
    axes[-1].set_xlabel('Time (min)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'spectral_trace.png'), dpi=150, bbox_inches='tight')
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

    if not HAS_PYWT:
        print("Tip: install PyWavelets for wavelet features:")
        print("     pip install PyWavelets")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir  = f'models/spectral_pinn_lnn_{timestamp}'

    train(
        save_dir      = save_dir,
        hidden        = 64,
        dt            = 0.1,
        lambda_phys_t = LAMBDA_PHYS_T,
        lambda_phys_f = LAMBDA_PHYS_F,
        lambda_event  = LAMBDA_EVENT,
        epsilon_w     = EPSILON_W,
    )
