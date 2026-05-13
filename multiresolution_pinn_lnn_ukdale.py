"""
Multi-Resolution PINN-LNN for UKDALE NILM
==========================================

Three new architectural ideas beyond preprocessed_pinn_lnn_ukdale.py:

1. Dual-Branch CNN  (fast + slow temporal resolution)
   ─────────────────────────────────────────────────
   Fast branch: dilations [1, 2]       RF = 15 steps  (~90 s)
   Slow branch: dilations [1, 16, 64]  RF = 199 steps (~20 min)

   Appliance switching events leave sharp transients visible at 6 s resolution
   (fast branch), but cycle-level on/off patterns span 5-20 minutes and
   require a larger receptive field (slow branch).  Concatenating both gives
   the BiLNN access to multi-scale temporal context simultaneously.

2. Event Detection Auxiliary Loss
   ────────────────────────────────
   Each appliance gets a second Linear(hidden*2, 1) head that predicts a
   binary ON/OFF logit at every timestep.  Training uses BCE-with-logits and
   a class-weighted pos_weight calibrated from the per-appliance ON fraction
   in the training set (clamped to [1, 50] to avoid extreme weighting for
   very rare events like microwave).

   This auxiliary task forces the model to learn appliance-level state
   boundaries, not just mean power, which sharpens the MSE predictions
   especially for appliances with bimodal power distributions.

3. Frequency-Domain Physics Loss
   ────────────────────────────────
   L_phys_freq = MSE( log1p(|FFT(sum(p_hat_raw))| / WIN),
                      log1p(|FFT(P_agg_raw)|       / WIN) )

   Differentiable via torch.fft.rfft (PyTorch autograd supports complex FFT).
   Enforces spectral energy conservation across all frequency components:
   the aggregate spectrum must match the sum-of-appliances spectrum, not just
   their instantaneous values.  log1p compression prevents the DC component
   from dominating the loss.

Combined loss:
   L = L_MSE  +  lambda_t * L_phys_time  +  lambda_f * L_phys_freq
              +  lambda_e * L_event
   Warmup: MSE-only for the first WARMUP_EPOCHS epochs.

Same preprocessing pipeline as preprocessed_pinn_lnn_ukdale.py:
   raw -> median filter (k=5) -> EMA (span=10) -> residual
   12 input channels, Z-score for inputs, MinMax for targets.
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
EMA_SPAN      = 10       # EMA span: 10 steps = 60 s  (alpha ~ 0.18)
ROLL_S        = 10       # short rolling window: 60 s
ROLL_L        = 50       # long  rolling window:  5 min

LAMBDA_PHYS_T = 0.01     # time-domain physics loss weight
LAMBDA_PHYS_F = 0.005    # frequency-domain physics loss weight
LAMBDA_EVENT  = 0.05     # event detection loss weight (applied per appliance)
EPSILON_W     = 50.0     # slack for time-domain physics (unlabelled background)
WARMUP_EPOCHS = 20       # epochs with MSE-only before adding auxiliary losses

# Fast-branch RF:  (7-1)*1 + (5-1)*2 + 1 = 15 steps  = 90 s
# Slow-branch RF:  (7-1)*1 + (5-1)*16 + (3-1)*64 + 1 = 199 steps = ~20 min
FAST_DILATIONS = [1, 2]
SLOW_DILATIONS = [1, 16, 64]

APPLIANCES = ['dishwasher', 'fridge', 'microwave', 'washing_machine']

THRESHOLDS = {
    'dishwasher':      10.0,
    'fridge':          10.0,
    'microwave':       10.0,
    'washing_machine': 10.0,
}

POS_WEIGHT_CLAMP = (1.0, 50.0)   # clamp range for per-appliance pos_weight

DATA_DIR = os.path.join(os.path.dirname(__file__), 'dataset')


# ---------------------------------------------------------------------------
# Preprocessing pipeline  (identical to preprocessed_pinn_lnn_ukdale.py)
# ---------------------------------------------------------------------------

def _median_filter(arr: np.ndarray, k: int = MEDFILT_K) -> np.ndarray:
    s = pd.Series(arr)
    return s.rolling(k, center=True, min_periods=1).median().values.astype(np.float32)


def _ema_filter(arr: np.ndarray, span: int = EMA_SPAN) -> np.ndarray:
    return pd.Series(arr).ewm(span=span, adjust=False).mean().values.astype(np.float32)


def _n_step_diff(arr: np.ndarray, n: int) -> np.ndarray:
    d     = np.zeros_like(arr)
    d[n:] = arr[n:] - arr[:-n]
    return d


def compute_features(df: pd.DataFrame) -> np.ndarray:
    """Full 12-channel preprocessing pipeline -> (N, 12) feature matrix."""
    raw    = df['aggregate'].values.astype(np.float32)
    med    = _median_filter(raw, MEDFILT_K)
    smooth = _ema_filter(med, EMA_SPAN)
    resid  = (raw - smooth).astype(np.float32)

    d_raw_1    = _n_step_diff(raw,    1)
    d_smooth_1 = _n_step_diff(smooth, 1)
    d_smooth_6 = _n_step_diff(smooth, 6)

    s    = pd.Series(smooth)
    rm_s = s.rolling(ROLL_S, min_periods=1).mean().values.astype(np.float32)
    rs_s = s.rolling(ROLL_S, min_periods=1).std().fillna(0).values.astype(np.float32)
    rm_l = s.rolling(ROLL_L, min_periods=1).mean().values.astype(np.float32)
    rs_l = s.rolling(ROLL_L, min_periods=1).std().fillna(0).values.astype(np.float32)

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
# Dual-Branch CNN
# ---------------------------------------------------------------------------

class DualBranchCNN(nn.Module):
    """
    Fast branch: dilations 1, 2   (RF = 15 steps = ~90 s)
    Slow branch: dilations 1, 16, 64 (RF = 199 steps = ~20 min)

    Both branches receive the same input x: (batch, WIN, in_ch).
    Outputs are concatenated along the channel dimension:
        out: (batch, WIN, h_fast + h_slow)
    so total output channels equal the desired hidden size.
    """

    def __init__(self, in_ch: int, h_fast: int, h_slow: int):
        super().__init__()
        # Fast branch -- small dilations, catches 6s-90s transients
        self.fast = nn.Sequential(
            nn.Conv1d(in_ch,   h_fast, kernel_size=7, padding=3,  dilation=1),
            nn.BatchNorm1d(h_fast), nn.GELU(),
            nn.Conv1d(h_fast,  h_fast, kernel_size=5, padding=4,  dilation=2),
            nn.BatchNorm1d(h_fast), nn.GELU(),
        )
        # Slow branch -- large dilations, catches 5-20 min usage patterns
        self.slow = nn.Sequential(
            nn.Conv1d(in_ch,   h_slow, kernel_size=7, padding=3,  dilation=1),
            nn.BatchNorm1d(h_slow), nn.GELU(),
            nn.Conv1d(h_slow,  h_slow, kernel_size=5, padding=32, dilation=16),
            nn.BatchNorm1d(h_slow), nn.GELU(),
            nn.Conv1d(h_slow,  h_slow, kernel_size=3, padding=64, dilation=64),
            nn.BatchNorm1d(h_slow), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, WIN, in_ch) -> permute to (batch, in_ch, WIN) for Conv1d
        xp       = x.permute(0, 2, 1)
        fast_out = self.fast(xp).permute(0, 2, 1)   # (batch, WIN, h_fast)
        slow_out = self.slow(xp).permute(0, 2, 1)   # (batch, WIN, h_slow)
        return torch.cat([fast_out, slow_out], dim=-1)  # (batch, WIN, h_fast+h_slow)


# ---------------------------------------------------------------------------
# Liquid Cell  (same AdvancedLiquidCell as preceding scripts)
# ---------------------------------------------------------------------------

class LiquidCell(nn.Module):
    """Single-step continuous-time LNN cell with input-dependent time constant."""

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
# Multi-Resolution Seq2Seq model
# ---------------------------------------------------------------------------

class MultiResolutionSeq2SeqNet(nn.Module):
    """
    DualBranchCNN  ->  Bidirectional LNN  ->  power heads  +  event heads.

    Power heads:  n_apps x Linear(hidden*2, 1) + sigmoid  => (batch, WIN, n_apps)
    Event heads:  n_apps x Linear(hidden*2, 1)            => (batch, WIN, n_apps) logits

    h_fast = hidden // 2   (fast CNN channels)
    h_slow = hidden - h_fast  (slow CNN channels)
    Total CNN output = hidden, same as preceding single-branch scripts.
    """

    def __init__(self, in_ch: int, hidden: int, n_apps: int, dt: float = 0.1):
        super().__init__()
        h_fast = hidden // 2
        h_slow = hidden - h_fast   # = hidden // 2 when hidden is even

        self.cnn      = DualBranchCNN(in_ch, h_fast, h_slow)
        self.fwd_cell = LiquidCell(hidden, hidden, dt)
        self.bwd_cell = LiquidCell(hidden, hidden, dt)

        self.norm        = nn.LayerNorm(hidden * 2)
        self.power_heads = nn.ModuleList([nn.Linear(hidden * 2, 1) for _ in range(n_apps)])
        self.event_heads = nn.ModuleList([nn.Linear(hidden * 2, 1) for _ in range(n_apps)])

        self.hidden = hidden
        self.n_apps = n_apps

    def forward(self, x: torch.Tensor):
        """
        x: (batch, WIN, in_ch)
        Returns:
            power_pred:   (batch, WIN, n_apps)  in [0, 1] (sigmoid applied)
            event_logits: (batch, WIN, n_apps)  raw logits for BCE
        """
        feat  = self.cnn(x)          # (batch, WIN, hidden)
        batch = feat.shape[0]
        T     = feat.shape[1]

        # Forward LNN pass
        h_f = torch.zeros(batch, self.hidden, device=x.device)
        fwd = []
        for t in range(T):
            h_f = self.fwd_cell(feat[:, t, :], h_f)
            fwd.append(h_f)

        # Backward LNN pass
        h_b  = torch.zeros(batch, self.hidden, device=x.device)
        bwd  = [None] * T
        for t in reversed(range(T)):
            h_b    = self.bwd_cell(feat[:, t, :], h_b)
            bwd[t] = h_b

        # Per-timestep heads
        power_list = []
        event_list = []
        for t in range(T):
            h_t = self.norm(torch.cat([fwd[t], bwd[t]], dim=1))  # (batch, hidden*2)
            p_t = torch.cat(
                [torch.sigmoid(head(h_t)) for head in self.power_heads], dim=1
            )  # (batch, n_apps)
            e_t = torch.cat(
                [head(h_t) for head in self.event_heads], dim=1
            )  # (batch, n_apps) -- raw logits
            power_list.append(p_t)
            event_list.append(e_t)

        power_pred   = torch.stack(power_list, dim=1)  # (batch, WIN, n_apps)
        event_logits = torch.stack(event_list, dim=1)  # (batch, WIN, n_apps)
        return power_pred, event_logits


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class PhysicsConsistencyLoss(nn.Module):
    """
    Time-domain one-sided energy conservation at every timestep:
        L_phys_t = mean_{batch,t}( ReLU( sum_i(p_hat_raw_i(t)) - P_agg_raw(t) - eps ) )

    Inverse transform for Z-score inputs:  x_raw = x_z * scale + mean
    Inverse transform for MinMax targets:  p_raw = p_scaled * range + min
    """

    def __init__(self, agg_scaler: StandardScaler,
                 y_scalers: list,
                 epsilon_w: float = EPSILON_W):
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
        x_raw = x_z[:, :, 0] * self.x_scale + self.x_mean     # (batch, WIN) Watts
        p_raw = power_pred * self.y_ranges + self.y_mins        # (batch, WIN, n_apps)
        return F.relu(p_raw.sum(dim=-1) - x_raw - self.epsilon).mean()


class FrequencyPhysicsLoss(nn.Module):
    """
    Frequency-domain physics loss — spectral energy conservation.

    Enforces that the spectrum of the predicted appliance sum matches
    the spectrum of the measured aggregate at every frequency bin:

        L_phys_f = MSE( log1p(|FFT(sum_p_hat_raw)| / WIN),
                        log1p(|FFT(P_agg_raw)|       / WIN) )

    log1p compression prevents the large DC component from dominating.
    Division by WIN normalises FFT magnitudes to the same Watt scale as
    the original power signals, making the loss dataset-independent.
    Differentiable via torch.fft.rfft (PyTorch complex autograd).
    """

    def __init__(self, agg_scaler: StandardScaler, y_scalers: list):
        super().__init__()
        self.register_buffer('x_mean',
            torch.tensor(float(agg_scaler.mean_[0]),  dtype=torch.float32))
        self.register_buffer('x_scale',
            torch.tensor(float(agg_scaler.scale_[0]), dtype=torch.float32))
        self.register_buffer('y_mins',
            torch.tensor([float(s.data_min_[0])   for s in y_scalers], dtype=torch.float32))
        self.register_buffer('y_ranges',
            torch.tensor([float(s.data_range_[0]) for s in y_scalers], dtype=torch.float32))

    def forward(self, x_z: torch.Tensor, power_pred: torch.Tensor) -> torch.Tensor:
        x_raw    = x_z[:, :, 0] * self.x_scale + self.x_mean   # (batch, WIN) Watts
        p_raw    = power_pred * self.y_ranges + self.y_mins      # (batch, WIN, n_apps)
        pred_sum = p_raw.sum(dim=-1)                             # (batch, WIN)

        n        = float(pred_sum.shape[-1])
        pred_fft = (torch.fft.rfft(pred_sum, dim=-1).abs() / n).log1p()
        agg_fft  = (torch.fft.rfft(x_raw,    dim=-1).abs() / n).log1p()
        return F.mse_loss(pred_fft, agg_fft)


class EventDetectionLoss(nn.Module):
    """
    Per-appliance binary cross-entropy with class-weighted pos_weight.

    event_logits: (batch, WIN, n_apps)  -- raw logits
    y_scaled:     (batch, WIN, n_apps)  -- MinMax-scaled targets

    Binary target:  y_on_i = (y_scaled[:,:,i] > threshold_scaled_i).float()
    Loss:           BCE_with_logits(logit_i, y_on_i, pos_weight=w_i)

    Averaged across appliances so lambda_e scales comparably to L_MSE.
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
            logit_i = event_logits[:, :, i]                          # (batch, WIN)
            y_on_i  = (y_scaled[:, :, i] > self.thresholds[i]).float()
            pw      = self.pos_weights[i:i+1]                        # scalar tensor
            total   = total + F.binary_cross_entropy_with_logits(
                logit_i, y_on_i, pos_weight=pw
            )
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
# Trace reconstruction
# ---------------------------------------------------------------------------

def reconstruct_trace(window_preds: list, n_total: int,
                      stride: int, win: int) -> np.ndarray:
    n_apps = window_preds[0].shape[1]
    acc    = np.zeros((n_total, n_apps), dtype=np.float64)
    count  = np.zeros((n_total, 1),     dtype=np.float64)
    for idx, pw in enumerate(window_preds):
        s = idx * stride
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

def train(save_dir:      str,
          hidden:        int   = 64,
          dt:            float = 0.1,
          lambda_phys_t: float = LAMBDA_PHYS_T,
          lambda_phys_f: float = LAMBDA_PHYS_F,
          lambda_event:  float = LAMBDA_EVENT,
          epsilon_w:     float = EPSILON_W) -> tuple:

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}  |  WIN={WIN}  hidden={hidden}  dt={dt}")
    print(f"  Fast branch RF: {(7-1)*1 + (5-1)*2 + 1} steps  "
          f"({((7-1)*1 + (5-1)*2 + 1)*6}s  ~{((7-1)*1 + (5-1)*2 + 1)*6/60:.0f} min)")
    print(f"  Slow branch RF: {(7-1)*1 + (5-1)*16 + (3-1)*64 + 1} steps  "
          f"({((7-1)*1 + (5-1)*16 + (3-1)*64 + 1)*6}s  "
          f"~{((7-1)*1 + (5-1)*16 + (3-1)*64 + 1)*6/60:.0f} min)")

    # ── Load and preprocess ───────────────────────────────────────────────────
    print("\nLoading data ...")
    df_tr = load_csv('train')
    df_va = load_csv('validation')
    df_te = load_csv('test')
    n_tr, n_va, n_te = len(df_tr), len(df_va), len(df_te)

    print("Computing features (median + EMA + delta + rolling stats) ...")
    feat_tr = compute_features(df_tr)
    feat_va = compute_features(df_va)
    feat_te = compute_features(df_te)
    n_feat  = feat_tr.shape[1]   # 12

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

    # ── Z-score for input features ────────────────────────────────────────────
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

    agg_scaler = feat_scalers[0]
    print(f"\n  Agg Z-score:  mean={agg_scaler.mean_[0]:.1f} W  "
          f"std={agg_scaler.scale_[0]:.1f} W")

    # ── MinMax for targets ────────────────────────────────────────────────────
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

    # ── Calibrate pos_weights from training ON fraction ───────────────────────
    pos_weights = []
    print("\n  Event pos_weight per appliance (clamped to "
          f"[{POS_WEIGHT_CLAMP[0]:.0f}, {POS_WEIGHT_CLAMP[1]:.0f}]):")
    for i, app in enumerate(APPLIANCES):
        flat   = Y_tr[:, :, i].flatten()
        n_on   = float((flat > thresholds_scaled[i]).sum())
        n_off  = float((flat <= thresholds_scaled[i]).sum())
        raw_pw = n_off / max(n_on, 1.0)
        pw     = float(np.clip(raw_pw, *POS_WEIGHT_CLAMP))
        pos_weights.append(pw)
        on_pct = 100.0 * n_on / (n_on + n_off)
        print(f"    {app:<22}  on={on_pct:5.1f}%  pos_weight={pw:6.1f}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    tr_ld = torch.utils.data.DataLoader(
        Seq2SeqDataset(X_tr, Y_tr), batch_size=BATCH, shuffle=True,  drop_last=False)
    va_ld = torch.utils.data.DataLoader(
        Seq2SeqDataset(X_va, Y_va), batch_size=BATCH, shuffle=False, drop_last=False)
    te_ld = torch.utils.data.DataLoader(
        Seq2SeqDataset(X_te, Y_te), batch_size=BATCH, shuffle=False, drop_last=False)

    # ── Model and losses ──────────────────────────────────────────────────────
    model        = MultiResolutionSeq2SeqNet(
                       in_ch=n_feat, hidden=hidden,
                       n_apps=len(APPLIANCES), dt=dt).to(device)
    mse_crit     = nn.MSELoss()
    phys_t_crit  = PhysicsConsistencyLoss(agg_scaler, y_scalers, epsilon_w).to(device)
    phys_f_crit  = FrequencyPhysicsLoss(agg_scaler, y_scalers).to(device)
    event_crit   = EventDetectionLoss(thresholds_scaled, pos_weights).to(device)

    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode='min', factor=0.5, patience=8, min_lr=1e-5)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")
    print(f"Warmup: MSE-only for first {WARMUP_EPOCHS} epochs, "
          f"then add phys_t + phys_f + event losses.")
    print("Starting Multi-Resolution PINN-LNN training ...\n")

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

            ep_tot += loss.item()
            ep_mse += l_mse.item()
            ep_pt  += l_phys_t.item()
            ep_pf  += l_phys_f.item()
            ep_ev  += l_event.item()
            pbar.set_postfix({
                'mse': f'{l_mse.item():.4f}',
                'pt':  f'{l_phys_t.item():.4f}',
                'pf':  f'{l_phys_f.item():.4f}',
                'ev':  f'{l_event.item():.4f}',
            })

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

                vl_mse += l_mse.item()
                vl_pt  += l_phys_t.item()
                vl_pf  += l_phys_f.item()
                vl_tot += (l_mse
                           + lambda_phys_t * l_phys_t
                           + lambda_phys_f * l_phys_f).item()

                for b in range(power_pred.shape[0]):
                    va_preds.append(power_pred[b].cpu().numpy())
                    va_trues.append(yb[b].cpu().numpy())

        nv = len(va_ld)
        avg_va_mse  = vl_mse / nv
        avg_va_pt   = vl_pt  / nv
        avg_va_pf   = vl_pf  / nv
        avg_va_tot  = vl_tot / nv

        history['val_loss'].append(avg_va_tot)
        history['val_mse'].append(avg_va_mse)
        history['val_phys_t'].append(avg_va_pt)
        history['val_phys_f'].append(avg_va_pf)
        sched.step(avg_va_mse)

        pred_trace = reconstruct_trace(va_preds, n_va, STRIDE, WIN)
        true_trace = reconstruct_trace(va_trues, n_va, STRIDE, WIN)
        vm         = per_app_metrics(true_trace, pred_trace, y_scalers)
        history['val_metrics'].append(vm)

        avg_f1  = np.mean([vm[a]['f1']  for a in APPLIANCES])
        avg_mae = np.mean([vm[a]['mae'] for a in APPLIANCES])

        status = "WARMUP" if epoch < WARMUP_EPOCHS else "FULL"
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
    test_metrics  = per_app_metrics(true_trace_te, pred_trace_te, y_scalers)

    print(f"\n{'Appliance':<24} {'F1':>8} {'Precision':>10} "
          f"{'Recall':>8} {'MAE':>8} {'SAE':>8}")
    print("-" * 72)
    for app in APPLIANCES:
        m = test_metrics[app]
        print(f"{app:<24} {m['f1']:>8.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['mae']:>8.2f} {m['sae']:>8.4f}")

    _plot_loss(history, save_dir)
    _plot_metrics(history, test_metrics, save_dir)
    _plot_trace(true_trace_te, pred_trace_te, y_scalers, save_dir)

    cfg = {
        'dataset':  'UKDALE HF 6s',
        'model':    'MultiResolutionSeq2SeqNet',
        'architecture': {
            'dual_branch_cnn': {
                'fast_dilations':  FAST_DILATIONS,
                'fast_rf_steps':   (7-1)*1 + (5-1)*2 + 1,
                'fast_rf_seconds': ((7-1)*1 + (5-1)*2 + 1) * 6,
                'slow_dilations':  SLOW_DILATIONS,
                'slow_rf_steps':   (7-1)*1 + (5-1)*16 + (3-1)*64 + 1,
                'slow_rf_seconds': ((7-1)*1 + (5-1)*16 + (3-1)*64 + 1) * 6,
            },
            'lnn_direction': 'bidirectional',
            'output':        'seq2seq (all WIN timesteps)',
            'event_heads':   'per-appliance binary ON/OFF logits',
        },
        'losses': {
            'mse':    'L2 on scaled power predictions',
            'phys_t': f'time-domain one-sided sum constraint (eps={epsilon_w} W)',
            'phys_f': 'frequency-domain spectral MSE via torch.fft.rfft (log1p)',
            'event':  'per-appliance BCE_with_logits + class-weighted pos_weight',
        },
        'loss_weights': {
            'lambda_phys_t': lambda_phys_t,
            'lambda_phys_f': lambda_phys_f,
            'lambda_event':  lambda_event,
        },
        'pos_weights': {app: float(pw) for app, pw in zip(APPLIANCES, pos_weights)},
        'preprocessing': {
            'denoising': [
                f'Median filter: window={MEDFILT_K} steps ({MEDFILT_K*6}s)',
                f'EMA low-pass: span={EMA_SPAN} steps (alpha={2/(EMA_SPAN+1):.3f})',
            ],
            'normalization': {
                'inputs':  'Z-score (StandardScaler) per channel',
                'targets': 'Min-Max (MinMaxScaler) per appliance',
            },
        },
        'window': {'win': WIN, 'stride_train': STRIDE, 'stride_test': WIN},
        'model_params': {'in_ch': n_feat, 'hidden': hidden,
                         'n_apps': len(APPLIANCES), 'dt': dt},
        'train_params': {'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE,
                         'warmup_epochs': WARMUP_EPOCHS},
        'test_metrics': {
            app: {k: float(v) for k, v in m.items()}
            for app, m in test_metrics.items()
        },
    }
    with open(os.path.join(save_dir, 'multiresolution_results.json'),
              'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=4)
    print(f"\nResults saved to: {save_dir}")
    return test_metrics, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_loss(history: dict, save_dir: str) -> None:
    ep   = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle('Multi-Resolution PINN-LNN -- Training Loss Components', fontsize=11)

    pairs = [
        ('train_loss',    'val_loss',    'Total Loss'),
        ('train_mse',     'val_mse',     'MSE Loss'),
        ('train_phys_t',  'val_phys_t',  'Physics Time'),
        ('train_phys_f',  'val_phys_f',  'Physics Freq'),
        ('train_event',   None,          'Event BCE (train)'),
    ]
    colors = ['steelblue', 'tomato']
    for ax, (tr_k, va_k, title) in zip(axes, pairs):
        ax.plot(ep, history[tr_k], label='Train', color=colors[0])
        if va_k and va_k in history:
            ax.plot(ep, history[va_k], label='Val', color=colors[1])
        ax.set_title(title); ax.set_xlabel('Epoch')
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'multiresolution_loss.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def _plot_metrics(history: dict, test_metrics: dict, save_dir: str) -> None:
    ep   = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(len(APPLIANCES), 2,
                             figsize=(12, 4 * len(APPLIANCES)))
    fig.suptitle('Multi-Resolution PINN-LNN -- Per-Appliance Validation Metrics')

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
    plt.savefig(os.path.join(save_dir, 'multiresolution_per_appliance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def _plot_trace(true_trace: np.ndarray, pred_trace: np.ndarray,
                y_scalers: list, save_dir: str, n_steps: int = 600) -> None:
    fig, axes = plt.subplots(len(APPLIANCES), 1,
                             figsize=(14, 3 * len(APPLIANCES)))
    fig.suptitle('Multi-Resolution PINN-LNN -- Reconstructed Test Trace (1 hour)')
    t = np.arange(n_steps) * 6 / 60   # minutes

    for row, app in enumerate(APPLIANCES):
        i     = APPLIANCES.index(app)
        raw_t = y_scalers[i].inverse_transform(
                    true_trace[:n_steps, i:i+1]).flatten()
        raw_p = y_scalers[i].inverse_transform(
                    pred_trace[:n_steps, i:i+1]).flatten()
        axes[row].plot(t, raw_t, label='Ground truth', color='steelblue',
                       linewidth=1.0, alpha=0.8)
        axes[row].plot(t, raw_p, label='Prediction', color='tomato',
                       linewidth=1.0, alpha=0.8)
        axes[row].set_title(app)
        axes[row].set_ylabel('Power (W)')
        axes[row].legend(loc='upper right')
        axes[row].grid(alpha=0.3)

    axes[-1].set_xlabel('Time (min)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'multiresolution_trace.png'),
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
    save_dir  = f'models/multiresolution_pinn_lnn_{timestamp}'

    train(
        save_dir      = save_dir,
        hidden        = 64,
        dt            = 0.1,
        lambda_phys_t = LAMBDA_PHYS_T,
        lambda_phys_f = LAMBDA_PHYS_F,
        lambda_event  = LAMBDA_EVENT,
        epsilon_w     = EPSILON_W,
    )
