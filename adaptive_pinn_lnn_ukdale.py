"""
Adaptive-dt Spectral Phase-Portrait PINN-LNN for UKDALE NILM
=============================================================

Implements three LNN-specific HF transformations adapted for the UKDALE
6-second active-power dataset.  Note: UKDALE does not record raw voltage or
current waveforms, so each idea is realised using the available power signal.

─────────────────────────────────────────────────────────────────────────────
A.  Phase Portrait  (analog of V-I Trajectory)
─────────────────────────────────────────────────────────────────────────────
In HF NILM, the V-I trajectory maps (V_t, I_t) pairs into a 2D loop whose
shape fingerprints the appliance.  With only active power P(t) at 6s, the
equivalent is the POWER PHASE PORTRAIT — the trajectory in (P, dP/dt) space.

Features derived from the phase portrait:
  • dP²/dt²     — 2nd derivative (acceleration): is the velocity itself
                  changing? Zero in steady state, large during ramps.
  • Loop area   — Shoelace formula over the last LOOP_K points in (P, dP/dt)
                  space.  Near-zero in steady state (point cluster), large
                  during transients (wide loop).  Directly captures the
                  "shape" of the phase-space trajectory the LNN must track.

─────────────────────────────────────────────────────────────────────────────
B.  Log-Spectral Power Features  (analog of MFCCs)
─────────────────────────────────────────────────────────────────────────────
MFCCs compress an audio signal by FFT → Mel-scale → DCT.  Applied to the
6s power signal, the Mel scale is not meaningful (our Nyquist is 0.083 Hz),
so we use CAUSAL LOG-SPECTRAL BINS instead:
  1. Sliding Hanning-windowed FFT over the last SPEC_WIN steps (~3 min).
  2. Log-magnitude of N_SPEC_BINS frequency bins (skip DC).
  3. These bins encode whether the aggregate power is oscillating at
     fridge-cycle frequencies, washing-machine-phase frequencies, etc.
  4. Because the window is causal, no future leakage — safe for seq2seq.

─────────────────────────────────────────────────────────────────────────────
C.  Adaptive dt  (LNN-Specific Resampling)
─────────────────────────────────────────────────────────────────────────────
LNNs are continuous-time ODE systems: dh/dt = f(h, x, t).  The discrete
step dt controls integration resolution.  Key insight:

  • During a transient (appliance switching): power changes rapidly →
    small dt for accurate ODE integration of the state transition.
  • During steady state: power is stable → large dt to skip quickly,
    reducing compute and forcing the LNN to learn "nothing interesting here".

Implementation:
  dt(t) = dt_max − tanh(|dP/dt(t)| × sensitivity) × (dt_max − dt_min)

  |dP/dt| ≈ 0  (steady)  →  score ≈ 0  →  dt ≈ dt_max   (coarse)
  |dP/dt| >> 0 (transient) →  score ≈ 1  →  dt ≈ dt_min   (fine)

The adaptive dt tensor (batch, WIN) is passed directly into the LNN cells
at each timestep, replacing the fixed scalar dt used in earlier scripts.

─────────────────────────────────────────────────────────────────────────────
Architecture
─────────────────────────────────────────────────────────────────────────────
    Input  (batch, WIN, N_FEAT=17)    AdaptiveDt (batch, WIN)
         |                                   |
    LengthPreservingCNN                      |
      (batch, WIN, hidden)                   |
         |                                   |
    Bidirectional Adaptive LNN ←─────────────┘
      fwd cell: dt(t) per step (left→right)
      bwd cell: dt(t) per step (right→left)
         |
    LayerNorm(hidden * 2)  at each timestep
         |
    4 × Linear(hidden*2, 1)  — per timestep
         |
    Output (batch, WIN, 4)

Input channels (N_FEAT = 17):
    0   aggregate            raw mains power (W)
    1   dP/dt  1-step        switching transients
    2   dP/dt 10-step        phase transitions
    3   d²P/dt²              acceleration (velocity of velocity)
    4   loop area            Shoelace area in (P, dP) phase space
    5   rolling mean 10      60 s local baseline
    6   rolling std  10      60 s local variability
    7   rolling mean 50      5 min trend
    8   rolling std  50      5 min variability
    9–16  log-spectral bins  8 causal FFT log-magnitude bins

Physics loss — applied at every timestep (same as hf_seq2seq_pinn_lnn_ukdale.py).
Output      — Seq2Seq (batch, WIN, 4), full-day trace reconstructed for eval.
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
WIN           = 299
STRIDE        = 10
STRIDE_TEST   = WIN     # non-overlapping for test reconstruction

ROLL_S        = 10      # short rolling window: 10 steps = 60 s
ROLL_L        = 50      # long  rolling window: 50 steps =  5 min
LOOP_K        = 10      # phase-portrait Shoelace window (10 steps = 60 s)
SPEC_WIN      = 30      # FFT window: 30 steps = 3 min
N_SPEC_BINS   = 8       # log-spectral bins to keep (skip DC)

DT_MIN        = 0.02    # minimum dt (used during transients)
DT_MAX        = 0.40    # maximum dt (used during steady state)
DT_SENS       = 0.008   # sensitivity: tanh(|dP| × sens); calibrated to
                        # UKDALE typical dP range (1–200 W/step)

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
# Feature engineering
# ---------------------------------------------------------------------------

def _n_step_diff(arr: np.ndarray, n: int) -> np.ndarray:
    """arr[t] − arr[t-n], zero-padded at the head."""
    d     = np.zeros_like(arr)
    d[n:] = arr[n:] - arr[:-n]
    return d


def _phase_portrait_loop_area(P: np.ndarray,
                               dP: np.ndarray,
                               k: int = LOOP_K) -> np.ndarray:
    """
    Shoelace formula applied to the last k points of the (P, dP/dt) trajectory.

    Geometrically: a cluster of near-identical points (steady state) has area ≈ 0.
    A wide looping path (transient) has area >> 0.
    The result is normalised by k² to be scale-invariant.
    """
    N    = len(P)
    area = np.zeros(N, dtype=np.float32)
    for t in range(k, N):
        x  = P[t - k : t + 1]   # (k+1,)
        y  = dP[t - k : t + 1]
        # Shoelace
        a  = np.abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1])) * 0.5
        area[t] = a / (k * k + 1e-8)   # normalise
    area[:k] = area[k]                  # fill head
    return area


def _causal_log_spectral(signal: np.ndarray,
                          win: int = SPEC_WIN,
                          n_bins: int = N_SPEC_BINS) -> np.ndarray:
    """
    At each timestep t, compute a Hanning-windowed FFT over the causal window
    [t-win+1 … t] and return log-magnitude of the first n_bins non-DC bins.

    This is the power-signal analog of MFCCs:
      FFT  →  log-magnitude  →  n_bins frequency descriptors

    Causal design ensures no future leakage into the input features.
    Frequencies range from 1/T_win to Nyquist (1/12s ≈ 0.083 Hz), capturing
    oscillations from ~3 min (fridge cycle edge) down to 12 s (fast bursts).
    """
    N    = len(signal)
    out  = np.zeros((N, n_bins), dtype=np.float32)
    hann = np.hanning(win).astype(np.float32)

    for t in range(N):
        start   = max(0, t - win + 1)
        seg_raw = signal[start : t + 1].astype(np.float32)
        # Left-pad with edge value if window not yet full
        seg = np.full(win, seg_raw[0], dtype=np.float32)
        seg[win - len(seg_raw):] = seg_raw
        fft_mag = np.abs(np.fft.rfft(seg * hann))          # (win//2 + 1,)
        out[t]  = np.log1p(fft_mag[1 : n_bins + 1])        # skip DC bin 0
    return out


def _adaptive_dt(d_agg: np.ndarray,
                 dt_min: float = DT_MIN,
                 dt_max: float = DT_MAX,
                 sensitivity: float = DT_SENS) -> np.ndarray:
    """
    Per-timestep integration step for the LNN ODE.

    Large |dP/dt| (transient) → score → 1 → dt → dt_min  (fine integration)
    Small |dP/dt| (steady)    → score → 0 → dt → dt_max  (coarse / skip)

    dt(t) = dt_max − tanh(|dP/dt(t)| × sensitivity) × (dt_max − dt_min)
    """
    score = np.tanh(np.abs(d_agg) * sensitivity).astype(np.float32)
    return (dt_max - score * (dt_max - dt_min)).astype(np.float32)


def compute_all_features(df: pd.DataFrame):
    """
    Returns:
        feat:    (N, 17)  input feature matrix
        adt:     (N,)     adaptive dt signal
    """
    agg  = df['aggregate'].values.astype(np.float32)
    d1   = _n_step_diff(agg,  1)
    d10  = _n_step_diff(agg, 10)
    d2   = _n_step_diff(d1,   1)    # d²P/dt² ≈ second difference of agg

    loop = _phase_portrait_loop_area(agg, d1, LOOP_K)
    spec = _causal_log_spectral(agg, SPEC_WIN, N_SPEC_BINS)  # (N, 8)

    s      = pd.Series(agg)
    rm_s   = s.rolling(ROLL_S, min_periods=1).mean().values.astype(np.float32)
    rs_s   = s.rolling(ROLL_S, min_periods=1).std().fillna(0).values.astype(np.float32)
    rm_l   = s.rolling(ROLL_L, min_periods=1).mean().values.astype(np.float32)
    rs_l   = s.rolling(ROLL_L, min_periods=1).std().fillna(0).values.astype(np.float32)

    # Stack into (N, 9) dense features
    dense = np.stack([agg, d1, d10, d2, loop, rm_s, rs_s, rm_l, rs_l], axis=1)
    feat  = np.concatenate([dense, spec], axis=1)   # (N, 9+8=17)

    adt = _adaptive_dt(d1)   # (N,)
    return feat.astype(np.float32), adt


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(split: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f'UKDALE_HF_{split}.csv')
    df   = pd.read_csv(path, index_col='timestamp', parse_dates=True)
    print(f"  {split:12s}: {len(df):6d} rows")
    return df


def create_sequences(feat: np.ndarray, targets: np.ndarray,
                     adt: np.ndarray, stride: int):
    """
    Seq2Seq sliding-window sequences.

    Returns:
        X:  (M, WIN, N_FEAT)
        Y:  (M, WIN, n_apps)
        DT: (M, WIN)           adaptive dt per timestep
    """
    X, Y, DT = [], [], []
    for i in range(0, len(feat) - WIN, stride):
        X.append(feat[i : i + WIN])
        Y.append(targets[i : i + WIN])
        DT.append(adt[i : i + WIN])
    return (np.array(X, dtype=np.float32),
            np.array(Y, dtype=np.float32),
            np.array(DT, dtype=np.float32))


# ---------------------------------------------------------------------------
# Physics Consistency Loss — every timestep
# ---------------------------------------------------------------------------

class PhysicsConsistencyLoss(nn.Module):
    """
    L_phys = mean_{batch, t}( ReLU( Σ_i p̂_i_raw(t) − P_agg_raw(t) − ε ) )

    Applied at every timestep in the window, giving WIN × more constraints
    than the midpoint-only version.
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
        x_scaled:    (batch, WIN, N_FEAT) — channel 0 is scaled aggregate
        pred_scaled: (batch, WIN, n_apps)
        """
        x_raw = x_scaled[:, :, 0] * self.x_range + self.x_min
        p_raw = pred_scaled * self.y_ranges + self.y_mins
        return F.relu(p_raw.sum(dim=-1) - x_raw - self.epsilon).mean()


# ---------------------------------------------------------------------------
# Length-preserving dilated CNN
# ---------------------------------------------------------------------------

class LengthPreservingCNN(nn.Module):
    """
    Four dilated Conv1d layers, no pooling.
    Output length == input length; receptive field = 183 steps (~18 min).

    Dilation: 1 → 4 → 16 → 64
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
# Adaptive Liquid Cell  (variable dt per timestep)
# ---------------------------------------------------------------------------

class AdaptiveLiquidCell(nn.Module):
    """
    AdvancedLiquidTimeLayer with a per-sample, per-timestep dt.

    The ODE update:
        dh = (−h/τ + gate ⊙ tanh(Wx + Uh)) × dt(t)

    When dt(t) is small (transient detected), the cell takes a fine
    integration step — tracking the fast state change accurately.
    When dt(t) is large (steady state), the cell jumps further along the
    ODE trajectory — efficiently skipping stable regions.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_proj  = nn.Linear(input_size, hidden_size)
        self.tau_base    = nn.Parameter(torch.ones(hidden_size))
        self.tau_mod     = nn.Linear(input_size, hidden_size)
        self.rec_weights = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.rec_weights)
        self.gate        = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x_t: torch.Tensor,
                h: torch.Tensor,
                dt_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t:  (batch, input_size)
            h:    (batch, hidden_size)
            dt_t: (batch,)            per-sample adaptive time step
        """
        ip  = self.input_proj(x_t)
        rp  = h @ self.rec_weights
        tb  = F.softplus(self.tau_base).unsqueeze(0)          # (1, hidden)
        tm  = torch.sigmoid(self.tau_mod(x_t))
        tau = (tb * tm).clamp(min=DT_MIN)
        g   = torch.sigmoid(self.gate(torch.cat([x_t, h], dim=1)))
        dh  = ((-h / tau) + g * torch.tanh(ip + rp)) * dt_t.unsqueeze(1)
        return (h + dh).clamp(-10.0, 10.0)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class AdaptiveSeq2SeqLiquidNet(nn.Module):
    """
    LengthPreservingCNN  +  Bidirectional Adaptive LNN  +  per-timestep heads.

    Both the forward and backward LNN cells receive the adaptive dt signal,
    so coarse/fine integration is applied in both temporal directions.
    """

    def __init__(self, in_ch: int, hidden: int, n_apps: int):
        super().__init__()
        self.hidden = hidden
        self.n_apps = n_apps

        self.cnn      = LengthPreservingCNN(in_ch, hidden)
        self.fwd_cell = AdaptiveLiquidCell(hidden, hidden)
        self.bwd_cell = AdaptiveLiquidCell(hidden, hidden)

        self.norm  = nn.LayerNorm(hidden * 2)
        self.heads = nn.ModuleList([
            nn.Linear(hidden * 2, 1) for _ in range(n_apps)
        ])

    def forward(self, x: torch.Tensor,
                dt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:  (batch, WIN, in_ch)
            dt: (batch, WIN)         adaptive dt — large in steady state,
                                     small during transients
        Returns:
            (batch, WIN, n_apps)
        """
        feat  = self.cnn(x)
        batch = feat.shape[0]
        T     = feat.shape[1]

        # Forward pass (causal: left → right)
        h_f = torch.zeros(batch, self.hidden, device=x.device)
        fwd = []
        for t in range(T):
            h_f = self.fwd_cell(feat[:, t, :], h_f, dt[:, t])
            fwd.append(h_f)

        # Backward pass (anti-causal: right → left)
        h_b = torch.zeros(batch, self.hidden, device=x.device)
        bwd = [None] * T
        for t in reversed(range(T)):
            h_b    = self.bwd_cell(feat[:, t, :], h_b, dt[:, t])
            bwd[t] = h_b

        # Concatenate, normalise, apply per-timestep heads
        out = []
        for t in range(T):
            h_t = self.norm(torch.cat([fwd[t], bwd[t]], dim=1))
            out.append(torch.cat([head(h_t) for head in self.heads], dim=1))
        return torch.stack(out, dim=1)     # (batch, WIN, n_apps)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AdaptiveDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, DT):
        self.X  = torch.FloatTensor(X)
        self.Y  = torch.FloatTensor(Y)
        self.DT = torch.FloatTensor(DT)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.DT[i]


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
        out[app] = calculate_nilm_metrics(raw_t, raw_p,
                                          threshold=THRESHOLDS[app])
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(save_dir: str,
          hidden:      int   = 64,
          lambda_phys: float = LAMBDA_PHYS,
          epsilon_w:   float = EPSILON_W) -> tuple:

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}  |  WIN={WIN}  hidden={hidden}  "
          f"dt_range=[{DT_MIN}, {DT_MAX}]  sens={DT_SENS}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading and computing HF features...")
    df_tr = load_csv('train')
    df_va = load_csv('validation')
    df_te = load_csv('test')
    n_tr, n_va, n_te = len(df_tr), len(df_va), len(df_te)

    feat_tr, adt_tr = compute_all_features(df_tr)
    feat_va, adt_va = compute_all_features(df_va)
    feat_te, adt_te = compute_all_features(df_te)

    n_feat = feat_tr.shape[1]    # 17
    print(f"  Feature channels: {n_feat}  "
          f"(9 dense + {N_SPEC_BINS} spectral)")
    print(f"  Adaptive dt range observed in train: "
          f"[{adt_tr.min():.4f}, {adt_tr.max():.4f}]")
    print(f"  Transient fraction (dt < midpoint): "
          f"{(adt_tr < (DT_MIN + DT_MAX) / 2).mean()*100:.1f}%")

    tgt_tr = df_tr[APPLIANCES].values.astype(np.float32)
    tgt_va = df_va[APPLIANCES].values.astype(np.float32)
    tgt_te = df_te[APPLIANCES].values.astype(np.float32)

    # ── Sequences ─────────────────────────────────────────────────────────────
    X_tr, Y_tr, DT_tr = create_sequences(feat_tr, tgt_tr, adt_tr, STRIDE)
    X_va, Y_va, DT_va = create_sequences(feat_va, tgt_va, adt_va, STRIDE)
    X_te, Y_te, DT_te = create_sequences(feat_te, tgt_te, adt_te, WIN)
    print(f"\nTrain : {X_tr.shape} -> {Y_tr.shape}  "
          f"({X_tr.shape[0] * WIN:,} predictions)")
    print(f"Val   : {X_va.shape} -> {Y_va.shape}")
    print(f"Test  : {X_te.shape} -> {Y_te.shape}  [non-overlapping]")

    # ── Scale features (each channel independently) ───────────────────────────
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

    # Scale DT to [0,1] so it enters the LNN as a normalised weight
    dt_scaler = MinMaxScaler()
    DT_tr = dt_scaler.fit_transform(DT_tr.reshape(-1, 1)).reshape(-1, WIN)
    DT_va = dt_scaler.transform(DT_va.reshape(-1, 1)).reshape(-1, WIN)
    DT_te = dt_scaler.transform(DT_te.reshape(-1, 1)).reshape(-1, WIN)
    # Rescale back to physical dt range so the ODE is meaningful
    DT_tr = DT_tr * (DT_MAX - DT_MIN) + DT_MIN
    DT_va = DT_va * (DT_MAX - DT_MIN) + DT_MIN
    DT_te = DT_te * (DT_MAX - DT_MIN) + DT_MIN

    # ── Scale targets ─────────────────────────────────────────────────────────
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
        AdaptiveDataset(X_tr, Y_tr, DT_tr),
        batch_size=BATCH, shuffle=True, drop_last=False)
    va_ld = torch.utils.data.DataLoader(
        AdaptiveDataset(X_va, Y_va, DT_va),
        batch_size=BATCH, shuffle=False, drop_last=False)
    te_ld = torch.utils.data.DataLoader(
        AdaptiveDataset(X_te, Y_te, DT_te),
        batch_size=BATCH, shuffle=False, drop_last=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = AdaptiveSeq2SeqLiquidNet(
                    in_ch=n_feat, hidden=hidden,
                    n_apps=len(APPLIANCES)).to(device)
    mse_crit  = nn.MSELoss()
    phys_crit = PhysicsConsistencyLoss(
                    agg_scaler, y_scalers, epsilon_w).to(device)
    opt       = torch.optim.Adam(model.parameters(), lr=LR)
    sched     = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode='min', factor=0.5, patience=8, min_lr=1e-5)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")
    print("Starting Adaptive-dt Seq2Seq PINN-LNN training...")

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

        for xb, yb, dtb in pbar:
            xb, yb, dtb = xb.to(device), yb.to(device), dtb.to(device)
            opt.zero_grad()
            pred = model(xb, dtb)              # (B, WIN, n_apps)

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
            for xb, yb, dtb in va_ld:
                xb, yb, dtb = xb.to(device), yb.to(device), dtb.to(device)
                pred   = model(xb, dtb)
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
        for xb, yb, dtb in te_ld:
            pred = model(xb.to(device), dtb.to(device))
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

    _plot_loss(history, save_dir)
    _plot_metrics(history, test_metrics, save_dir)
    _plot_trace(true_trace_te, pred_trace_te, y_scalers, save_dir)
    _plot_adaptive_dt(adt_tr[:1500], save_dir)

    cfg = {
        'dataset':  'UKDALE HF 6s',
        'model':    'AdaptiveSeq2SeqLiquidNet',
        'features': {
            'dense':   ['aggregate', 'dP_1step', 'dP_10step',
                        'd2P', 'phase_loop_area',
                        'roll_mean_10', 'roll_std_10',
                        'roll_mean_50', 'roll_std_50'],
            'spectral': f'{N_SPEC_BINS} causal log-FFT bins (win={SPEC_WIN} steps)',
            'total_channels': n_feat,
        },
        'adaptive_dt': {
            'dt_min': DT_MIN, 'dt_max': DT_MAX, 'sensitivity': DT_SENS,
            'interpretation': 'small dt during transients, large dt during steady state',
        },
        'architecture': {
            'cnn_dilations': [1, 4, 16, 64],
            'lnn_direction': 'bidirectional',
            'output': 'seq2seq (all WIN timesteps)',
            'physics_constraint': 'every timestep',
        },
        'window': {'win': WIN, 'stride_train': STRIDE, 'stride_test': WIN},
        'model_params': {'in_ch': n_feat, 'hidden': hidden,
                         'n_apps': len(APPLIANCES)},
        'train_params': {'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE,
                         'lambda_phys': lambda_phys, 'epsilon_w': epsilon_w,
                         'warmup_epochs': WARMUP_EPOCHS},
        'test_metrics': {
            app: {k: float(v) for k, v in m.items()}
            for app, m in test_metrics.items()
        },
    }
    with open(os.path.join(save_dir, 'adaptive_results.json'),
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
    plt.savefig(os.path.join(save_dir, 'adaptive_loss.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def _plot_metrics(history: dict, test_metrics: dict, save_dir: str) -> None:
    ep  = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(len(APPLIANCES), 2,
                             figsize=(12, 4 * len(APPLIANCES)))
    fig.suptitle('Adaptive-dt PINN-LNN — Per-Appliance Validation Metrics')
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
    plt.savefig(os.path.join(save_dir, 'adaptive_per_appliance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def _plot_trace(true_trace, pred_trace, y_scalers, save_dir,
                n_steps: int = 600) -> None:
    fig, axes = plt.subplots(len(APPLIANCES), 1,
                             figsize=(14, 3 * len(APPLIANCES)))
    fig.suptitle('Adaptive-dt — Reconstructed Test Trace (first 1 hour)')
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
    plt.savefig(os.path.join(save_dir, 'adaptive_trace.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def _plot_adaptive_dt(adt_sample: np.ndarray, save_dir: str) -> None:
    """Visualise the adaptive dt signal over a 2.5-hour window of training data."""
    t = np.arange(len(adt_sample)) * 6 / 60
    plt.figure(figsize=(14, 3))
    plt.plot(t, adt_sample, color='steelblue', linewidth=0.8)
    plt.axhline(DT_MAX, color='gray',   linestyle='--', linewidth=0.8,
                label=f'dt_max={DT_MAX} (steady state)')
    plt.axhline(DT_MIN, color='tomato', linestyle='--', linewidth=0.8,
                label=f'dt_min={DT_MIN} (transient)')
    plt.fill_between(t, adt_sample, DT_MAX, alpha=0.15, color='tomato',
                     label='Integration effort (inverse of dt)')
    plt.title('Adaptive dt — LNN spends more time on transient regions')
    plt.xlabel('Time (min)'); plt.ylabel('dt value')
    plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'adaptive_dt_signal.png'),
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
    save_dir  = f'models/adaptive_pinn_lnn_{timestamp}'

    train(
        save_dir     = save_dir,
        hidden       = 64,
        lambda_phys  = LAMBDA_PHYS,
        epsilon_w    = EPSILON_W,
    )
