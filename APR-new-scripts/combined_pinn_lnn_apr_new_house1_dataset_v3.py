"""
combined_pinn_lnn_apr_new_house1_dataset_v3.py
===============================================
v3 of combined_pinn_lnn_apr_new_house1_dataset_v2.py -- two targeted fixes
aimed specifically at dishwasher precision, which v2 only raised to
P=0.58 (test) despite the same architecture reaching P=0.81 on House 2.

Context: v2's FIX 1-3 (train-locked thresholds, min-F1 checkpoint selection,
lambda ramp) fixed the training instability that caused wild run-to-run
precision/recall swings in v1 (P=0.73/R=0.30 in one run, P=0.56/R=0.51 in
another, same script, same day). But on House 1, dishwasher precision is
still capped well below what the same model achieves on House 2 -- pointing
at flicker/boundary noise in the prediction, not the loss schedule.

FIX 4 -- Minimum-run-length post-filter (denoising)
    The gate can flip on/off for a handful of samples at a time -- a few
    seconds of spurious "ON" that inflates FP, or a few seconds of dropout
    inside a real cycle that inflates FN. Both hurt precision and recall,
    but isolated blips are disproportionately false positives on a
    low-duty-cycle appliance like dishwasher (2.9% ON in training).
    Fix: after reconstructing the val/test prediction trace, run a
    morphological open+close pass per appliance:
      1. Close (bridge) OFF gaps shorter than MIN_GAP_MIN minutes -- treats
         a brief mid-cycle dropout as still-ON, filled by interpolating the
         surrounding predicted power.
      2. Open (remove) ON runs shorter than MIN_ON_MIN minutes -- zeroes out
         isolated flicker that doesn't look like a real appliance cycle.
    Only dishwasher has non-zero MIN_ON_MIN/MIN_GAP_MIN by default (see
    DENOISE_PARAMS below) so the other three appliances -- already close to
    their ceiling -- are left untouched. Both raw (pre-filter) and filtered
    test metrics are printed so the effect is visible, not just assumed.

FIX 5 -- Per-appliance pos_weight ceiling override
    The global POS_WEIGHT_CLAMP=(1.0, 50.0) applies to all four appliances.
    Dishwasher's actual pos_weight (34.0 in the last House-1 v2 run) sits
    comfortably under that ceiling, meaning the gate is already being pushed
    fairly hard to fire on the rare positive class. Lowering dishwasher's
    ceiling specifically is a cheap, reversible experiment to check whether
    that push is trading away precision faster than it buys recall.
    Fix: POS_WEIGHT_CLAMP_OVERRIDE lets any appliance use a tighter clamp;
    dishwasher defaults to (1.0, 20.0) here. Other appliances fall back to
    the global POS_WEIGHT_CLAMP unchanged.

Everything else (architecture, features, scalers, sequence construction,
FIX 1-3 from v2) is unchanged.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Source Code'))
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

MEDFILT_K     = 5
EMA_SPAN      = 10
ROLL_K        = 10

LAMBDA_PHYS   = 0.01
LAMBDA_EVENT  = 0.05
EPSILON_W     = 50.0
WARMUP_EPOCHS = 20
RAMP_EPOCHS   = 10

THRESHOLD_DELTA   = 20.0
THRESHOLD_LOW_PCT = 0.05
THRESHOLD_MIN     = 10.0
CYCLING_P5_W      = 80.0
POS_WEIGHT_CLAMP  = (1.0, 50.0)

# FIX 5: per-appliance override of the pos_weight ceiling. Appliances not
# listed here fall back to POS_WEIGHT_CLAMP unchanged.
POS_WEIGHT_CLAMP_OVERRIDE = {
    'dishwasher': (1.0, 20.0),
}

# FIX 4: per-appliance minimum ON-run / OFF-gap length (minutes) for the
# post-hoc denoising filter. 0.0 disables filtering for that appliance
# (fridge/microwave/washing_machine are left untouched by default).
STEP_SECONDS   = 6.0
DENOISE_PARAMS = {
    'dishwasher':      {'min_on_min': 1.0, 'min_gap_min': 0.5},
    'fridge':          {'min_on_min': 0.0, 'min_gap_min': 0.0},
    'microwave':       {'min_on_min': 0.0, 'min_gap_min': 0.0},
    'washing_machine': {'min_on_min': 0.0, 'min_gap_min': 0.0},
}

APPLIANCES  = ['dishwasher', 'fridge', 'microwave', 'washing_machine']
AGG_COL     = 'aggregate'

DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'APR-new-House1-dataset')


# ---------------------------------------------------------------------------
# Adaptive thresholds
# ---------------------------------------------------------------------------

def compute_adaptive_thresholds(df: pd.DataFrame) -> dict:
    thresholds = {}
    for app in APPLIANCES:
        col     = df[app]
        nonzero = col[col > 0]
        if len(nonzero) == 0:
            thresholds[app] = THRESHOLD_MIN
            continue
        p5 = float(nonzero.quantile(THRESHOLD_LOW_PCT))
        thresholds[app] = THRESHOLD_MIN if p5 > CYCLING_P5_W else max(p5 + THRESHOLD_DELTA, THRESHOLD_MIN)
    return thresholds


def compute_event_thresholds(df: pd.DataFrame) -> dict:
    thresholds = {}
    for app in APPLIANCES:
        col     = df[app]
        nonzero = col[col > 0]
        if len(nonzero) == 0:
            thresholds[app] = THRESHOLD_MIN
        else:
            p5 = float(nonzero.quantile(THRESHOLD_LOW_PCT))
            thresholds[app] = max(p5 + THRESHOLD_DELTA, THRESHOLD_MIN)
    return thresholds


# ---------------------------------------------------------------------------
# 8-channel feature extraction
# ---------------------------------------------------------------------------

def _median_filter(arr, k):
    return pd.Series(arr).rolling(k, center=True, min_periods=1).median().values.astype(np.float32)

def _ema_filter(arr, span):
    return pd.Series(arr).ewm(span=span, adjust=False).mean().values.astype(np.float32)

def _n_step_diff(arr, n):
    d = np.zeros_like(arr); d[n:] = arr[n:] - arr[:-n]; return d

def compute_features(mains: np.ndarray) -> np.ndarray:
    raw    = mains.astype(np.float32)
    med    = _median_filter(raw, MEDFILT_K)
    smooth = _ema_filter(med, EMA_SPAN)
    resid  = (raw - smooth).astype(np.float32)
    d_raw_1    = _n_step_diff(raw,    1)
    d_smooth_1 = _n_step_diff(smooth, 1)
    s  = pd.Series(smooth)
    rm = s.rolling(ROLL_K, min_periods=1).mean().values.astype(np.float32)
    rs = s.rolling(ROLL_K, min_periods=1).std().fillna(0).values.astype(np.float32)
    return np.stack([raw, med, smooth, resid, d_raw_1, d_smooth_1, rm, rs], axis=1)


# ---------------------------------------------------------------------------
# Sequence creation  (Seq2Seq)
# ---------------------------------------------------------------------------

def create_sequences(df: pd.DataFrame, stride: int):
    feat     = compute_features(df[AGG_COL].values)
    app_arrs = {app: df[app].values.astype(np.float32) for app in APPLIANCES}
    N = len(feat)
    X, Y = [], []
    for i in range(0, N - WIN, stride):
        X.append(feat[i:i + WIN])
        Y.append(np.stack([app_arrs[app][i:i + WIN] for app in APPLIANCES], axis=1))
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(dataset_dir: str) -> dict:
    print(f"Loading APR-new-House1-dataset CSV data from '{dataset_dir}' ...")
    file_map = {'train': 'UKDALE_HF_train.csv',
                'val':   'UKDALE_HF_validation.csv',
                'test':  'UKDALE_HF_test.csv'}
    splits = {}
    for name, fname in file_map.items():
        path = os.path.join(dataset_dir, fname)
        splits[name] = pd.read_csv(path, index_col='timestamp', parse_dates=True)
        df = splits[name]
        print(f"  {name:6s}: {len(df):>7,} rows  "
              f"{df.index.min().date()} -> {df.index.max().date()}")
    print(f"  Columns: {list(splits['train'].columns)}")
    return splits


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class LengthPreservingCNN(nn.Module):
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
    def forward(self, x):
        return self.net(x.permute(0, 2, 1)).permute(0, 2, 1)


class LiquidCell(nn.Module):
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
# Combined model
# ---------------------------------------------------------------------------

class CombinedPINNAdvancedLNN(nn.Module):
    """
    Dual-head model: gated power estimation.

    gate_heads  produce detection confidence g in [0,1] at every timestep.
    power_heads produce raw power estimate   p in [0,1] at every timestep.
    Output:  gated_power = g x p  (batch, WIN, n_apps)
    """

    def __init__(self, in_ch: int, hidden: int, n_apps: int, dt: float = 0.1):
        super().__init__()
        self.hidden = hidden
        self.n_apps = n_apps

        self.cnn      = LengthPreservingCNN(in_ch, hidden)
        self.fwd_cell = LiquidCell(hidden, hidden, dt)
        self.bwd_cell = LiquidCell(hidden, hidden, dt)
        self.norm     = nn.LayerNorm(hidden * 2)

        self.gate_heads  = nn.ModuleList([nn.Linear(hidden * 2, 1) for _ in range(n_apps)])
        self.power_heads = nn.ModuleList([nn.Linear(hidden * 2, 1) for _ in range(n_apps)])

    def forward(self, x: torch.Tensor):
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

        gated_power_list, gate_logit_list = [], []
        for t in range(T):
            h_t = self.norm(torch.cat([fwd[t], bwd[t]], dim=1))
            g_logits = torch.cat([head(h_t) for head in self.gate_heads], dim=1)
            g        = torch.sigmoid(g_logits)
            p = torch.cat([torch.sigmoid(head(h_t)) for head in self.power_heads], dim=1)
            gated_power_list.append(g * p)
            gate_logit_list.append(g_logits)

        return (torch.stack(gated_power_list, dim=1),
                torch.stack(gate_logit_list,  dim=1))


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class PhysicsConsistencyLoss(nn.Module):
    def __init__(self, agg_scaler: StandardScaler,
                 y_scalers: list, epsilon_w: float = EPSILON_W):
        super().__init__()
        self.epsilon = epsilon_w
        self.register_buffer('x_mean',  torch.tensor(float(agg_scaler.mean_[0]),  dtype=torch.float32))
        self.register_buffer('x_scale', torch.tensor(float(agg_scaler.scale_[0]), dtype=torch.float32))
        self.register_buffer('y_mins',   torch.tensor([float(s.data_min_[0])   for s in y_scalers], dtype=torch.float32))
        self.register_buffer('y_ranges', torch.tensor([float(s.data_range_[0]) for s in y_scalers], dtype=torch.float32))

    def forward(self, x_z: torch.Tensor, gated_power: torch.Tensor) -> torch.Tensor:
        x_raw = x_z[:, :, 0] * self.x_scale + self.x_mean
        p_raw = gated_power * self.y_ranges + self.y_mins
        return F.relu(p_raw.sum(dim=-1) - x_raw - self.epsilon).mean()


class GatedEventLoss(nn.Module):
    def __init__(self, thresholds_scaled: list, pos_weights: list):
        super().__init__()
        self.n_apps = len(thresholds_scaled)
        self.register_buffer('thresholds', torch.tensor(thresholds_scaled, dtype=torch.float32))
        self.register_buffer('pos_weights', torch.tensor(pos_weights,       dtype=torch.float32))

    def forward(self, gate_logits: torch.Tensor, y_scaled: torch.Tensor) -> torch.Tensor:
        total = torch.zeros(1, device=gate_logits.device)
        for i in range(self.n_apps):
            y_on = (y_scaled[:, :, i] > self.thresholds[i]).float()
            total = total + F.binary_cross_entropy_with_logits(
                gate_logits[:, :, i], y_on, pos_weight=self.pos_weights[i:i+1])
        return total / self.n_apps


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Seq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
    def __len__(self):          return len(self.X)
    def __getitem__(self, i):   return self.X[i], self.Y[i]


# ---------------------------------------------------------------------------
# Overlap-average reconstruction
# ---------------------------------------------------------------------------

def reconstruct_trace(window_preds: list, n_total: int,
                      stride: int, win: int) -> np.ndarray:
    n_apps = window_preds[0].shape[1]
    acc    = np.zeros((n_total, n_apps), dtype=np.float64)
    count  = np.zeros((n_total, 1),     dtype=np.float64)
    for idx, pw in enumerate(window_preds):
        s, e = idx * stride, idx * stride + win
        if e > n_total: break
        acc[s:e] += pw; count[s:e] += 1
    return (acc / np.maximum(count, 1)).astype(np.float32)


# ---------------------------------------------------------------------------
# FIX 4: minimum-run-length denoising filter
# ---------------------------------------------------------------------------

def _rle_runs(state: np.ndarray):
    """Start/end indices (end exclusive) of contiguous True runs in a 1D bool array."""
    padded = np.concatenate(([0], state.astype(np.int8), [0]))
    diffs  = np.diff(padded)
    starts = np.where(diffs ==  1)[0]
    ends   = np.where(diffs == -1)[0]
    return starts, ends


def denoise_power_trace(power: np.ndarray, threshold: float,
                        min_on_steps: int, min_gap_steps: int) -> np.ndarray:
    """
    Morphological close (bridge short OFF gaps) then open (drop short ON
    runs) on a single appliance's reconstructed power trace, evaluated at
    `threshold` watts. No-op when both step counts are 0.

    Bridged gaps are filled by linear interpolation between the real
    predicted-power samples on either side; dropped ON runs are zeroed.
    """
    if min_on_steps <= 0 and min_gap_steps <= 0:
        return power

    state = power > threshold

    if min_gap_steps > 0:
        starts, ends = _rle_runs(~state)
        for s, e in zip(starts, ends):
            if (e - s) < min_gap_steps and s > 0 and e < len(state):
                state[s:e] = True

    if min_on_steps > 0:
        starts, ends = _rle_runs(state)
        for s, e in zip(starts, ends):
            if (e - s) < min_on_steps:
                state[s:e] = False

    filtered = power.copy()
    was_on   = power > threshold
    filtered[~state] = 0.0

    bridged = state & ~was_on
    if bridged.any():
        idx = np.arange(len(power))
        known = was_on
        if known.any():
            filtered[bridged] = np.interp(idx[bridged], idx[known], power[known])

    return filtered


def denoise_all_appliances(pred_trace: np.ndarray, y_scalers: list,
                           thresholds: dict) -> np.ndarray:
    """Apply denoise_power_trace per appliance (raw-W space) to a scaled prediction trace."""
    out = pred_trace.copy()
    for i, app in enumerate(APPLIANCES):
        params = DENOISE_PARAMS.get(app, {'min_on_min': 0.0, 'min_gap_min': 0.0})
        min_on_steps  = int(round(params['min_on_min']  * 60.0 / STEP_SECONDS))
        min_gap_steps = int(round(params['min_gap_min'] * 60.0 / STEP_SECONDS))
        if min_on_steps <= 0 and min_gap_steps <= 0:
            continue
        raw_p     = y_scalers[i].inverse_transform(pred_trace[:, i:i+1]).flatten()
        raw_p_dn  = denoise_power_trace(raw_p, thresholds[app], min_on_steps, min_gap_steps)
        out[:, i] = y_scalers[i].transform(raw_p_dn.reshape(-1, 1)).flatten()
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def per_app_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_scalers: list, thresholds: dict) -> dict:
    out = {}
    for i, app in enumerate(APPLIANCES):
        raw_t = y_scalers[i].inverse_transform(y_true[:, i:i+1]).flatten()
        raw_p = y_scalers[i].inverse_transform(y_pred[:, i:i+1]).flatten()
        m     = calculate_nilm_metrics(raw_t, raw_p, threshold=thresholds[app])
        thr   = thresholds[app]
        t_on  = raw_t > thr;  p_on = raw_p > thr
        m['tp'] = int(np.sum( t_on &  p_on))
        m['tn'] = int(np.sum(~t_on & ~p_on))
        m['fp'] = int(np.sum(~t_on &  p_on))
        m['fn'] = int(np.sum( t_on & ~p_on))
        out[app] = m
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(data_dict: dict, save_dir: str,
          hidden: int = 64, dt: float = 0.1,
          lambda_phys: float = LAMBDA_PHYS,
          lambda_event: float = LAMBDA_EVENT,
          epsilon_w: float = EPSILON_W) -> tuple:

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}  |  WIN={WIN}  hidden={hidden}  dt={dt}")
    print(f"Model: CombinedPINNAdvancedLNN v3  (gated power = gate x power)\n")
    print(f"Fixes active:")
    print(f"  FIX 1 -- eval thresholds locked to train split for all splits")
    print(f"  FIX 2 -- checkpoint = best min-per-appliance val F1 (not val MSE)")
    print(f"  FIX 3 -- lambda ramp: {RAMP_EPOCHS} epochs linear warmup after epoch {WARMUP_EPOCHS}")
    print(f"  FIX 4 -- min-run-length denoise filter (dishwasher: "
          f"min_on={DENOISE_PARAMS['dishwasher']['min_on_min']:.1f}min "
          f"min_gap={DENOISE_PARAMS['dishwasher']['min_gap_min']:.1f}min)")
    print(f"  FIX 5 -- dishwasher pos_weight ceiling override: "
          f"{POS_WEIGHT_CLAMP_OVERRIDE.get('dishwasher', POS_WEIGHT_CLAMP)}\n")

    df_tr = data_dict['train']
    df_va = data_dict['val']
    df_te = data_dict['test']

    # FIX 1: compute thresholds from train only; reuse for val and test eval
    tr_thr       = compute_adaptive_thresholds(df_tr)
    tr_event_thr = compute_event_thresholds(df_tr)

    va_thr_diag = compute_adaptive_thresholds(df_va)
    te_thr_diag = compute_adaptive_thresholds(df_te)

    print("  Eval thresholds (W)  [all splits evaluated against TRAIN thresholds]:")
    print(f"  {'Appliance':<16} {'Train':>8} {'Val(raw)':>10} {'Test(raw)':>11}  note")
    for app in APPLIANCES:
        diff_va = va_thr_diag[app] - tr_thr[app]
        diff_te = te_thr_diag[app] - tr_thr[app]
        note = "MISMATCH" if abs(diff_te) > 5 else "ok"
        print(f"  {app:<16} {tr_thr[app]:>8.1f} {va_thr_diag[app]:>10.1f} "
              f"{te_thr_diag[app]:>11.1f}  {note} (Δtest={diff_te:+.1f})")
    print(f"  -> Val and Test F1 now computed against train thresholds (Fix 1)\n")

    print("Creating sequences ...")
    X_tr, Y_tr = create_sequences(df_tr, STRIDE)
    X_va, Y_va = create_sequences(df_va, STRIDE)
    X_te, Y_te = create_sequences(df_te, WIN)
    n_feat = X_tr.shape[2]
    n_tr, n_va, n_te = len(df_tr), len(df_va), len(df_te)
    print(f"  Train : {X_tr.shape} -> {Y_tr.shape}  ({X_tr.shape[0]*WIN:,} predictions)")
    print(f"  Val   : {X_va.shape} -> {Y_va.shape}")
    print(f"  Test  : {X_te.shape} -> {Y_te.shape}  [non-overlapping]")

    # Feature scaling (Z-score per channel)
    feat_scalers = []
    for ch in range(n_feat):
        sc = StandardScaler()
        X_tr[:, :, ch] = sc.fit_transform(X_tr[:, :, ch].reshape(-1, 1)).reshape(-1, WIN)
        X_va[:, :, ch] = sc.transform(    X_va[:, :, ch].reshape(-1, 1)).reshape(-1, WIN)
        X_te[:, :, ch] = sc.transform(    X_te[:, :, ch].reshape(-1, 1)).reshape(-1, WIN)
        feat_scalers.append(sc)
    agg_scaler = feat_scalers[0]
    print(f"\n  Agg Z-score: mean={agg_scaler.mean_[0]:.1f} W  std={agg_scaler.scale_[0]:.1f} W")

    # Target scaling (MinMax per appliance)
    y_scalers = []
    for i in range(len(APPLIANCES)):
        ys = MinMaxScaler()
        Y_tr[:, :, i] = ys.fit_transform(Y_tr[:, :, i].reshape(-1, 1)).reshape(-1, WIN)
        Y_va[:, :, i] = ys.transform(    Y_va[:, :, i].reshape(-1, 1)).reshape(-1, WIN)
        Y_te[:, :, i] = ys.transform(    Y_te[:, :, i].reshape(-1, 1)).reshape(-1, WIN)
        y_scalers.append(ys)

    # Scaled event thresholds for gate BCE targets (always from training split)
    thresholds_scaled = [
        (tr_event_thr[app] - float(y_scalers[i].data_min_[0]))
        / float(y_scalers[i].data_range_[0])
        for i, app in enumerate(APPLIANCES)
    ]

    # pos_weight from training ON fraction
    # FIX 5: per-appliance clamp override (dishwasher gets a lower ceiling)
    pos_weights = []
    print("\n  Gate pos_weight per appliance:")
    for i, app in enumerate(APPLIANCES):
        flat   = Y_tr[:, :, i].flatten()
        n_on   = float((flat > thresholds_scaled[i]).sum())
        n_off  = float((flat <= thresholds_scaled[i]).sum())
        clamp  = POS_WEIGHT_CLAMP_OVERRIDE.get(app, POS_WEIGHT_CLAMP)
        raw_pw = n_off / max(n_on, 1.0)
        pw     = float(np.clip(raw_pw, *clamp))
        pos_weights.append(pw)
        note = "  <-- overridden ceiling" if app in POS_WEIGHT_CLAMP_OVERRIDE else ""
        print(f"    {app:<16}  on={100*n_on/(n_on+n_off):5.1f}%  "
              f"raw_ratio={raw_pw:6.1f}  pos_weight={pw:5.1f}  clamp={clamp}{note}")

    tr_ld = torch.utils.data.DataLoader(Seq2SeqDataset(X_tr, Y_tr), batch_size=BATCH, shuffle=True,  drop_last=False)
    va_ld = torch.utils.data.DataLoader(Seq2SeqDataset(X_va, Y_va), batch_size=BATCH, shuffle=False, drop_last=False)
    te_ld = torch.utils.data.DataLoader(Seq2SeqDataset(X_te, Y_te), batch_size=BATCH, shuffle=False, drop_last=False)

    model       = CombinedPINNAdvancedLNN(n_feat, hidden, len(APPLIANCES), dt).to(device)
    mse_crit    = nn.MSELoss()
    phys_crit   = PhysicsConsistencyLoss(agg_scaler, y_scalers, epsilon_w).to(device)
    event_crit  = GatedEventLoss(thresholds_scaled, pos_weights).to(device)
    opt         = torch.optim.Adam(model.parameters(), lr=LR)
    sched       = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=8, min_lr=1e-5)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")
    print(f"Warmup: MSE-only for {WARMUP_EPOCHS} epochs, "
          f"then linear ramp over {RAMP_EPOCHS} epochs, then full loss.\n")

    history = {k: [] for k in ['train_loss','train_mse','train_phys','train_event',
                                'val_loss',  'val_mse',  'val_metrics', 'val_min_f1']}

    # FIX 2: track best min-per-appliance val F1 instead of best val MSE
    best_val_min_f1 = -float('inf')
    best_state      = None
    counter         = 0

    for epoch in range(EPOCHS):
        # FIX 3: compute lambda ramp coefficient
        if epoch < WARMUP_EPOCHS:
            lam = 0.0
            phase = "WARMUP"
        elif epoch < WARMUP_EPOCHS + RAMP_EPOCHS:
            lam = (epoch - WARMUP_EPOCHS + 1) / RAMP_EPOCHS
            phase = f"RAMP({lam:.2f})"
        else:
            lam = 1.0
            phase = "FULL  "

        # -- Train --
        model.train()
        ep_tot = ep_mse = ep_phys = ep_ev = 0.0
        pbar = tqdm(tr_ld, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            gated_power, gate_logits = model(xb)
            l_mse  = mse_crit(gated_power, yb)
            l_phys = phys_crit(xb, gated_power)
            l_ev   = event_crit(gate_logits, yb)
            loss   = l_mse + lam * (lambda_phys * l_phys + lambda_event * l_ev)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_tot  += loss.item();  ep_mse  += l_mse.item()
            ep_phys += l_phys.item(); ep_ev  += l_ev.item()
            pbar.set_postfix({'mse': f'{l_mse.item():.5f}', 'lam': f'{lam:.2f}'})

        nb = len(tr_ld)
        history['train_loss'].append(ep_tot  / nb)
        history['train_mse'].append(ep_mse   / nb)
        history['train_phys'].append(ep_phys / nb)
        history['train_event'].append(ep_ev  / nb)

        # -- Validate --
        model.eval()
        vl_mse = vl_tot = 0.0
        va_preds, va_trues = [], []
        with torch.no_grad():
            for xb, yb in va_ld:
                xb, yb = xb.to(device), yb.to(device)
                gated_power, _ = model(xb)
                l_mse  = mse_crit(gated_power, yb)
                l_phys = phys_crit(xb, gated_power)
                vl_mse += l_mse.item()
                vl_tot += (l_mse + lambda_phys * l_phys).item()
                for b in range(gated_power.shape[0]):
                    va_preds.append(gated_power[b].cpu().numpy())
                    va_trues.append(yb[b].cpu().numpy())

        avg_va_mse = vl_mse / len(va_ld)
        history['val_loss'].append(vl_tot / len(va_ld))
        history['val_mse'].append(avg_va_mse)
        sched.step(avg_va_mse)

        pred_trace = reconstruct_trace(va_preds, n_va, STRIDE, WIN)
        true_trace = reconstruct_trace(va_trues, n_va, STRIDE, WIN)
        # FIX 4: denoise before scoring, same as test-time eval
        pred_trace_dn = denoise_all_appliances(pred_trace, y_scalers, tr_thr)
        # FIX 1: evaluate val against train thresholds
        vm = per_app_metrics(true_trace, pred_trace_dn, y_scalers, tr_thr)
        history['val_metrics'].append(vm)

        avg_f1  = np.mean([vm[a]['f1']  for a in APPLIANCES])
        # FIX 2: selection score = minimum per-appliance F1
        min_f1  = float(np.min([vm[a]['f1']  for a in APPLIANCES]))
        avg_mae = np.mean([vm[a]['mae'] for a in APPLIANCES])
        history['val_min_f1'].append(min_f1)

        ckpt_marker = " <-- BEST" if min_f1 > best_val_min_f1 else ""
        print(f"  [{phase}] Epoch {epoch+1:3d}/{EPOCHS}  "
              f"train={history['train_loss'][-1]:.5f} "
              f"(mse={history['train_mse'][-1]:.5f} phys={history['train_phys'][-1]:.5f} "
              f"ev={history['train_event'][-1]:.5f})  "
              f"val_mse={avg_va_mse:.5f}  avgF1={avg_f1:.4f}  minF1={min_f1:.4f}  "
              f"avgMAE={avg_mae:.2f}  lr={opt.param_groups[0]['lr']:.2e}{ckpt_marker}")
        for app in APPLIANCES:
            m = vm[app]
            print(f"    {app:<18}  F1={m['f1']:.4f}  P={m['precision']:.4f}  "
                  f"R={m['recall']:.4f}  MAE={m['mae']:.2f}  SAE={m['sae']:.4f}  "
                  f"TP={m['tp']:,d}  TN={m['tn']:,d}  FP={m['fp']:,d}  FN={m['fn']:,d}")

        # FIX 2: update checkpoint on best *minimum* per-appliance val F1
        if min_f1 > best_val_min_f1:
            best_val_min_f1 = min_f1
            best_state      = {k: v.clone() for k, v in model.state_dict().items()}
            counter         = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch+1}  "
                      f"(best minF1={best_val_min_f1:.4f})")
                break

    print("\nTraining complete.")

    # -- Test --
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    te_preds, te_trues = [], []
    with torch.no_grad():
        for xb, yb in te_ld:
            gated_power, _ = model(xb.to(device))
            for b in range(gated_power.shape[0]):
                te_preds.append(gated_power[b].cpu().numpy())
                te_trues.append(yb[b].cpu().numpy())

    pred_trace_te = reconstruct_trace(te_preds, n_te, WIN, WIN)
    true_trace_te = reconstruct_trace(te_trues, n_te, WIN, WIN)
    # FIX 1: evaluate test against train thresholds
    test_metrics_raw = per_app_metrics(true_trace_te, pred_trace_te, y_scalers, tr_thr)

    # FIX 4: denoised test metrics -- printed alongside raw so the effect is visible
    pred_trace_te_dn = denoise_all_appliances(pred_trace_te, y_scalers, tr_thr)
    test_metrics      = per_app_metrics(true_trace_te, pred_trace_te_dn, y_scalers, tr_thr)

    print(f"\n{'Appliance':<18} {'F1':>7} {'Prec':>7} {'Rec':>7} "
          f"{'MAE':>7} {'SAE':>7} {'TP':>7} {'TN':>7} {'FP':>7} {'FN':>7}   [pre-filter]")
    print("-" * 95)
    for app in APPLIANCES:
        m = test_metrics_raw[app]
        print(f"{app:<18} {m['f1']:>7.4f} {m['precision']:>7.4f} {m['recall']:>7.4f} "
              f"{m['mae']:>7.2f} {m['sae']:>7.4f} {m['tp']:>7,d} {m['tn']:>7,d} "
              f"{m['fp']:>7,d} {m['fn']:>7,d}")

    print(f"\n{'Appliance':<18} {'F1':>7} {'Prec':>7} {'Rec':>7} "
          f"{'MAE':>7} {'SAE':>7} {'TP':>7} {'TN':>7} {'FP':>7} {'FN':>7}   [FIX 4 denoised]")
    print("-" * 95)
    for app in APPLIANCES:
        m = test_metrics[app]
        d_f1 = m['f1'] - test_metrics_raw[app]['f1']
        d_p  = m['precision'] - test_metrics_raw[app]['precision']
        print(f"{app:<18} {m['f1']:>7.4f} {m['precision']:>7.4f} {m['recall']:>7.4f} "
              f"{m['mae']:>7.2f} {m['sae']:>7.4f} {m['tp']:>7,d} {m['tn']:>7,d} "
              f"{m['fp']:>7,d} {m['fn']:>7,d}   (ΔF1={d_f1:+.4f} ΔP={d_p:+.4f})")

    # -- Save checkpoint + config --
    ckpt_path = os.path.join(save_dir, 'best_model.pt')
    torch.save(best_state, ckpt_path)
    print(f"\n  Checkpoint saved: {ckpt_path}")

    _plot_loss(history, save_dir)
    _plot_metrics(history, test_metrics, save_dir)

    cfg = {
        'version': 'v3',
        'fixes': {
            'fix1_threshold_consistency': 'train thresholds used for val and test eval',
            'fix2_checkpoint_criterion':  'min per-appliance val F1 (not val MSE)',
            'fix3_lambda_ramp':           f'linear ramp over {RAMP_EPOCHS} epochs after warmup',
            'fix4_denoise_filter':        DENOISE_PARAMS,
            'fix5_pos_weight_override':   {k: list(v) for k, v in POS_WEIGHT_CLAMP_OVERRIDE.items()},
        },
        'dataset': 'APR-new-House1-dataset (H1 train 2014-09-08..10-07 / val 10-08..10-14 / test 10-15..10-21)',
        'model':   'CombinedPINNAdvancedLNN',
        'description': 'gated power = sigmoid(gate_logit) x sigmoid(power_logit), + FIX4 post-hoc denoise',
        'thresholds': {
            'train': tr_thr,
            'val_raw':  va_thr_diag,
            'test_raw': te_thr_diag,
            'used_for_eval': tr_thr,
        },
        'pos_weights': dict(zip(APPLIANCES, [float(w) for w in pos_weights])),
        'window': {'win': WIN, 'stride_train': STRIDE, 'stride_test': WIN},
        'model_params': {'in_ch': n_feat, 'hidden': hidden, 'n_apps': len(APPLIANCES), 'dt': dt},
        'train_params': {'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE,
                         'lambda_phys': lambda_phys, 'lambda_event': lambda_event,
                         'epsilon_w': epsilon_w, 'warmup_epochs': WARMUP_EPOCHS,
                         'ramp_epochs': RAMP_EPOCHS},
        'best_val_min_f1': float(best_val_min_f1),
        'test_metrics_pre_filter': {app: {k: float(v) for k, v in m.items()}
                                    for app, m in test_metrics_raw.items()},
        'test_metrics': {app: {k: float(v) for k, v in m.items()}
                         for app, m in test_metrics.items()},
    }
    with open(os.path.join(save_dir, 'combined_pinn_lnn_apr_new_house1_dataset_v3_results.json'), 'w') as f:
        json.dump(cfg, f, indent=4)

    print(f"  Results saved to: {save_dir}")
    return test_metrics, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_loss(history, save_dir):
    ep = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle('CombinedPINNAdvancedLNN v3 (APR-new-House1-dataset) -- Loss Curves')
    pairs = [('train_loss', 'val_loss',   'Total Loss'),
             ('train_mse',  'val_mse',    'MSE Loss'),
             ('train_phys', None,         'Physics Loss (train)'),
             ('train_event',None,         'Gate-Event BCE (train)')]
    for ax, (tr_k, va_k, title) in zip(axes, pairs):
        ax.plot(ep, history[tr_k], label='Train', color='steelblue')
        if va_k:
            ax.plot(ep, history[va_k], label='Val', color='tomato')
        ax.set_title(title); ax.set_xlabel('Epoch')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'combined_pinn_lnn_apr_new_house1_dataset_v3_loss.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def _plot_metrics(history, test_metrics, save_dir):
    ep = range(1, len(history['train_loss']) + 1)

    fig_top, ax_top = plt.subplots(figsize=(10, 3))
    ax_top.plot(ep, history['val_min_f1'], color='darkorange', label='val min-F1 (ckpt criterion)')
    best_ep = int(np.argmax(history['val_min_f1'])) + 1
    ax_top.axvline(best_ep, color='green', linestyle='--', label=f'best (ep {best_ep})')
    ax_top.set_title('Checkpoint selection: min per-appliance val F1  [Fix 2, denoised via Fix 4]')
    ax_top.set_xlabel('Epoch'); ax_top.legend(); ax_top.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'combined_pinn_lnn_apr_new_house1_dataset_v3_minf1.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(len(APPLIANCES), 3, figsize=(17, 4 * len(APPLIANCES)))
    fig.suptitle('CombinedPINNAdvancedLNN v3 (APR-new-House1-dataset) -- Per-Appliance Val Metrics')
    for row, app in enumerate(APPLIANCES):
        f1s  = [m[app]['f1']  for m in history['val_metrics']]
        maes = [m[app]['mae'] for m in history['val_metrics']]
        saes = [m[app]['sae'] for m in history['val_metrics']]
        axes[row][0].plot(ep, f1s, color='steelblue')
        axes[row][0].axvline(best_ep, color='green', linestyle='--', alpha=0.7, label=f'ckpt ep{best_ep}')
        axes[row][0].axhline(test_metrics[app]['f1'], color='green', linestyle=':', label='Test')
        axes[row][0].set_title(f'{app} -- F1'); axes[row][0].legend(fontsize=7); axes[row][0].grid(alpha=0.3)
        axes[row][1].plot(ep, maes, color='tomato')
        axes[row][1].axhline(test_metrics[app]['mae'], color='green', linestyle='--', label='Test')
        axes[row][1].set_title(f'{app} -- MAE (W)'); axes[row][1].legend(); axes[row][1].grid(alpha=0.3)
        axes[row][2].plot(ep, saes, color='purple')
        axes[row][2].axhline(test_metrics[app]['sae'], color='green', linestyle='--', label='Test')
        axes[row][2].set_title(f'{app} -- SAE'); axes[row][2].legend(); axes[row][2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'combined_pinn_lnn_apr_new_house1_dataset_v3_per_appliance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', default=DEFAULT_DATASET_DIR)
    parser.add_argument('--hidden',       type=int,   default=64)
    parser.add_argument('--lambda-phys',  type=float, default=LAMBDA_PHYS)
    parser.add_argument('--lambda-event', type=float, default=LAMBDA_EVENT)
    parser.add_argument('--epsilon-w',    type=float, default=EPSILON_W)
    args = parser.parse_args()

    for f in ['UKDALE_HF_train.csv', 'UKDALE_HF_validation.csv', 'UKDALE_HF_test.csv']:
        p = os.path.join(args.dataset_dir, f)
        if not os.path.exists(p):
            print(f"Error: {p} not found."); sys.exit(1)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir  = os.path.join(
        os.path.dirname(__file__), '..', 'models',
        f'combined_pinn_lnn_apr_new_house1_dataset_v3_{timestamp}'
    )

    data_dict = load_data(args.dataset_dir)
    train(data_dict, save_dir,
          hidden       = args.hidden,
          dt           = 0.1,
          lambda_phys  = args.lambda_phys,
          lambda_event = args.lambda_event,
          epsilon_w    = args.epsilon_w)
