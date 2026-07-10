"""
combined_pinn_lnn_apr_new_house1_dataset.py
=============================================
Dual-head PINN + Gated-Detection LNN for NILM -- APR-new-House1-dataset/ splits.

Ported from combined_pinn_lnn_ukdale.py (root) with two changes:

1. DATA SOURCE -- default --dataset-dir now points at APR-new-House1-dataset/
   instead of medium_dataset/:
       APR-new-House1-dataset/UKDALE_HF_train.csv      (House 1, 2014-09-08 to 2014-10-07, 30 days)
       APR-new-House1-dataset/UKDALE_HF_validation.csv (House 1, 2014-10-08 to 2014-10-14, 7 days)
       APR-new-House1-dataset/UKDALE_HF_test.csv       (House 1, 2014-10-15 to 2014-10-21, 7 days)
   No other change needed here -- the source script's APPLIANCES list
   ('dishwasher', 'fridge', 'microwave', 'washing_machine') already matches
   this dataset's column names exactly (unlike the Old-LNN_Algorithms scripts,
   which used 'dish washer'/'washer dryer' from the old pickled splits).

2. PATH FIXES -- sys.path insert and default save_dir adjusted for this
   script living one directory deeper (APR-new-scripts/ instead of root).

Everything else -- architecture, losses, adaptive thresholds, Seq2Seq window
construction -- is unchanged from the source script:

Architecture
------------
    Input (batch, WIN=299, 8)   -- 8-channel mains features (raw, median
                                    filtered, EMA smoothed, residual, delta,
                                    delta_smooth, rolling mean, rolling std)
           |
    LengthPreservingCNN     -- dilated conv RF = 183 steps
           |
    Bidirectional LiquidCell -- fwd + bwd, concat -> hidden*2
           |
    LayerNorm(hidden*2)
           | (per appliance)
    Gate  head: Linear -> sigmoid  ->  g in [0,1]   (detection confidence)
    Power head: Linear -> sigmoid  ->  p in [0,1]   (raw power estimate)
           |
    gated_power = g x p     <- the combination

    Returns: gated_power (batch, WIN, n_apps)   -- used for MSE + physics
             gate_logits (batch, WIN, n_apps)   -- used for BCEWithLogits

Why this works
--------------
The gate is the detection head from "Advanced LNN". When g ~ 0 the power
estimate is suppressed to near-zero regardless of what the power head says,
directly fixing the precision problem (false positives) we kept seeing across
every PINN-LNN run on this dataset (recall~1.0, precision near 0). The
physics loss acts on gated_power, so it also benefits from the suppression.

Joint training signal:
  - MSE(gated_power, y_true)   drives both heads simultaneously
  - BCEWithLogits(gate_logit)  explicitly trains the detection gate
                                (pos_weight per appliance from training ON
                                fraction -- handles the severe class
                                imbalance found in this dataset's dishwasher
                                and microwave channels)
  - L_phys(gated_power)        physics acts on the final gated estimate

Loss schedule:
  Warmup (0 - WARMUP_EPOCHS-1):  MSE only
  Full   (WARMUP_EPOCHS +):       MSE + lambda_phys * L_phys + lambda_event * L_event

WIN=299 (~30 min at 6s) was chosen in the source script to cover a full
fridge compressor cycle (24.6 min avg); our own on-duration analysis of this
dataset (see analyze_apr_new_house1_distributions.py) found dishwasher
segments running up to 51 min (p95) and fridge medians of 11-16 min, which
WIN=100/200 (used by the other PINN-LNN scripts in this folder) cannot fully
see in one window -- this is the main architectural reason to prefer this
model over test_pinn_lnn_apr_new_house1_dataset.py /
test_pinn_advanced_lnn_apr_new_house1_dataset.py for this dataset.
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

THRESHOLD_DELTA   = 20.0
THRESHOLD_LOW_PCT = 0.05
THRESHOLD_MIN     = 10.0
CYCLING_P5_W      = 80.0
POS_WEIGHT_CLAMP  = (1.0, 50.0)

APPLIANCES  = ['dishwasher', 'fridge', 'microwave', 'washing_machine']
AGG_COL     = 'aggregate'

DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'APR-new-House1-dataset')


# ---------------------------------------------------------------------------
# Adaptive thresholds  (identical to fixed_pinn_lnn_ukdale.py / combined_pinn_lnn_ukdale.py)
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
# 8-channel feature extraction  (identical to fixed_pinn_lnn_ukdale.py)
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
# Sequence creation  (Seq2Seq, identical to fixed_pinn_lnn_ukdale.py)
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
# Model components  (CNN + LiquidCell identical to fixed_pinn_lnn_ukdale.py)
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
# Combined model  -- the key model class
# ---------------------------------------------------------------------------

class CombinedPINNAdvancedLNN(nn.Module):
    """
    Dual-head model: gated power estimation.

    gate_heads  produce detection confidence g in [0,1] at every timestep.
    power_heads produce raw power estimate   p in [0,1] at every timestep.
    Output:  gated_power = g x p  (batch, WIN, n_apps)

    The gate directly suppresses false-positive power predictions:
    if detection says the appliance is OFF (g ~ 0), the physics-constrained
    power estimate is zeroed regardless of what the regression head says.
    """

    def __init__(self, in_ch: int, hidden: int, n_apps: int, dt: float = 0.1):
        super().__init__()
        self.hidden = hidden
        self.n_apps = n_apps

        self.cnn      = LengthPreservingCNN(in_ch, hidden)
        self.fwd_cell = LiquidCell(hidden, hidden, dt)
        self.bwd_cell = LiquidCell(hidden, hidden, dt)
        self.norm     = nn.LayerNorm(hidden * 2)

        # Detection gate heads -- logits for BCEWithLogits; sigmoid = gate
        self.gate_heads  = nn.ModuleList([nn.Linear(hidden * 2, 1) for _ in range(n_apps)])
        # Power regression heads -- sigmoid output in [0,1]
        self.power_heads = nn.ModuleList([nn.Linear(hidden * 2, 1) for _ in range(n_apps)])

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, WIN, in_ch)
        Returns:
            gated_power : (batch, WIN, n_apps)  values in [0,1]
            gate_logits : (batch, WIN, n_apps)  raw logits for BCEWithLogits
        """
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

            # Gate logits and sigmoid gate
            g_logits = torch.cat([head(h_t) for head in self.gate_heads], dim=1)   # (batch, n_apps)
            g        = torch.sigmoid(g_logits)                                       # detection gate

            # Power estimate
            p = torch.cat([torch.sigmoid(head(h_t)) for head in self.power_heads], dim=1)

            gated_power_list.append(g * p)
            gate_logit_list.append(g_logits)

        return (torch.stack(gated_power_list, dim=1),    # (batch, WIN, n_apps)
                torch.stack(gate_logit_list,  dim=1))    # (batch, WIN, n_apps)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class PhysicsConsistencyLoss(nn.Module):
    """ReLU(sum(gated_power_raw) - P_agg_raw - epsilon) -- acts on the gated estimate."""

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
    """
    BCEWithLogits on gate logits.  Trains the gate head as a detection classifier.
    pos_weight compensates for the ON fraction being very small (~1-6% for
    dishwasher/microwave/washing_machine in this dataset).
    """

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
    print(f"Model: CombinedPINNAdvancedLNN  (gated power = gate x power)\n")

    df_tr = data_dict['train']
    df_va = data_dict['val']
    df_te = data_dict['test']

    tr_thr        = compute_adaptive_thresholds(df_tr)
    va_thr        = compute_adaptive_thresholds(df_va)
    te_thr        = compute_adaptive_thresholds(df_te)
    tr_event_thr  = compute_event_thresholds(df_tr)

    print("  Eval thresholds (W):")
    print(f"  {'Appliance':<16} {'Train':>8} {'Val':>8} {'Test':>8}")
    for app in APPLIANCES:
        print(f"  {app:<16} {tr_thr[app]:>8.1f} {va_thr[app]:>8.1f} {te_thr[app]:>8.1f}")

    print("\nCreating sequences ...")
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

    # Scaled event thresholds for gate BCE targets
    thresholds_scaled = [
        (tr_event_thr[app] - float(y_scalers[i].data_min_[0]))
        / float(y_scalers[i].data_range_[0])
        for i, app in enumerate(APPLIANCES)
    ]

    # pos_weight from training ON fraction
    pos_weights = []
    print("\n  Gate pos_weight per appliance:")
    for i, app in enumerate(APPLIANCES):
        flat  = Y_tr[:, :, i].flatten()
        n_on  = float((flat > thresholds_scaled[i]).sum())
        n_off = float((flat <= thresholds_scaled[i]).sum())
        pw    = float(np.clip(n_off / max(n_on, 1.0), *POS_WEIGHT_CLAMP))
        pos_weights.append(pw)
        print(f"    {app:<16}  on={100*n_on/(n_on+n_off):5.1f}%  pos_weight={pw:.1f}")

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
    print(f"Warmup: MSE-only for {WARMUP_EPOCHS} epochs, then + physics + gate-event.\n")

    history = {k: [] for k in ['train_loss','train_mse','train_phys','train_event',
                                'val_loss',  'val_mse',  'val_metrics']}
    best_val_mse = float('inf')
    best_state   = None
    counter      = 0

    for epoch in range(EPOCHS):
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
            loss   = l_mse if epoch < WARMUP_EPOCHS else (
                     l_mse + lambda_phys * l_phys + lambda_event * l_ev)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_tot  += loss.item();  ep_mse  += l_mse.item()
            ep_phys += l_phys.item(); ep_ev  += l_ev.item()
            pbar.set_postfix({'mse': f'{l_mse.item():.5f}', 'phys': f'{l_phys.item():.5f}'})

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
        vm = per_app_metrics(true_trace, pred_trace, y_scalers, va_thr)
        history['val_metrics'].append(vm)

        avg_f1  = np.mean([vm[a]['f1']  for a in APPLIANCES])
        avg_mae = np.mean([vm[a]['mae'] for a in APPLIANCES])
        status  = "WARMUP" if epoch < WARMUP_EPOCHS else "FULL  "
        print(f"  [{status}] Epoch {epoch+1:3d}/{EPOCHS}  "
              f"train={history['train_loss'][-1]:.5f} "
              f"(mse={history['train_mse'][-1]:.5f} phys={history['train_phys'][-1]:.5f} "
              f"ev={history['train_event'][-1]:.5f})  "
              f"val_mse={avg_va_mse:.5f}  avgF1={avg_f1:.4f}  avgMAE={avg_mae:.2f}  "
              f"lr={opt.param_groups[0]['lr']:.2e}")
        for app in APPLIANCES:
            m = vm[app]
            print(f"    {app:<18}  F1={m['f1']:.4f}  P={m['precision']:.4f}  "
                  f"R={m['recall']:.4f}  MAE={m['mae']:.2f}  SAE={m['sae']:.4f}  "
                  f"TP={m['tp']:,d}  TN={m['tn']:,d}  FP={m['fp']:,d}  FN={m['fn']:,d}")

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
    test_metrics  = per_app_metrics(true_trace_te, pred_trace_te, y_scalers, te_thr)

    print(f"\n{'Appliance':<18} {'F1':>7} {'Prec':>7} {'Rec':>7} "
          f"{'MAE':>7} {'SAE':>7} {'TP':>7} {'TN':>7} {'FP':>7} {'FN':>7}")
    print("-" * 85)
    for app in APPLIANCES:
        m = test_metrics[app]
        print(f"{app:<18} {m['f1']:>7.4f} {m['precision']:>7.4f} {m['recall']:>7.4f} "
              f"{m['mae']:>7.2f} {m['sae']:>7.4f} {m['tp']:>7,d} {m['tn']:>7,d} "
              f"{m['fp']:>7,d} {m['fn']:>7,d}")

    # -- Save checkpoint + config --
    ckpt_path = os.path.join(save_dir, 'best_model.pt')
    torch.save(best_state, ckpt_path)
    print(f"\n  Checkpoint saved: {ckpt_path}")

    _plot_loss(history, save_dir)
    _plot_metrics(history, test_metrics, save_dir)

    cfg = {
        'dataset': 'APR-new-House1-dataset (H1 train 2014-09-08..10-07 / val 10-08..10-14 / test 10-15..10-21)',
        'model':   'CombinedPINNAdvancedLNN',
        'description': 'gated power = sigmoid(gate_logit) x sigmoid(power_logit)',
        'thresholds': {'train': tr_thr, 'val': va_thr, 'test': te_thr},
        'pos_weights': dict(zip(APPLIANCES, [float(w) for w in pos_weights])),
        'window': {'win': WIN, 'stride_train': STRIDE, 'stride_test': WIN},
        'model_params': {'in_ch': n_feat, 'hidden': hidden, 'n_apps': len(APPLIANCES), 'dt': dt},
        'train_params': {'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE,
                         'lambda_phys': lambda_phys, 'lambda_event': lambda_event,
                         'epsilon_w': epsilon_w, 'warmup_epochs': WARMUP_EPOCHS},
        'test_metrics': {app: {k: float(v) for k, v in m.items()}
                         for app, m in test_metrics.items()},
    }
    with open(os.path.join(save_dir, 'combined_pinn_lnn_apr_new_house1_dataset_results.json'), 'w') as f:
        json.dump(cfg, f, indent=4)

    print(f"  Results saved to: {save_dir}")
    return test_metrics, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_loss(history, save_dir):
    ep = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle('CombinedPINNAdvancedLNN (APR-new-House1-dataset) -- Loss Curves')
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
    plt.savefig(os.path.join(save_dir, 'combined_pinn_lnn_apr_new_house1_dataset_loss.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def _plot_metrics(history, test_metrics, save_dir):
    ep = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(len(APPLIANCES), 3, figsize=(17, 4 * len(APPLIANCES)))
    fig.suptitle('CombinedPINNAdvancedLNN (APR-new-House1-dataset) -- Per-Appliance Validation Metrics')
    for row, app in enumerate(APPLIANCES):
        f1s  = [m[app]['f1']  for m in history['val_metrics']]
        maes = [m[app]['mae'] for m in history['val_metrics']]
        saes = [m[app]['sae'] for m in history['val_metrics']]
        axes[row][0].plot(ep, f1s, color='steelblue')
        axes[row][0].axhline(test_metrics[app]['f1'], color='green', linestyle='--', label='Test')
        axes[row][0].set_title(f'{app} -- F1'); axes[row][0].legend(); axes[row][0].grid(alpha=0.3)
        axes[row][1].plot(ep, maes, color='tomato')
        axes[row][1].axhline(test_metrics[app]['mae'], color='green', linestyle='--', label='Test')
        axes[row][1].set_title(f'{app} -- MAE (W)'); axes[row][1].legend(); axes[row][1].grid(alpha=0.3)
        axes[row][2].plot(ep, saes, color='purple')
        axes[row][2].axhline(test_metrics[app]['sae'], color='green', linestyle='--', label='Test')
        axes[row][2].set_title(f'{app} -- SAE'); axes[row][2].legend(); axes[row][2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'combined_pinn_lnn_apr_new_house1_dataset_per_appliance.png'),
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
        f'combined_pinn_lnn_apr_new_house1_dataset_{timestamp}'
    )

    data_dict = load_data(args.dataset_dir)
    train(data_dict, save_dir,
          hidden       = args.hidden,
          dt           = 0.1,
          lambda_phys  = args.lambda_phys,
          lambda_event = args.lambda_event,
          epsilon_w    = args.epsilon_w)
