"""
PINN-LNN UKDALE — v4 (old dataset) + FNO-inspired spectral input features.

Extends test_pinn_lnn_ukdale_old_dataset_v4.py with frequency-domain
pre-processing inspired by Fourier Neural Operators (FNOs):

  Spectral pre-processing:
    1. Hanning window applied to each mains segment to reduce spectral
       leakage at window edges:
           x_windowed[n] = x[n] * 0.5 * (1 - cos(2π n / (N-1)))
    2. Real FFT of the windowed segment; first N_MODES magnitude bins kept.
    3. FFT magnitudes (shape: N_MODES) are broadcast across the WIN time axis
       and concatenated with the raw mains signal as extra input channels:
           input shape: (batch, WIN, 1 + N_MODES)
       Channel 0 = MinMaxScaled raw mains.
       Channels 1..N_MODES = StandardScaled FFT magnitudes (constant per
       window — global spectral context injected at every LiquidCell step).

Why N_MODES=16 with WIN=100 at 6s sampling?
    rfft gives 51 bins; bin k corresponds to period WIN*6/k seconds.
    Bin 1 → 600 s (10 min), Bin 8 → 75 s, Bin 16 → 37.5 s.
    N_MODES=16 captures all appliance-relevant switching patterns without
    feeding high-frequency noise into the cell.

Everything else (LiquidCell, seq2seq heads, physics loss, stitched metrics,
adaptive thresholds, TP/TN/FP/FN) is identical to old_dataset_v4.

Expected CSV columns:
    timestamp, aggregate, dishwasher, fridge, microwave, washing_machine
"""

import argparse
import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Source Code"))
from utils import calculate_nilm_metrics


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPOCHS   = 80
PATIENCE = 20
LR       = 1e-3
BATCH    = 32
WIN      = 100
STRIDE   = 5

LAMBDA_PHYS   = 0.01
EPSILON_W     = 50.0
LAMBDA_SPARSE = 0.005  # L1 sparsity: penalises non-zero predictions, fixes over-prediction

APPLIANCES = ['dishwasher', 'fridge', 'microwave', 'washing_machine']
AGG_COL    = 'aggregate'

THRESHOLD_DELTA   = 20.0
THRESHOLD_LOW_PCT = 0.05
THRESHOLD_MIN     = 10.0
CYCLING_P5_W      = 80.0

# Spectral pre-processing
N_MODES = 16          # FFT magnitude bins (0..N_MODES-1) kept as extra channels
N_IN    = 1 + N_MODES # total LiquidCell input channels

SPLIT_FILES = {
    'train': 'UKDALE_HF_train.csv',
    'val':   'UKDALE_HF_validation.csv',
    'test':  'UKDALE_HF_test.csv',
}

DEFAULT_DATASET_DIR = 'dataset'


# ---------------------------------------------------------------------------
# Frequency-domain helpers
# ---------------------------------------------------------------------------

def _make_hanning(n: int) -> np.ndarray:
    return (0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / (n - 1)))
            ).astype(np.float32)

_HANN = _make_hanning(WIN)   # precomputed once at import time


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
        if p5 > CYCLING_P5_W:
            thresholds[app] = THRESHOLD_MIN
        else:
            thresholds[app] = max(p5 + THRESHOLD_DELTA, THRESHOLD_MIN)
    return thresholds


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class Seq2SeqDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def load_split(dataset_dir: str, split_name: str) -> pd.DataFrame:
    path = os.path.join(dataset_dir, SPLIT_FILES[split_name])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {split_name} split: {path}")
    df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
    missing = [c for c in [AGG_COL] + APPLIANCES if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df[[AGG_COL] + APPLIANCES].astype('float32').clip(lower=0.0)


def load_dataset(dataset_dir: str) -> dict:
    dataset_dir = os.path.abspath(dataset_dir)
    print(f"Loading UKDALE data from: {dataset_dir}")
    data = {s: load_split(dataset_dir, s) for s in ['train', 'val', 'test']}
    for split, df in data.items():
        print(f"  {split:<5} rows={len(df):6d}  "
              f"{df.index.min().date()} -> {df.index.max().date()}")
    print(f"  Columns: {list(data['train'].columns)}")
    return data


def create_sequences(df: pd.DataFrame, stride: int):
    """
    Returns:
        X : (M, WIN, N_IN)  — channel 0: raw mains; channels 1..N_MODES: FFT mags
        Y : (M, WIN, n_apps)
    """
    mains   = df[AGG_COL].values.astype(np.float32)
    targets = df[APPLIANCES].values.astype(np.float32)
    X, Y = [], []
    for i in range(0, len(mains) - WIN + 1, stride):
        seg      = mains[i: i + WIN]
        windowed = seg * _HANN                                # Hanning window
        fft_mag  = np.abs(np.fft.rfft(windowed))[:N_MODES]   # (N_MODES,)
        fft_feat = np.tile(fft_mag, (WIN, 1))                 # (WIN, N_MODES)
        x_feat   = np.concatenate([seg.reshape(WIN, 1), fft_feat], axis=1)
        X.append(x_feat)
        Y.append(targets[i: i + WIN])
    return (np.array(X, np.float32),   # (M, WIN, N_IN)
            np.array(Y, np.float32))   # (M, WIN, n_apps)


def sequence_starts(n_rows: int, stride: int) -> np.ndarray:
    return np.arange(0, n_rows - WIN + 1, stride, dtype=np.int64)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class PhysicsConsistencyLoss(nn.Module):
    """
    One-sided energy-conservation penalty at every timestep.
    Only channel 0 (MinMaxScaled raw mains) is used for the inverse transform;
    the spectral channels are ignored here.
    """

    def __init__(self, x_scaler, y_scalers, epsilon_w=EPSILON_W):
        super().__init__()
        self.epsilon = epsilon_w
        self.register_buffer('x_min',
            torch.tensor(float(x_scaler.data_min_[0]),   dtype=torch.float32))
        self.register_buffer('x_range',
            torch.tensor(float(x_scaler.data_range_[0]), dtype=torch.float32))
        self.register_buffer('y_mins',
            torch.tensor([float(s.data_min_[0])   for s in y_scalers],
                         dtype=torch.float32))
        self.register_buffer('y_ranges',
            torch.tensor([float(s.data_range_[0]) for s in y_scalers],
                         dtype=torch.float32))

    def forward(self, x_scaled, pred_scaled):
        # Channel 0 = MinMaxScaled raw mains; spectral channels ignored.
        x_raw = x_scaled[:, :, 0] * self.x_range + self.x_min
        p_raw = (pred_scaled * self.y_ranges.view(1, 1, -1)
                 + self.y_mins.view(1, 1, -1))
        return F.relu(p_raw.sum(dim=2) - x_raw - self.epsilon).mean()


# ---------------------------------------------------------------------------
# Model
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
        tau = (F.softplus(self.tau_base).unsqueeze(0)
               * torch.sigmoid(self.tau_mod(x_t))).clamp(min=self.dt)
        g   = torch.sigmoid(self.gate(torch.cat([x_t, h], dim=1)))
        dh  = ((-h / tau) + g * torch.tanh(ip + rp)) * self.dt
        return (h + dh).clamp(-10.0, 10.0)


class PINNLNNSeq2Seq(nn.Module):
    """
    Forward LiquidCell with per-appliance linear heads.
    Input channels: 1 (raw mains) + N_MODES (FFT magnitudes).
    Output: (B, T, n_apps).
    """

    def __init__(self, input_size, hidden_size, n_apps, dt=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell  = LiquidCell(input_size, hidden_size, dt)
        self.norm  = nn.LayerNorm(hidden_size)
        self.heads = nn.ModuleList([nn.Linear(hidden_size, 1)
                                    for _ in range(n_apps)])

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        outs = []
        for t in range(T):
            h = self.cell(x[:, t, :], h)
            hn = self.norm(h)
            outs.append(torch.cat([head(hn) for head in self.heads], dim=1))
        return torch.stack(outs, dim=1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def stitch_predictions(window_preds: np.ndarray, starts: np.ndarray,
                       n_rows: int) -> np.ndarray:
    n_apps = window_preds.shape[-1]
    acc    = np.zeros((n_rows, n_apps), np.float64)
    cnt    = np.zeros((n_rows, 1),     np.float64)
    for idx, s in enumerate(starts):
        e = s + WIN
        if e > n_rows:
            break
        acc[s:e] += window_preds[idx]
        cnt[s:e] += 1.0
    stitched = np.full((n_rows, n_apps), np.nan, np.float32)
    valid    = cnt[:, 0] > 0
    stitched[valid] = (acc[valid] / cnt[valid]).astype(np.float32)
    return stitched


def compute_metrics(df: pd.DataFrame, window_preds: np.ndarray,
                    y_scalers: list, thresholds: dict,
                    stride: int) -> tuple:
    starts   = sequence_starts(len(df), stride)
    stitched = stitch_predictions(window_preds, starts, len(df))
    valid    = ~np.isnan(stitched).any(axis=1)

    raw_pred = np.full_like(stitched, np.nan)
    raw_true = df[APPLIANCES].values.astype(np.float32)
    for i, sc in enumerate(y_scalers):
        raw_pred[valid, i] = sc.inverse_transform(
            stitched[valid, i:i+1]).flatten()

    metrics = {}
    for i, app in enumerate(APPLIANCES):
        m    = calculate_nilm_metrics(raw_true[valid, i], raw_pred[valid, i],
                                      threshold=thresholds[app])
        thr  = thresholds[app]
        t_on = raw_true[valid, i] > thr
        p_on = raw_pred[valid, i] > thr
        m['tp'] = int(np.sum( t_on &  p_on))
        m['tn'] = int(np.sum(~t_on & ~p_on))
        m['fp'] = int(np.sum(~t_on &  p_on))
        m['fn'] = int(np.sum( t_on & ~p_on))
        metrics[app] = m
    return metrics, raw_pred, valid


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(data_dict: dict, save_dir: str,
                hidden_size: int   = 64,
                dt:          float = 0.1,
                lambda_phys: float = LAMBDA_PHYS,
                epsilon_w:   float = EPSILON_W) -> tuple:

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}  |  WIN={WIN}  STRIDE={STRIDE}  "
          f"hidden={hidden_size}  dt={dt}")
    print(f"Spectral pre-processing: Hanning window + FFT  "
          f"N_MODES={N_MODES}  N_IN={N_IN}")
    print(f"seq2seq=True  stitched_metrics=True  "
          f"lambda_phys={lambda_phys}  epsilon={epsilon_w} W  BCE=disabled\n")

    df_tr = data_dict['train']
    df_va = data_dict['val']
    df_te = data_dict['test']

    tr_thr = compute_adaptive_thresholds(df_tr)
    va_thr = compute_adaptive_thresholds(df_va)
    te_thr = compute_adaptive_thresholds(df_te)
    print("  Adaptive thresholds (W):")
    print(f"  {'Appliance':<20} {'Train':>8} {'Val':>8} {'Test':>8}")
    for app in APPLIANCES:
        print(f"  {app:<20} {tr_thr[app]:>8.1f} "
              f"{va_thr[app]:>8.1f} {te_thr[app]:>8.1f}")

    X_tr, Y_tr = create_sequences(df_tr, STRIDE)
    X_va, Y_va = create_sequences(df_va, STRIDE)
    X_te, Y_te = create_sequences(df_te, WIN)   # non-overlapping test windows
    print(f"\n  Train : {X_tr.shape} -> {Y_tr.shape}")
    print(f"  Val   : {X_va.shape} -> {Y_va.shape}")
    print(f"  Test  : {X_te.shape} -> {Y_te.shape}  [non-overlapping]\n")

    # Channel 0 (raw mains): MinMaxScaler — inverse-transformable for physics loss
    x_sc = MinMaxScaler()
    X_tr[:, :, 0] = x_sc.fit_transform(
        X_tr[:, :, 0].reshape(-1, 1)).reshape(X_tr.shape[0], WIN)
    X_va[:, :, 0] = x_sc.transform(
        X_va[:, :, 0].reshape(-1, 1)).reshape(X_va.shape[0], WIN)
    X_te[:, :, 0] = x_sc.transform(
        X_te[:, :, 0].reshape(-1, 1)).reshape(X_te.shape[0], WIN)

    # Channels 1..N_MODES (FFT magnitudes): StandardScaler per mode
    # Values are constant within each window; scaler sees M*WIN samples but
    # between-window variance correctly determines the normalisation scale.
    fft_scalers = []
    for ch in range(1, N_IN):
        fsc = StandardScaler()
        X_tr[:, :, ch] = fsc.fit_transform(
            X_tr[:, :, ch].reshape(-1, 1)).reshape(X_tr.shape[0], WIN)
        X_va[:, :, ch] = fsc.transform(
            X_va[:, :, ch].reshape(-1, 1)).reshape(X_va.shape[0], WIN)
        X_te[:, :, ch] = fsc.transform(
            X_te[:, :, ch].reshape(-1, 1)).reshape(X_te.shape[0], WIN)
        fft_scalers.append(fsc)

    # Target scalers (MinMaxScaler per appliance)
    y_scalers = []
    for i in range(len(APPLIANCES)):
        ys = MinMaxScaler()
        Y_tr[:, :, i] = ys.fit_transform(
            Y_tr[:, :, i].reshape(-1, 1)).reshape(Y_tr[:, :, i].shape)
        Y_va[:, :, i] = ys.transform(
            Y_va[:, :, i].reshape(-1, 1)).reshape(Y_va[:, :, i].shape)
        Y_te[:, :, i] = ys.transform(
            Y_te[:, :, i].reshape(-1, 1)).reshape(Y_te[:, :, i].shape)
        y_scalers.append(ys)

    tr_ld = torch.utils.data.DataLoader(
        Seq2SeqDataset(X_tr, Y_tr), BATCH, shuffle=True)
    va_ld = torch.utils.data.DataLoader(
        Seq2SeqDataset(X_va, Y_va), BATCH, shuffle=False)
    te_ld = torch.utils.data.DataLoader(
        Seq2SeqDataset(X_te, Y_te), BATCH, shuffle=False)

    model     = PINNLNNSeq2Seq(N_IN, hidden_size, len(APPLIANCES), dt).to(device)
    mse_crit  = nn.MSELoss()
    phys_crit = PhysicsConsistencyLoss(x_sc, y_scalers, epsilon_w).to(device)
    opt       = torch.optim.Adam(model.parameters(), lr=LR)
    sched     = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode='min', factor=0.5, patience=8, min_lr=1e-5)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}  "
          f"(input_size increased from 1 to {N_IN} due to spectral channels)\n")

    history = {
        'train_loss': [], 'train_mse': [], 'train_phys': [],
        'val_loss':   [], 'val_mse':   [], 'val_phys':   [],
        'val_metrics': [],
    }
    best_val_mse = float('inf')
    best_state   = None
    counter      = 0

    for epoch in range(EPOCHS):
        model.train()
        ep_mse = ep_phys = ep_tot = 0.0
        pbar = tqdm(tr_ld, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred     = model(xb)
            l_mse    = mse_crit(pred, yb)
            l_phys   = phys_crit(xb, pred)
            l_sparse = pred.mean()
            loss     = l_mse + lambda_phys * l_phys + LAMBDA_SPARSE * l_sparse
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_mse  += l_mse.item()
            ep_phys += l_phys.item()
            ep_tot  += loss.item()
            pbar.set_postfix({'mse':  f'{l_mse.item():.5f}',
                              'phys': f'{l_phys.item():.3f}'})

        nb = len(tr_ld)
        history['train_mse'].append(ep_mse  / nb)
        history['train_phys'].append(ep_phys / nb)
        history['train_loss'].append(ep_tot  / nb)

        model.eval()
        vl_mse = vl_phys = vl_tot = 0.0
        va_preds = []
        with torch.no_grad():
            for xb, yb in va_ld:
                xb, yb = xb.to(device), yb.to(device)
                pred   = model(xb)
                l_mse  = mse_crit(pred, yb)
                l_phys = phys_crit(xb, pred)
                vl_mse  += l_mse.item()
                vl_phys += l_phys.item()
                vl_tot  += (l_mse + lambda_phys * l_phys).item()
                va_preds.append(pred.cpu().numpy())

        nv         = len(va_ld)
        avg_va_mse = vl_mse / nv
        history['val_mse'].append(avg_va_mse)
        history['val_phys'].append(vl_phys / nv)
        history['val_loss'].append(vl_tot  / nv)
        sched.step(avg_va_mse)

        va_pred_all = np.concatenate(va_preds)
        vm, _, _    = compute_metrics(df_va, va_pred_all,
                                      y_scalers, va_thr, STRIDE)
        history['val_metrics'].append(vm)

        avg_f1  = np.mean([vm[a]['f1']  for a in APPLIANCES])
        avg_mae = np.mean([vm[a]['mae'] for a in APPLIANCES])
        print(f"  Epoch {epoch+1:3d}/{EPOCHS}  "
              f"train={history['train_loss'][-1]:.5f} "
              f"(mse={history['train_mse'][-1]:.5f} "
              f"phys={history['train_phys'][-1]:.3f})  "
              f"val_mse={avg_va_mse:.5f}  "
              f"avgF1={avg_f1:.4f}  avgMAE={avg_mae:.2f}  "
              f"lr={opt.param_groups[0]['lr']:.2e}")
        for app in APPLIANCES:
            m = vm[app]
            print(f"    {app:<20}  F1={m['f1']:.4f}  P={m['precision']:.4f}  "
                  f"R={m['recall']:.4f}  MAE={m['mae']:.2f}  "
                  f"TP={m['tp']:,d}  TN={m['tn']:,d}  "
                  f"FP={m['fp']:,d}  FN={m['fn']:,d}")

        if avg_va_mse < best_val_mse:
            best_val_mse = avg_va_mse
            best_state   = {k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

    print("\nTraining complete.")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    te_preds = []
    with torch.no_grad():
        for xb, _ in te_ld:
            te_preds.append(model(xb.to(device)).cpu().numpy())

    te_pred_all = np.concatenate(te_preds)
    test_metrics, stitched_raw, valid_mask = compute_metrics(
        df_te, te_pred_all, y_scalers, te_thr, WIN)

    print(f"\n{'Appliance':<22} {'F1':>7} {'Prec':>7} {'Rec':>7} "
          f"{'MAE':>7} {'TP':>8} {'TN':>8} {'FP':>8} {'FN':>8}")
    print("-" * 90)
    for app in APPLIANCES:
        m = test_metrics[app]
        print(f"{app:<22} {m['f1']:>7.4f} {m['precision']:>7.4f} "
              f"{m['recall']:>7.4f} {m['mae']:>7.2f} "
              f"{m['tp']:>8,d} {m['tn']:>8,d} "
              f"{m['fp']:>8,d} {m['fn']:>8,d}")

    _plot_loss(history, save_dir)
    _plot_metrics(history, test_metrics, save_dir)
    _plot_test_trace(df_te, stitched_raw, valid_mask, te_thr, save_dir)
    _save_results(save_dir, test_metrics,
                  {'train': tr_thr, 'val': va_thr, 'test': te_thr},
                  hidden_size, dt, lambda_phys, epsilon_w)

    return test_metrics, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_loss(history: dict, save_dir: str) -> None:
    ep = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('PINN-LNN (old dataset v4 spectral) — Loss Curves')

    for ax, (tr_k, va_k, title) in zip(axes, [
        ('train_loss', 'val_loss', 'Total Loss'),
        ('train_mse',  'val_mse',  'MSE Loss'),
        ('train_phys', 'val_phys', 'Physics Loss'),
    ]):
        ax.plot(ep, history[tr_k], label='Train', color='steelblue')
        ax.plot(ep, history[va_k], label='Val',   color='tomato')
        ax.set_title(title); ax.set_xlabel('Epoch')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_lnn_old_v4_spectral_loss.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def _plot_metrics(history: dict, test_metrics: dict, save_dir: str) -> None:
    ep  = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(len(APPLIANCES), 2,
                             figsize=(12, 4 * len(APPLIANCES)))
    fig.suptitle('PINN-LNN (old dataset v4 spectral) — Stitched Val Metrics',
                 fontsize=13)

    for row, app in enumerate(APPLIANCES):
        f1s  = [m[app]['f1']  for m in history['val_metrics']]
        maes = [m[app]['mae'] for m in history['val_metrics']]

        axes[row][0].plot(ep, f1s, color='steelblue')
        axes[row][0].axhline(test_metrics[app]['f1'], color='green',
                              linestyle='--', label='Test F1')
        axes[row][0].set_title(f'{app} — F1')
        axes[row][0].legend(fontsize=8); axes[row][0].grid(alpha=0.3)

        axes[row][1].plot(ep, maes, color='tomato')
        axes[row][1].axhline(test_metrics[app]['mae'], color='green',
                              linestyle='--', label='Test MAE')
        axes[row][1].set_title(f'{app} — MAE (W)')
        axes[row][1].legend(fontsize=8); axes[row][1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_lnn_old_v4_spectral_per_appliance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def _plot_test_trace(test_df: pd.DataFrame, stitched_raw: np.ndarray,
                     valid_mask: np.ndarray, thresholds: dict,
                     save_dir: str) -> None:
    x_hours = np.arange(len(test_df), dtype=np.float32) * 6.0 / 3600.0

    fig, axes = plt.subplots(len(APPLIANCES) + 1, 1,
                             figsize=(16, 12), sharex=True)
    fig.suptitle('PINN-LNN (old dataset v4 spectral) — Stitched Test Trace',
                 fontsize=13)

    axes[0].plot(x_hours, test_df[AGG_COL].values,
                 color='steelblue', linewidth=0.8, label='aggregate')
    axes[0].set_ylabel('W'); axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(alpha=0.3)

    for i, app in enumerate(APPLIANCES):
        ax = axes[i + 1]
        ax.plot(x_hours, test_df[app].values,
                color='black', linewidth=0.8, label=f'actual {app}')
        ax.plot(x_hours[valid_mask], stitched_raw[valid_mask, i],
                color='tomato', linestyle='--', linewidth=0.8,
                label=f'predicted {app}')
        ax.axhline(thresholds[app], color='gray', linewidth=0.6,
                   linestyle=':', label=f'thr={thresholds[app]:.0f} W')
        ax.set_title(app); ax.set_ylabel('W')
        ax.legend(loc='upper right', fontsize=7); ax.grid(alpha=0.3)

    axes[-1].set_xlabel('Hours')
    axes[-1].set_xlim(0, 24)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_lnn_old_v4_spectral_test_trace.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def _save_results(save_dir: str, test_metrics: dict,
                  thresholds_all: dict, hidden_size: int, dt: float,
                  lambda_phys: float, epsilon_w: float) -> None:
    cfg = {
        'dataset':     'UKDALE dataset/',
        'model':       'PINNLNNSeq2Seq (old dataset v4 + spectral)',
        'description': ('seq2seq PINN-LNN, Hanning+FFT spectral input channels, '
                        'adaptive thresholds, stitched timeline metrics, BCE disabled'),
        'spectral': {
            'n_modes':     N_MODES,
            'n_in':        N_IN,
            'window_fn':   'Hanning',
            'fft':         'rfft, magnitude, first N_MODES bins',
            'encoding':    'broadcast across WIN axis (global context per window)',
            'scaling':     'StandardScaler per FFT mode',
        },
        'window': {'win': WIN, 'stride_train': STRIDE, 'stride_test': WIN},
        'appliances':  APPLIANCES,
        'thresholds':  thresholds_all,
        'loss':        f'MSE + {lambda_phys} * physics_consistency',
        'model_params': {
            'input_size':  N_IN,
            'hidden_size': hidden_size,
            'n_apps':      len(APPLIANCES),
            'dt':          dt,
        },
        'train_params': {
            'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE, 'batch': BATCH,
            'lambda_phys': lambda_phys, 'epsilon_w': epsilon_w,
            'bce_enabled': False, 'stitched_metrics': True,
        },
        'test_metrics': {
            app: {k: float(v) for k, v in m.items()}
            for app, m in test_metrics.items()
        },
    }
    out = os.path.join(save_dir, 'pinn_lnn_old_v4_spectral_results.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=4)
    print(f"\nResults saved to: {save_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Train v4 spectral PINN-LNN on dataset/ CSV splits.')
    p.add_argument('--dataset-dir', default=DEFAULT_DATASET_DIR,
                   help=f'Directory with UKDALE_HF_*.csv (default: {DEFAULT_DATASET_DIR})')
    p.add_argument('--hidden-size', type=int,   default=64)
    p.add_argument('--dt',          type=float, default=0.1)
    p.add_argument('--lambda-phys', type=float, default=LAMBDA_PHYS)
    p.add_argument('--epsilon-w',   type=float, default=EPSILON_W)
    p.add_argument('--n-modes',     type=int,   default=N_MODES,
                   help='Number of FFT magnitude bins to use as spectral channels '
                        f'(default: {N_MODES})')
    return p.parse_args()


def main():
    args = parse_args()

    global N_MODES, N_IN, _HANN
    if args.n_modes != N_MODES:
        N_MODES = args.n_modes
        N_IN    = 1 + N_MODES
        _HANN   = _make_hanning(WIN)

    missing = [os.path.join(args.dataset_dir, f)
               for f in SPLIT_FILES.values()
               if not os.path.exists(os.path.join(args.dataset_dir, f))]
    if missing:
        print("Missing required CSV files:")
        for path in missing:
            print(f"  {path}")
        sys.exit(1)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir  = f'models/pinn_lnn_old_v4_spectral_{timestamp}'

    data = load_dataset(args.dataset_dir)
    test_metrics, history = train_model(
        data,
        save_dir     = save_dir,
        hidden_size  = args.hidden_size,
        dt           = args.dt,
        lambda_phys  = args.lambda_phys,
        epsilon_w    = args.epsilon_w,
    )

    print(f"Results saved to {save_dir}")
    return test_metrics, history


if __name__ == '__main__':
    main()
