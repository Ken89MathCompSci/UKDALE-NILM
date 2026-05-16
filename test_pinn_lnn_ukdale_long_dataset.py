"""
Physics-Informed LNN (PINN-LNN) for NILM — UKDALE long dataset.

Extends test_pinn_lnn_ukdale_specific_splits.py to use the preprocessed
long_dataset/ CSV files (6-month train, 2-week val/test) from preprocess_hf.py.

Improvements over the original:
  - Loads from long_dataset/UKDALE_HF_{train,validation,test}.csv
  - Adaptive ON/OFF thresholds (p5 of non-zero + delta) with cycling-appliance fix
  - Dual event thresholds: cycling fix applied at eval; raw p5+delta used for BCE targets
  - L1 sparsity loss (LAMBDA_SPARSE) to prevent over-prediction collapse
  - TP/TN/FP/FN reported per appliance
  - CLI args for dataset directory and key hyperparameters

Architecture:
    Input (batch, WIN, 1)  — scaled mains window
         ↓
    Shared AdvancedLiquidTimeLayer encoder (adaptive tau, input-dependent gate)
         ↓
    LayerNorm(hidden)
         ↓
    ┌──────────┬───────┬──────────┬─────────────────┐
    │dishwasher│ fridge│microwave │washing_machine  │  — one Linear head per appliance
    └──────────┴───────┴──────────┴─────────────────┘
    output: (batch, 4) — midpoint prediction per window
"""

import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
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
WIN           = 100
STRIDE        = 5

LAMBDA_PHYS   = 0.01
LAMBDA_SPARSE = 0.005   # L1 sparsity — prevents over-prediction collapse
EPSILON_W     = 50.0    # physics tolerance for unlabelled background loads (W)
WARMUP_EPOCHS = 20      # Stage 1 MSE-only before physics + BCE are added

APPLIANCES = ['dishwasher', 'fridge', 'microwave', 'washing_machine']

# Adaptive threshold hyperparameters
CYCLING_P5_W      = 80.0  # if p5(nonzero) > this → cycling appliance → use THRESHOLD_MIN
THRESHOLD_LOW_PCT = 0.05
THRESHOLD_DELTA   = 20.0  # W above p5
THRESHOLD_MIN     = 10.0

BCE_LAMBDA = {'dishwasher': 0.5, 'fridge': 0.3, 'microwave': 0.5, 'washing_machine': 0.5}
BCE_ALPHA  = {'dishwasher': 2.0, 'fridge': 2.0, 'microwave': 2.0, 'washing_machine': 2.0}

DEFAULT_DATASET_DIR = 'long_dataset'


# ---------------------------------------------------------------------------
# Adaptive threshold helpers
# ---------------------------------------------------------------------------

def compute_adaptive_thresholds(df):
    """Eval thresholds: p5+delta with cycling-appliance override (e.g. fridge → 10 W)."""
    thresholds = {}
    for app in APPLIANCES:
        nz = df[app][df[app] > 0]
        if len(nz) < 10:
            thresholds[app] = THRESHOLD_MIN
            continue
        p5 = float(nz.quantile(THRESHOLD_LOW_PCT))
        if p5 > CYCLING_P5_W:
            thresholds[app] = THRESHOLD_MIN
        else:
            thresholds[app] = max(p5 + THRESHOLD_DELTA, THRESHOLD_MIN)
    return thresholds


def compute_event_thresholds(df):
    """BCE training thresholds: raw p5+delta, no cycling fix."""
    thresholds = {}
    for app in APPLIANCES:
        nz = df[app][df[app] > 0]
        if len(nz) < 10:
            thresholds[app] = THRESHOLD_MIN
            continue
        p5 = float(nz.quantile(THRESHOLD_LOW_PCT))
        thresholds[app] = max(p5 + THRESHOLD_DELTA, THRESHOLD_MIN)
    return thresholds


# ---------------------------------------------------------------------------
# Physics Consistency Loss
# ---------------------------------------------------------------------------

class PhysicsConsistencyLoss(nn.Module):
    """
    Soft one-sided penalty: ReLU(Σ p_hat_i_raw − P_agg_raw − ε)

    Inverse-scaling is differentiable (linear), so gradients flow back to the model.
    """

    def __init__(self, x_scaler, y_scalers, appliances, epsilon_w=EPSILON_W):
        super().__init__()
        self.epsilon = epsilon_w

        x_min   = float(x_scaler.data_min_[0])
        x_range = float(x_scaler.data_range_[0])
        self.register_buffer('x_min',   torch.tensor(x_min,   dtype=torch.float32))
        self.register_buffer('x_range', torch.tensor(x_range, dtype=torch.float32))

        y_mins   = [float(y_scalers[i].data_min_[0])   for i in range(len(appliances))]
        y_ranges = [float(y_scalers[i].data_range_[0]) for i in range(len(appliances))]
        self.register_buffer('y_mins',   torch.tensor(y_mins,   dtype=torch.float32))
        self.register_buffer('y_ranges', torch.tensor(y_ranges, dtype=torch.float32))

    def forward(self, x_mid_scaled, pred_scaled):
        """
        Args:
            x_mid_scaled: (batch,)        — scaled mains at window midpoint
            pred_scaled:  (batch, n_apps) — scaled appliance predictions
        """
        x_raw     = x_mid_scaled * self.x_range + self.x_min   # (batch,)
        p_raw     = pred_scaled  * self.y_ranges + self.y_mins  # (batch, n_apps)
        p_sum     = p_raw.sum(dim=1)                            # (batch,)
        violation = F.relu(p_sum - x_raw - self.epsilon)
        return violation.mean()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PhysicsInformedLiquidNetworkModel(nn.Module):
    """Shared LiquidCell encoder → per-appliance linear heads."""

    def __init__(self, input_size, hidden_size, n_appliances, dt=0.1):
        super().__init__()
        self.hidden_size  = hidden_size
        self.n_appliances = n_appliances
        self.dt           = dt

        self.input_proj  = nn.Linear(input_size, hidden_size)
        self.tau_base    = nn.Parameter(torch.ones(hidden_size))
        self.tau_mod     = nn.Linear(input_size, hidden_size)
        self.rec_weights = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.rec_weights)
        self.gate        = nn.Linear(input_size + hidden_size, hidden_size)
        self.norm        = nn.LayerNorm(hidden_size)
        self.heads       = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(n_appliances)
        ])

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            (batch, n_appliances)
        """
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        for t in range(seq_len):
            x_t        = x[:, t, :]
            input_proj = self.input_proj(x_t)
            rec_proj   = torch.matmul(h, self.rec_weights)
            tau_base   = F.softplus(self.tau_base).unsqueeze(0)
            tau_mod    = torch.sigmoid(self.tau_mod(x_t))
            tau        = (tau_base * tau_mod).clamp(min=self.dt)
            gate       = torch.sigmoid(self.gate(torch.cat([x_t, h], dim=1)))
            f_t        = torch.tanh(input_proj + rec_proj)
            dh         = ((-h / tau) + gate * f_t) * self.dt
            h          = (h + dh).clamp(-10.0, 10.0)

        h = self.norm(h)
        return torch.cat([head(h) for head in self.heads], dim=1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultiApplianceDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


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


def create_sequences(df, stride=STRIDE):
    """Midpoint targeting — y[i] is appliance values at the window centre."""
    mains    = df['aggregate'].values.astype(np.float32)
    app_vals = {app: df[app].values.astype(np.float32) for app in APPLIANCES}
    X, Y = [], []
    for i in range(0, len(mains) - WIN + 1, stride):
        X.append(mains[i: i + WIN])
        mid = i + WIN // 2
        Y.append([app_vals[app][mid] for app in APPLIANCES])
    return (
        np.array(X, dtype=np.float32).reshape(-1, WIN, 1),
        np.array(Y, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_per_appliance_metrics(y_true, y_pred, y_scalers, thresholds):
    """Returns dict {app: metrics_dict} including TP/TN/FP/FN."""
    results = {}
    for i, app in enumerate(APPLIANCES):
        raw_true = y_scalers[i].inverse_transform(y_true[:, i:i+1]).flatten()
        raw_pred = y_scalers[i].inverse_transform(y_pred[:, i:i+1]).flatten()
        thr      = thresholds[app]
        m        = calculate_nilm_metrics(raw_true, raw_pred, threshold=thr)
        true_bin = (raw_true > thr)
        pred_bin = (raw_pred > thr)
        m['TP']  = int(( true_bin &  pred_bin).sum())
        m['TN']  = int((~true_bin & ~pred_bin).sum())
        m['FP']  = int((~true_bin &  pred_bin).sum())
        m['FN']  = int(( true_bin & ~pred_bin).sum())
        results[app] = m
    return results


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(data_splits, save_dir,
                hidden_size=64, dt=0.1,
                lambda_phys=LAMBDA_PHYS, epsilon_w=EPSILON_W):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}  λ_phys={lambda_phys}  λ_sparse={LAMBDA_SPARSE}  "
          f"ε={epsilon_w}W  hidden={hidden_size}  dt={dt}")

    tr_df = data_splits['train']
    va_df = data_splits['validation']
    te_df = data_splits['test']

    # Adaptive thresholds derived from training distribution
    eval_thr  = compute_adaptive_thresholds(tr_df)
    event_thr = compute_event_thresholds(tr_df)
    print("\nEval thresholds  (cycling fix):", eval_thr)
    print("Event thresholds (BCE targets) :", event_thr)

    X_tr, Y_tr = create_sequences(tr_df)
    X_va, Y_va = create_sequences(va_df)
    X_te, Y_te = create_sequences(te_df)

    x_scaler = MinMaxScaler()
    X_tr = x_scaler.fit_transform(X_tr.reshape(-1, 1)).reshape(X_tr.shape)
    X_va = x_scaler.transform(X_va.reshape(-1, 1)).reshape(X_va.shape)
    X_te = x_scaler.transform(X_te.reshape(-1, 1)).reshape(X_te.shape)

    y_scalers = []
    for i in range(len(APPLIANCES)):
        ys = MinMaxScaler()
        Y_tr[:, i:i+1] = ys.fit_transform(Y_tr[:, i:i+1])
        Y_va[:, i:i+1] = ys.transform(Y_va[:, i:i+1])
        Y_te[:, i:i+1] = ys.transform(Y_te[:, i:i+1])
        y_scalers.append(ys)

    # Scaled event thresholds for BCE binary targets
    event_thr_scaled = [
        (event_thr[app] - float(y_scalers[i].data_min_[0]))
        / float(y_scalers[i].data_range_[0])
        for i, app in enumerate(APPLIANCES)
    ]

    print(f"\nTrain: {X_tr.shape} → {Y_tr.shape}")
    print(f"Val:   {X_va.shape} → {Y_va.shape}")
    print(f"Test:  {X_te.shape} → {Y_te.shape}")

    tr_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_tr, Y_tr), batch_size=BATCH, shuffle=True,  drop_last=False)
    va_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_va, Y_va), batch_size=BATCH, shuffle=False, drop_last=False)
    te_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_te, Y_te), batch_size=BATCH, shuffle=False, drop_last=False)

    model = PhysicsInformedLiquidNetworkModel(
        input_size=1, hidden_size=hidden_size,
        n_appliances=len(APPLIANCES), dt=dt,
    ).to(device)

    mse_crit  = nn.MSELoss()
    phys_crit = PhysicsConsistencyLoss(
        x_scaler, y_scalers, APPLIANCES, epsilon_w=epsilon_w
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-5)

    history = {
        'train_loss': [], 'train_mse': [], 'train_phys': [],
        'val_loss':   [], 'val_mse':   [], 'val_phys':   [],
        'val_metrics': [],
    }
    best_val_loss = float('inf')
    best_state    = None
    counter       = 0

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print("Starting training...")

    for epoch in range(EPOCHS):
        model.train()
        ep_mse = ep_phys = ep_total = 0.0
        bar = tqdm(tr_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        for xb, yb in bar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            pred     = model(xb)                     # (batch, n_apps)
            l_mse    = mse_crit(pred, yb)
            x_mid    = xb[:, WIN // 2, 0]            # aggregate at midpoint
            l_phys   = phys_crit(x_mid, pred)
            l_sparse = pred.mean()

            if epoch < WARMUP_EPOCHS:
                loss = l_mse + LAMBDA_SPARSE * l_sparse
            else:
                l_bce = torch.tensor(0.0, device=device)
                for i, app in enumerate(APPLIANCES):
                    if BCE_LAMBDA[app] > 0:
                        pred_i = pred[:, i].clamp(1e-7, 1 - 1e-7)
                        thr_s  = event_thr_scaled[i]
                        y_bin  = (yb[:, i] > thr_s).float()
                        w      = torch.where(y_bin == 1,
                                             torch.full_like(y_bin, BCE_ALPHA[app]),
                                             torch.ones_like(y_bin))
                        l_bce  = l_bce + BCE_LAMBDA[app] * F.binary_cross_entropy(
                            pred_i, y_bin, weight=w)
                loss = l_mse + lambda_phys * l_phys + LAMBDA_SPARSE * l_sparse + l_bce

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ep_mse   += l_mse.item()
            ep_phys  += l_phys.item()
            ep_total += loss.item()
            bar.set_postfix(mse=f'{l_mse.item():.5f}', phys=f'{l_phys.item():.5f}')

        avg_tr_mse   = ep_mse   / len(tr_loader)
        avg_tr_phys  = ep_phys  / len(tr_loader)
        avg_tr_total = ep_total / len(tr_loader)
        history['train_mse'].append(avg_tr_mse)
        history['train_phys'].append(avg_tr_phys)
        history['train_loss'].append(avg_tr_total)

        # Validation
        model.eval()
        vl_mse = vl_phys = vl_total = 0.0
        val_preds, val_trues = [], []

        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred   = model(xb)
                l_mse  = mse_crit(pred, yb)
                x_mid  = xb[:, WIN // 2, 0]
                l_phys = phys_crit(x_mid, pred)
                loss   = l_mse + lambda_phys * l_phys
                vl_mse   += l_mse.item()
                vl_phys  += l_phys.item()
                vl_total += loss.item()
                val_preds.append(pred.cpu().numpy())
                val_trues.append(yb.cpu().numpy())

        avg_va_mse   = vl_mse   / len(va_loader)
        avg_va_phys  = vl_phys  / len(va_loader)
        avg_va_total = vl_total / len(va_loader)
        history['val_mse'].append(avg_va_mse)
        history['val_phys'].append(avg_va_phys)
        history['val_loss'].append(avg_va_total)

        scheduler.step(avg_va_mse)

        y_pred_all = np.concatenate(val_preds)
        y_true_all = np.concatenate(val_trues)
        per_app_m  = compute_per_appliance_metrics(
            y_true_all, y_pred_all, y_scalers, eval_thr)
        history['val_metrics'].append(per_app_m)

        avg_f1  = np.mean([per_app_m[a]['f1']  for a in APPLIANCES])
        avg_mae = np.mean([per_app_m[a]['mae'] for a in APPLIANCES])

        print(
            f"  Epoch {epoch+1:3d}/{EPOCHS}  "
            f"train={avg_tr_total:.5f} (mse={avg_tr_mse:.5f} phys={avg_tr_phys:.5f})  "
            f"val={avg_va_total:.5f} (mse={avg_va_mse:.5f} phys={avg_va_phys:.5f})  "
            f"avgF1={avg_f1:.4f}  avgMAE={avg_mae:.2f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        for app in APPLIANCES:
            m = per_app_m[app]
            print(f"    {app:<22s}  F1={m['f1']:.4f}  "
                  f"P={m['precision']:.4f}  R={m['recall']:.4f}  "
                  f"MAE={m['mae']:.2f}  TP={m['TP']}  FP={m['FP']}  FN={m['FN']}")

        if avg_va_mse < best_val_loss:
            best_val_loss = avg_va_mse
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            counter       = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print("Training complete.")

    # Test evaluation
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    test_preds, test_trues = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            test_preds.append(model(xb.to(device)).cpu().numpy())
            test_trues.append(yb.cpu().numpy())

    y_pred_te = np.concatenate(test_preds)
    y_true_te = np.concatenate(test_trues)

    test_metrics = compute_per_appliance_metrics(
        y_true_te, y_pred_te, y_scalers, eval_thr)

    print(f"\n{'Appliance':<22} {'F1':>6} {'Prec':>6} {'Rec':>6} "
          f"{'MAE':>7} {'SAE':>7} {'TP':>7} {'FP':>7} {'FN':>7}")
    print("-" * 80)
    for app in APPLIANCES:
        m = test_metrics[app]
        print(f"{app:<22} {m['f1']:>6.4f} {m['precision']:>6.4f} {m['recall']:>6.4f} "
              f"{m['mae']:>7.2f} {m['sae']:>7.4f} {m['TP']:>7} {m['FP']:>7} {m['FN']:>7}")

    _plot_results(history, test_metrics, save_dir)
    _save_json(test_metrics, hidden_size, dt, lambda_phys, epsilon_w, save_dir)

    return test_metrics, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_results(history, test_metrics, save_dir):
    epochs_x = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs_x, history['train_loss'], label='Train total', color='blue')
    plt.plot(epochs_x, history['val_loss'],   label='Val total',   color='red')
    plt.title('Total Loss (MSE + λ·Phys + λ·Sparse)')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(epochs_x, history['train_mse'], label='Train MSE', color='blue')
    plt.plot(epochs_x, history['val_mse'],   label='Val MSE',   color='red')
    plt.title('MSE Loss')
    plt.xlabel('Epoch'); plt.ylabel('MSE')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(epochs_x, history['train_phys'], label='Train Phys', color='blue')
    plt.plot(epochs_x, history['val_phys'],   label='Val Phys',   color='red')
    plt.title('Physics Consistency Loss')
    plt.xlabel('Epoch'); plt.ylabel('L_phys')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(len(APPLIANCES), 2, figsize=(12, 4 * len(APPLIANCES)))
    fig.suptitle('PINN-LNN Long Dataset — Per-Appliance Val Metrics', fontsize=13)

    for row, app in enumerate(APPLIANCES):
        f1_series  = [m[app]['f1']  for m in history['val_metrics']]
        mae_series = [m[app]['mae'] for m in history['val_metrics']]

        axes[row][0].plot(epochs_x, f1_series, color='blue', linewidth=1.5)
        axes[row][0].axhline(test_metrics[app]['f1'], color='green',
                             linestyle='--', label='Test F1')
        axes[row][0].set_title(f'{app} — F1')
        axes[row][0].set_xlabel('Epoch'); axes[row][0].set_ylabel('F1')
        axes[row][0].legend(); axes[row][0].grid(True, alpha=0.3)

        axes[row][1].plot(epochs_x, mae_series, color='red', linewidth=1.5)
        axes[row][1].axhline(test_metrics[app]['mae'], color='green',
                             linestyle='--', label='Test MAE')
        axes[row][1].set_title(f'{app} — MAE (W)')
        axes[row][1].set_xlabel('Epoch'); axes[row][1].set_ylabel('MAE (W)')
        axes[row][1].legend(); axes[row][1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_appliance_metrics.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------

def _save_json(test_metrics, hidden_size, dt, lambda_phys, epsilon_w, save_dir):
    config = {
        'dataset': 'UKDALE-long_dataset',
        'model': 'PhysicsInformedLiquidNetworkModel',
        'hyperparams': {
            'WIN': WIN, 'STRIDE': STRIDE, 'BATCH': BATCH,
            'EPOCHS': EPOCHS, 'PATIENCE': PATIENCE, 'LR': LR,
            'hidden_size': hidden_size, 'dt': dt,
            'lambda_phys': lambda_phys, 'lambda_sparse': LAMBDA_SPARSE,
            'epsilon_w': epsilon_w, 'warmup_epochs': WARMUP_EPOCHS,
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
    parser = argparse.ArgumentParser(
        description='PINN-LNN NILM on UKDALE long_dataset CSVs')
    parser.add_argument('--dataset-dir', default=DEFAULT_DATASET_DIR,
                        help='Directory containing UKDALE_HF_*.csv files')
    parser.add_argument('--hidden-size', type=int,   default=64)
    parser.add_argument('--dt',          type=float, default=0.1)
    parser.add_argument('--lambda-phys', type=float, default=LAMBDA_PHYS)
    parser.add_argument('--epsilon-w',   type=float, default=EPSILON_W)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    for split in ('train', 'validation', 'test'):
        p = os.path.join(args.dataset_dir, f'UKDALE_HF_{split}.csv')
        if not os.path.exists(p):
            print(f"Error: {p} not found. Run preprocess_hf.py first.")
            sys.exit(1)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir  = f'models/pinn_lnn_long_{timestamp}'

    data_splits = load_data(args.dataset_dir)

    train_model(
        data_splits,
        save_dir    = save_dir,
        hidden_size = args.hidden_size,
        dt          = args.dt,
        lambda_phys = args.lambda_phys,
        epsilon_w   = args.epsilon_w,
    )
    print(f"\nAll outputs saved to {save_dir}/")
