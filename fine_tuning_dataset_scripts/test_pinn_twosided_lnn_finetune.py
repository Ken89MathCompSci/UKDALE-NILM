"""
Physics-Informed Basic LNN — Two-Sided Physical Loss — Cross-House Fine-Tuning.

Extends test_pinn_basic_lnn_finetune.py with a two-sided energy-consistency
loss that penalises both directions of physically invalid predictions:

    Upper (over-prediction):
        L_over  = E[ relu( Σpred_raw − P_agg_raw − EPSILON_OVER ) ]
        Enforces energy conservation: appliances cannot collectively draw
        more power than the measured aggregate mains (plus tolerance).
        This is identical to the one-sided loss in the base script.

    Lower (under-prediction floor):
        L_under = E[ relu( P_agg_raw × ALPHA_FLOOR − Σpred_raw ) ]
        If predictions sum to less than ALPHA_FLOOR × aggregate, the model
        is ignoring too much real load.  This directly counters always-OFF
        collapse: a policy that predicts zero for everything gets a large
        under-prediction penalty whenever the house is actively consuming
        power.  ALPHA_FLOOR is deliberately conservative (default 0.15) to
        account for unmonitored loads (lights, TV, EV chargers, etc.) that
        are included in the aggregate but not in our four target appliances.

    Combined:
        L_phys = L_over + LAMBDA_UNDER × L_under

Both components are logged separately each epoch so their individual
contributions can be monitored and tuned independently.

Three-phase training strategy (unchanged from base script):
    Phase 1 — Pretrain : House 1 data, validate on House 1 validation set.
    Phase 2 — Fine-tune: House 5 first 2 h (finetune.csv), low LR.
    Phase 3 — Test     : House 5 remaining hours (test.csv).

Dataset: fine_tuning_dataset/
    UKDALE_HF_pretrain.csv    — House 1, Nov-09 + Dec-07 (28 800 rows)
    UKDALE_HF_validation.csv  — House 1, Dec-07           (14 400 rows)
    UKDALE_HF_finetune.csv    — House 5, Aug-24 first 2 h  (1 200 rows)
    UKDALE_HF_test.csv        — House 5, Aug-24 remaining  (13 200 rows)
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Source Code'))
from utils import calculate_nilm_metrics


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(__file__),
                                   '..', 'fine_tuning_dataset')

# Pretrain phase
EPOCHS        = 80
PATIENCE      = 20
LR            = 1e-3
WARMUP_EPOCHS = 20      # Stage 1: MSE-only

# Fine-tune phase
EPOCHS_FT     = 30
PATIENCE_FT   = 10
LR_FT         = 1e-4

BATCH         = 32
WIN           = 100
STRIDE        = 5

# Two-sided physics loss parameters
LAMBDA_PHYS   = 0.01   # weight of combined physics term in total loss
EPSILON_OVER  = 50.0   # W — tolerance above aggregate for over-prediction
ALPHA_FLOOR   = 0.15   # fraction of aggregate that predictions must at least reach
LAMBDA_UNDER  = 0.5    # relative weight of under-prediction vs over-prediction

APPLIANCES = ['dishwasher', 'fridge', 'microwave', 'washing_machine']

THRESHOLDS = {
    'dishwasher':       10.0,
    'fridge':           10.0,
    'microwave':        10.0,
    'washing_machine':  10.0,
}

BCE_LAMBDA = {'dishwasher': 0.5, 'fridge': 0.3, 'microwave': 0.0, 'washing_machine': 0.0}
BCE_ALPHA  = {'dishwasher': 2.0, 'fridge': 2.0, 'microwave': 1.0, 'washing_machine': 1.0}


# ---------------------------------------------------------------------------
# Two-Sided Physics Consistency Loss
# ---------------------------------------------------------------------------

class TwoSidedPhysicsLoss(nn.Module):
    """
    Two-sided energy-consistency penalty.

    Upper side — energy conservation (identical to base script):
        L_over = relu( Σpred_raw − P_agg_raw − epsilon_over )
        Prevents the model from predicting more power than the mains supplies.

    Lower side — energy floor (new):
        L_under = relu( P_agg_raw × alpha_floor − Σpred_raw )
        Prevents always-OFF collapse by requiring predictions to account for
        at least alpha_floor fraction of the measured aggregate.

    Combined loss returned per batch:
        L_phys = mean( L_over + lambda_under × L_under )

    Both components are also returned separately for logging.

    Parameters
    ----------
    x_scaler     : fitted MinMaxScaler for aggregate mains
    y_scalers    : list of fitted MinMaxScalers for each appliance
    appliances   : list of appliance names (determines ordering)
    epsilon_over : W — over-prediction tolerance (default 50 W)
    alpha_floor  : fraction of aggregate for under-prediction floor (default 0.15)
    lambda_under : relative weight of under-prediction penalty (default 0.5)
    """

    def __init__(self, x_scaler, y_scalers, appliances,
                 epsilon_over=EPSILON_OVER,
                 alpha_floor=ALPHA_FLOOR,
                 lambda_under=LAMBDA_UNDER):
        super().__init__()
        self.epsilon_over = epsilon_over
        self.alpha_floor  = alpha_floor
        self.lambda_under = lambda_under

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
        Parameters
        ----------
        x_mid_scaled : (B,) scaled aggregate at window midpoint
        pred_scaled  : (B, n_apps) scaled appliance predictions

        Returns
        -------
        total  : scalar combined loss
        over   : scalar over-prediction component (unweighted)
        under  : scalar under-prediction component (unweighted)
        """
        x_raw    = x_mid_scaled * self.x_range + self.x_min      # (B,) Watts
        p_raw    = pred_scaled  * self.y_ranges + self.y_mins     # (B, n_apps) Watts
        sum_pred = p_raw.sum(dim=1)                               # (B,)

        over  = F.relu(sum_pred - x_raw - self.epsilon_over)
        under = F.relu(x_raw * self.alpha_floor - sum_pred)

        over_mean  = over.mean()
        under_mean = under.mean()
        total      = over_mean + self.lambda_under * under_mean
        return total, over_mean, under_mean


# ---------------------------------------------------------------------------
# Model  (identical to base script)
# ---------------------------------------------------------------------------

class PhysicsInformedBasicLiquidNetworkModel(nn.Module):
    """
    Basic LNN cell (fixed tau, no input-dependent gate):
        tau  = softplus(tau_param)
        f_t  = tanh(LayerNorm(W_in*x_t + W_rec*h))
        dh   = (-h/tau + f_t) * dt
        h    = clamp(h + dh, -10, 10)
    """

    def __init__(self, input_size, hidden_size, n_appliances, dt=0.1):
        super().__init__()
        self.hidden_size  = hidden_size
        self.n_appliances = n_appliances
        self.dt           = dt

        self.input_proj  = nn.Linear(input_size, hidden_size)
        self.tau         = nn.Parameter(torch.ones(hidden_size))
        self.rec_weights = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.rec_weights)
        self.intra_norm  = nn.LayerNorm(hidden_size)
        self.norm        = nn.LayerNorm(hidden_size)
        self.heads       = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(n_appliances)
        ])

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h   = torch.zeros(batch_size, self.hidden_size, device=x.device)
        tau = F.softplus(self.tau).unsqueeze(0)
        for t in range(seq_len):
            x_t = x[:, t, :]
            f_t = torch.tanh(self.intra_norm(
                self.input_proj(x_t) + torch.matmul(h, self.rec_weights)))
            dh  = (-h / tau + f_t) * self.dt
            h   = (h + dh).clamp(-10.0, 10.0)
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

def load_split(dataset_dir: str, name: str) -> pd.DataFrame:
    path = os.path.join(dataset_dir, f'UKDALE_HF_{name}.csv')
    df   = pd.read_csv(path)
    print(f"  {name:12s}: {len(df):6,} rows  "
          f"[{df['timestamp'].iloc[0][:19]}  to  {df['timestamp'].iloc[-1][:19]}]")
    return df


def create_sequences(df: pd.DataFrame, window_size: int = WIN, stride: int = STRIDE):
    mains    = df['aggregate'].values
    app_vals = {app: df[app].values for app in APPLIANCES}
    X, Y = [], []
    for i in range(0, len(mains) - window_size, stride):
        X.append(mains[i:i + window_size])
        mid = i + window_size // 2
        Y.append([app_vals[app][mid] for app in APPLIANCES])
    return (
        np.array(X, dtype=np.float32).reshape(-1, window_size, 1),
        np.array(Y, dtype=np.float32),
    )


def compute_per_appliance_metrics(y_true, y_pred, y_scalers):
    """Inverse-scale and compute full metrics including TP/FP/TN/FN."""
    metrics = {}
    for i, app in enumerate(APPLIANCES):
        raw_true = y_scalers[i].inverse_transform(y_true[:, i:i+1]).flatten()
        raw_pred = y_scalers[i].inverse_transform(y_pred[:, i:i+1]).flatten()
        thr = THRESHOLDS[app]
        m   = calculate_nilm_metrics(raw_true, raw_pred, threshold=thr)
        # Ensure TP/FP/TN/FN present even if calculate_nilm_metrics omits them
        m['TP'] = int(m.get('TP', ((raw_true > thr) & (raw_pred > thr)).sum()))
        m['FP'] = int(m.get('FP', ((raw_true <= thr) & (raw_pred > thr)).sum()))
        m['TN'] = int(m.get('TN', ((raw_true <= thr) & (raw_pred <= thr)).sum()))
        m['FN'] = int(m.get('FN', ((raw_true > thr) & (raw_pred <= thr)).sum()))
        metrics[app] = m
    return metrics


# ---------------------------------------------------------------------------
# One-epoch helpers
# ---------------------------------------------------------------------------

def _train_epoch(model, loader, optimizer, mse_crit, phys_crit,
                 thresholds_scaled, device, use_phys_bce: bool):
    """
    Returns
    -------
    ep_total, ep_mse, ep_over, ep_under : per-batch averages
    """
    model.train()
    ep_mse = ep_over = ep_under = ep_total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred     = model(xb)
        mse_loss = mse_crit(pred, yb)
        x_mid    = xb[:, WIN // 2, 0]
        phys_total, phys_over, phys_under = phys_crit(x_mid, pred)

        if not use_phys_bce:
            loss = mse_loss
        else:
            bce_loss = torch.tensor(0.0, device=device)
            for i, app in enumerate(APPLIANCES):
                if BCE_LAMBDA[app] > 0:
                    pred_i = pred[:, i].clamp(1e-7, 1 - 1e-7)
                    y_bin  = (yb[:, i] > thresholds_scaled[i]).float()
                    w      = torch.where(y_bin == 1,
                                         torch.full_like(y_bin, BCE_ALPHA[app]),
                                         torch.ones_like(y_bin))
                    bce_loss = bce_loss + BCE_LAMBDA[app] * F.binary_cross_entropy(
                        pred_i, y_bin, weight=w)
            loss = mse_loss + LAMBDA_PHYS * phys_total + bce_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        ep_mse   += mse_loss.item()
        ep_over  += phys_over.item()
        ep_under += phys_under.item()
        ep_total += loss.item()

    n = len(loader)
    return ep_total / n, ep_mse / n, ep_over / n, ep_under / n


def _val_epoch(model, loader, mse_crit, phys_crit, device):
    """
    Returns
    -------
    vl_total, vl_mse, vl_over, vl_under, preds_np, trues_np
    """
    model.eval()
    vl_mse = vl_over = vl_under = vl_total = 0.0
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred   = model(xb)
            mse_loss = mse_crit(pred, yb)
            phys_total, phys_over, phys_under = phys_crit(xb[:, WIN // 2, 0], pred)
            loss     = mse_loss + LAMBDA_PHYS * phys_total
            vl_mse   += mse_loss.item()
            vl_over  += phys_over.item()
            vl_under += phys_under.item()
            vl_total += loss.item()
            preds.append(pred.cpu().numpy())
            trues.append(yb.cpu().numpy())
    n = len(loader)
    return (vl_total / n, vl_mse / n, vl_over / n, vl_under / n,
            np.concatenate(preds), np.concatenate(trues))


# ---------------------------------------------------------------------------
# Phase 1 — Pretrain
# ---------------------------------------------------------------------------

def pretrain(model, tr_loader, va_loader, phys_crit, device, y_scalers,
             thresholds_scaled):
    mse_crit  = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-5)

    history = {
        'train_loss': [], 'train_mse': [], 'train_over': [], 'train_under': [],
        'val_loss':   [], 'val_mse':   [], 'val_over':   [], 'val_under':   [],
        'val_metrics': [],
    }
    best_val_mse = float('inf')
    best_state   = None
    counter      = 0

    print("\n=== Phase 1: Pretrain (House 1) ===")
    print(f"  Two-sided physics: ε_over={EPSILON_OVER}W  α_floor={ALPHA_FLOOR}  "
          f"λ_under={LAMBDA_UNDER}  λ_phys={LAMBDA_PHYS}")

    for epoch in range(EPOCHS):
        use_full_loss = (epoch >= WARMUP_EPOCHS)
        tr_tot, tr_mse, tr_over, tr_under = _train_epoch(
            model, tr_loader, optimizer, mse_crit, phys_crit,
            thresholds_scaled, device, use_phys_bce=use_full_loss)

        va_tot, va_mse, va_over, va_under, y_pred_va, y_true_va = _val_epoch(
            model, va_loader, mse_crit, phys_crit, device)
        scheduler.step(va_mse)

        history['train_loss'].append(tr_tot);   history['train_mse'].append(tr_mse)
        history['train_over'].append(tr_over);  history['train_under'].append(tr_under)
        history['val_loss'].append(va_tot);     history['val_mse'].append(va_mse)
        history['val_over'].append(va_over);    history['val_under'].append(va_under)

        m = compute_per_appliance_metrics(y_true_va, y_pred_va, y_scalers)
        history['val_metrics'].append(m)
        avg_f1 = np.mean([m[a]['f1'] for a in APPLIANCES])

        print(f"  Epoch {epoch+1:3d}/{EPOCHS}  "
              f"train={tr_tot:.5f} (mse={tr_mse:.5f} over={tr_over:.5f} under={tr_under:.5f})  "
              f"val={va_tot:.5f} (mse={va_mse:.5f} over={va_over:.5f} under={va_under:.5f})  "
              f"avgF1={avg_f1:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}"
              + ("  [warmup]" if not use_full_loss else ""))
        for app in APPLIANCES:
            ma = m[app]
            print(f"    {app:<20}  F1={ma['f1']:.4f}  P={ma['precision']:.4f}  "
                  f"R={ma['recall']:.4f}  MAE={ma['mae']:.2f}  SAE={ma['sae']:.4f}  "
                  f"TP={ma['TP']}  FP={ma['FP']}  TN={ma['TN']}  FN={ma['FN']}")

        if va_mse < best_val_mse:
            best_val_mse = va_mse
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            counter      = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return best_state, history


# ---------------------------------------------------------------------------
# Phase 2 — Fine-tune
# ---------------------------------------------------------------------------

def finetune(model, ft_loader, va_loader, phys_crit, device, y_scalers,
             thresholds_scaled):
    mse_crit  = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_FT)

    ft_history = {'train_loss': [], 'train_mse': [], 'train_over': [], 'train_under': []}
    best_ft_loss = float('inf')
    best_state   = None
    counter      = 0

    print(f"\n=== Phase 2: Fine-tune (House 5, {len(ft_loader.dataset)} windows) ===")
    print(f"  LR={LR_FT}  max_epochs={EPOCHS_FT}  patience={PATIENCE_FT}")

    for epoch in range(EPOCHS_FT):
        tr_tot, tr_mse, tr_over, tr_under = _train_epoch(
            model, ft_loader, optimizer, mse_crit, phys_crit,
            thresholds_scaled, device, use_phys_bce=True)

        ft_history['train_loss'].append(tr_tot)
        ft_history['train_mse'].append(tr_mse)
        ft_history['train_over'].append(tr_over)
        ft_history['train_under'].append(tr_under)

        print(f"  FT Epoch {epoch+1:2d}/{EPOCHS_FT}  "
              f"loss={tr_tot:.5f}  mse={tr_mse:.5f}  "
              f"over={tr_over:.5f}  under={tr_under:.5f}")

        if tr_mse < best_ft_loss:
            best_ft_loss = tr_mse
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            counter      = 0
        else:
            counter += 1
            if counter >= PATIENCE_FT:
                print(f"  Early stopping at fine-tune epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return ft_history


# ---------------------------------------------------------------------------
# Phase 3 — Test
# ---------------------------------------------------------------------------

def evaluate_test(model, te_loader, device, y_scalers):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            preds.append(model(xb.to(device)).cpu().numpy())
            trues.append(yb.cpu().numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(trues)
    return compute_per_appliance_metrics(y_true, y_pred, y_scalers), y_pred, y_true


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def run(dataset_dir=DEFAULT_DATASET_DIR, hidden_size=64, dt=0.1,
        epsilon_over=EPSILON_OVER, alpha_floor=ALPHA_FLOOR,
        lambda_under=LAMBDA_UNDER, save_dir=None):
    if save_dir is None:
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"models/pinn_twosided_lnn_finetune_{ts}"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Dataset: {dataset_dir}")

    print("\nLoading splits...")
    df_pre  = load_split(dataset_dir, 'pretrain')
    df_val  = load_split(dataset_dir, 'validation')
    df_ft   = load_split(dataset_dir, 'finetune')
    df_te   = load_split(dataset_dir, 'test')

    X_pre, Y_pre = create_sequences(df_pre)
    X_val, Y_val = create_sequences(df_val)
    X_ft,  Y_ft  = create_sequences(df_ft)
    X_te,  Y_te  = create_sequences(df_te)
    print(f"\n  pretrain  X={X_pre.shape}  Y={Y_pre.shape}")
    print(f"  val       X={X_val.shape}  Y={Y_val.shape}")
    print(f"  finetune  X={X_ft.shape}   Y={Y_ft.shape}")
    print(f"  test      X={X_te.shape}   Y={Y_te.shape}")

    x_scaler = MinMaxScaler()
    X_pre = x_scaler.fit_transform(X_pre.reshape(-1, 1)).reshape(X_pre.shape)
    X_val = x_scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
    X_ft  = x_scaler.transform(X_ft.reshape(-1, 1)).reshape(X_ft.shape)
    X_te  = x_scaler.transform(X_te.reshape(-1, 1)).reshape(X_te.shape)

    y_scalers = []
    for i in range(len(APPLIANCES)):
        ys = MinMaxScaler()
        Y_pre[:, i:i+1] = ys.fit_transform(Y_pre[:, i:i+1])
        Y_val[:, i:i+1] = ys.transform(Y_val[:, i:i+1])
        Y_ft[:, i:i+1]  = ys.transform(Y_ft[:, i:i+1])
        Y_te[:, i:i+1]  = ys.transform(Y_te[:, i:i+1])
        y_scalers.append(ys)

    thresholds_scaled = [
        (THRESHOLDS[app] - float(y_scalers[i].data_min_[0]))
        / float(y_scalers[i].data_range_[0])
        for i, app in enumerate(APPLIANCES)
    ]

    pre_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_pre, Y_pre), batch_size=BATCH, shuffle=True,  drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_val, Y_val), batch_size=BATCH, shuffle=False, drop_last=False)
    ft_loader  = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_ft,  Y_ft),  batch_size=BATCH, shuffle=True,  drop_last=False)
    te_loader  = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_te,  Y_te),  batch_size=BATCH, shuffle=False, drop_last=False)

    model = PhysicsInformedBasicLiquidNetworkModel(
        input_size=1, hidden_size=hidden_size,
        n_appliances=len(APPLIANCES), dt=dt,
    ).to(device)

    phys_crit = TwoSidedPhysicsLoss(
        x_scaler, y_scalers, APPLIANCES,
        epsilon_over=epsilon_over,
        alpha_floor=alpha_floor,
        lambda_under=lambda_under,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    # ── Phase 1: Pretrain ──────────────────────────────────────────────────
    best_pt_state, pt_history = pretrain(
        model, pre_loader, val_loader, phys_crit,
        device, y_scalers, thresholds_scaled)

    pt_ckpt = os.path.join(save_dir, 'pretrain_model.pt')
    torch.save(best_pt_state, pt_ckpt)
    print(f"  Pretrain checkpoint saved: {pt_ckpt}")

    print("\n--- Test metrics BEFORE fine-tuning (pretrain model on House 5) ---")
    pre_test_metrics, _, _ = evaluate_test(model, te_loader, device, y_scalers)
    _print_metrics(pre_test_metrics)

    # ── Phase 2: Fine-tune ─────────────────────────────────────────────────
    ft_history = finetune(
        model, ft_loader, val_loader, phys_crit,
        device, y_scalers, thresholds_scaled)

    ft_ckpt = os.path.join(save_dir, 'finetuned_model.pt')
    torch.save({k: v.clone() for k, v in model.state_dict().items()}, ft_ckpt)
    print(f"  Fine-tuned checkpoint saved: {ft_ckpt}")

    # ── Phase 3: Test ──────────────────────────────────────────────────────
    print("\n--- Test metrics AFTER fine-tuning (fine-tuned model on House 5) ---")
    test_metrics, y_pred_te, y_true_te = evaluate_test(
        model, te_loader, device, y_scalers)
    _print_metrics(test_metrics)

    _plot(pt_history, ft_history, pre_test_metrics, test_metrics, save_dir)
    _save_json(pre_test_metrics, test_metrics, pt_history, ft_history,
               hidden_size, dt, epsilon_over, alpha_floor, lambda_under, save_dir)

    return test_metrics, pre_test_metrics, pt_history, ft_history


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _print_metrics(metrics):
    print(f"  {'Appliance':<20} {'F1':>6} {'Prec':>6} {'Rec':>6} "
          f"{'MAE':>7} {'SAE':>7} {'TP':>6} {'FP':>6} {'TN':>6} {'FN':>6}")
    print("  " + "-" * 80)
    for app in APPLIANCES:
        m = metrics[app]
        print(f"  {app:<20} {m['f1']:>6.4f} {m['precision']:>6.4f} {m['recall']:>6.4f} "
              f"{m['mae']:>7.2f} {m['sae']:>7.4f} "
              f"{m['TP']:>6} {m['FP']:>6} {m['TN']:>6} {m['FN']:>6}")


def _plot(pt_history, ft_history, pre_test_metrics, test_metrics, save_dir):
    ep = range(1, len(pt_history['train_loss']) + 1)

    # ── Pretrain loss curves ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle('Phase 1: Pretrain Loss — Two-Sided Physics')
    for ax, tr_key, va_key, title in zip(
        axes,
        ['train_loss', 'train_mse', 'train_over', 'train_under'],
        ['val_loss',   'val_mse',   'val_over',   'val_under'],
        ['Total Loss', 'MSE',       'Phys Over',  'Phys Under'],
    ):
        ax.plot(ep, pt_history[tr_key], label='Train', color='blue')
        ax.plot(ep, pt_history[va_key], label='Val',   color='red')
        ax.set_title(title); ax.set_xlabel('Epoch')
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pretrain_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Fine-tune loss curve ───────────────────────────────────────────────
    ft_ep = range(1, len(ft_history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Phase 2: Fine-tune Loss')
    for ax, key, color, title in zip(
        axes,
        ['train_mse', 'train_over', 'train_under'],
        ['blue', 'orange', 'green'],
        ['MSE', 'Physics Over', 'Physics Under'],
    ):
        ax.plot(ft_ep, ft_history[key], color=color)
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'finetune_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Before / After F1 bar chart ────────────────────────────────────────
    x     = np.arange(len(APPLIANCES))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, [pre_test_metrics[a]['f1'] for a in APPLIANCES],
           width, label='Before fine-tune', color='steelblue')
    ax.bar(x + width/2, [test_metrics[a]['f1']     for a in APPLIANCES],
           width, label='After fine-tune',  color='darkorange')
    ax.set_xticks(x); ax.set_xticklabels(APPLIANCES, rotation=15)
    ax.set_ylabel('F1'); ax.set_ylim(0, 1)
    ax.set_title('Test F1: Before vs After Fine-tuning (House 5) — Two-Sided PINN')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'f1_before_after.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Per-appliance val F1 during pretrain ───────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Pretrain Val F1 per Appliance — Two-Sided PINN')
    for ax, app in zip(axes.flat, APPLIANCES):
        f1s = [m[app]['f1'] for m in pt_history['val_metrics']]
        ax.plot(ep[:len(f1s)], f1s, color='blue')
        ax.axhline(pre_test_metrics[app]['f1'],  color='steelblue',  linestyle='--',
                   label='Test (pre-FT)')
        ax.axhline(test_metrics[app]['f1'],      color='darkorange', linestyle='--',
                   label='Test (post-FT)')
        ax.set_title(app); ax.set_xlabel('Epoch'); ax.set_ylabel('F1')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pretrain_val_f1.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _save_json(pre_test_metrics, test_metrics, pt_history, ft_history,
               hidden_size, dt, epsilon_over, alpha_floor, lambda_under, save_dir):
    def _serialise(m):
        return {k: (int(v) if isinstance(v, (np.integer, int)) else float(v))
                for k, v in m.items()}

    config = {
        'dataset': 'fine_tuning_dataset',
        'model':   'PhysicsInformedBasicLiquidNetworkModel',
        'physics': 'TwoSidedPhysicsLoss',
        'strategy': 'Cross-House Fine-Tuning (pretrain H1, fine-tune H5 first 2h)',
        'pretrain_params': {
            'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE,
            'warmup_epochs': WARMUP_EPOCHS,
            'lambda_phys': LAMBDA_PHYS,
            'epsilon_over': epsilon_over,
            'alpha_floor':  alpha_floor,
            'lambda_under': lambda_under,
        },
        'finetune_params': {
            'lr': LR_FT, 'epochs': EPOCHS_FT, 'patience': PATIENCE_FT,
            'lambda_phys': LAMBDA_PHYS,
            'epsilon_over': epsilon_over,
            'alpha_floor':  alpha_floor,
            'lambda_under': lambda_under,
        },
        'model_params': {
            'input_size': 1, 'hidden_size': hidden_size, 'dt': dt,
            'n_appliances': len(APPLIANCES),
        },
        'pretrain_epochs_run': len(pt_history['train_loss']),
        'finetune_epochs_run': len(ft_history['train_loss']),
        'test_metrics_before_finetune': {
            app: _serialise(m) for app, m in pre_test_metrics.items()},
        'test_metrics_after_finetune': {
            app: _serialise(m) for app, m in test_metrics.items()},
    }
    out = os.path.join(save_dir, 'pinn_twosided_lnn_finetune_results.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)
    print(f"\nResults saved to {save_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='PINN-BasicLNN with two-sided physical loss — cross-house fine-tuning')
    parser.add_argument('--dataset-dir',   default=DEFAULT_DATASET_DIR)
    parser.add_argument('--hidden-size',   type=int,   default=64)
    parser.add_argument('--dt',            type=float, default=0.1)
    parser.add_argument('--epsilon-over',  type=float, default=EPSILON_OVER,
                        help='Over-prediction tolerance in W (default 50)')
    parser.add_argument('--alpha-floor',   type=float, default=ALPHA_FLOOR,
                        help='Under-prediction floor as fraction of aggregate (default 0.15)')
    parser.add_argument('--lambda-under',  type=float, default=LAMBDA_UNDER,
                        help='Relative weight of under-prediction penalty (default 0.5)')
    args = parser.parse_args()

    run(dataset_dir=args.dataset_dir,
        hidden_size=args.hidden_size,
        dt=args.dt,
        epsilon_over=args.epsilon_over,
        alpha_floor=args.alpha_floor,
        lambda_under=args.lambda_under)
