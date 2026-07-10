"""
Physics-Informed Advanced LNN (PINN-AdvancedLNN) for NILM — APR-new-House1-dataset splits.

Derived from Old-LNN_Algorithms/test_pinn_advanced_lnn_ukdale_specific_splits.py
with two changes:

1. DATA SOURCE — reads directly from APR-new-House1-dataset/ CSV files instead
   of the pickled data/ukdale/*_small.pkl splits:
       APR-new-House1-dataset/UKDALE_HF_train.csv      (House 1, 2014-09-08 to 2014-10-07, 30 days)
       APR-new-House1-dataset/UKDALE_HF_validation.csv (House 1, 2014-10-08 to 2014-10-14, 7 days)
       APR-new-House1-dataset/UKDALE_HF_test.csv       (House 1, 2014-10-15 to 2014-10-21, 7 days)
   All three splits are House 1, chronologically contiguous — no cross-house
   domain shift, no standby artefact.
   Columns: timestamp, aggregate, dishwasher, fridge, microwave, washing_machine

2. UNIFORM 10 W THRESHOLD — all appliances use 10 W (the old script special-
   cased 'washer dryer' to 0.5, tuned for a different pickled data scale that
   doesn't apply here). Matches the convention already established for this
   repo's House-1-only datasets (test_pinn_lnn_apr_dataset.py and
   test_pinn_lnn_apr_new_house1_dataset.py).

Architecture, loss, and training loop are otherwise unchanged from the source
script (this is the "advanced" upgrade over the single-layer PINN-LNN: a
2-layer stacked AdvancedLiquidNetworkModel encoder with inter-layer LayerNorm,
vs. the single AdvancedLiquidTimeLayer used in test_pinn_lnn_apr_new_house1_dataset.py):

    Input (batch, WIN, 1)  — scaled mains window
         |
    Layer 0: AdvancedLiquidTimeLayer (adaptive tau + input gate)
         |
    InterLayerNorm(hidden)
         |
    Layer 1: AdvancedLiquidTimeLayer (adaptive tau + input gate)
         |
    InterLayerNorm(hidden)
         |
    [DW | FR | MW | WM]  -- one Linear head per appliance
    output: (batch, 4)

Loss:
    L_total = MSE + lambda_phys * L_phys   (no BCE term -- matches source script)
    L_phys  = mean(ReLU(sum(p_hat_i_raw) - P_agg_raw - epsilon))
"""

import sys
import os
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Source Code'))
from utils import calculate_nilm_metrics, save_model


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPOCHS      = 80
PATIENCE    = 20
LR          = 1e-3
BATCH       = 32
WIN         = 200
STRIDE      = 5

LAMBDA_PHYS = 0.01   # physics loss weight -- kept small so MSE dominates
EPSILON_W   = 50.0   # tolerance for background / unlabelled loads (Watts)

APPLIANCES = ['dishwasher', 'fridge', 'microwave', 'washing_machine']

# Uniform 10 W threshold -- valid for all appliances with House 1 data.
THRESHOLDS = {app: 10.0 for app in APPLIANCES}

DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'APR-new-House1-dataset')


# ---------------------------------------------------------------------------
# Physics Consistency Loss
# ---------------------------------------------------------------------------

class PhysicsConsistencyLoss(nn.Module):
    """
    Soft one-sided penalty:  ReLU(sum(p_hat_i_raw) - P_agg_raw - epsilon)

    All arithmetic is in raw Watts via differentiable linear inverse-scaling.
    MinMaxScaler inverse: x_raw = x_scaled * data_range_ + data_min_
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
            x_mid_scaled: (batch,)        -- scaled mains value at window midpoint
            pred_scaled:  (batch, n_apps) -- scaled appliance predictions
        Returns:
            scalar loss
        """
        x_raw = x_mid_scaled * self.x_range + self.x_min          # (batch,)
        p_raw = pred_scaled  * self.y_ranges + self.y_mins        # (batch, n_apps)

        p_sum     = p_raw.sum(dim=1)                               # (batch,)
        violation = F.relu(p_sum - x_raw - self.epsilon)           # (batch,)
        return violation.mean()


# ---------------------------------------------------------------------------
# Advanced LNN cell (inline -- same as AdvancedLiquidTimeLayer in models.py)
# ---------------------------------------------------------------------------

class AdvancedLiquidTimeLayer(nn.Module):
    def __init__(self, input_size, hidden_size, dt=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt

        self.input_proj  = nn.Linear(input_size, hidden_size)
        self.tau_base    = nn.Parameter(torch.ones(hidden_size))
        self.tau_mod     = nn.Linear(input_size, hidden_size)
        self.rec_weights = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.rec_weights)
        self.gate        = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x_t, hidden):
        """
        Args:
            x_t:    (batch, input_size)
            hidden: (batch, hidden_size) or None
        Returns:
            new_hidden: (batch, hidden_size)
        """
        batch_size = x_t.size(0)
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=x_t.device)

        input_proj = self.input_proj(x_t)
        rec_proj   = torch.matmul(hidden, self.rec_weights)

        tau_base = F.softplus(self.tau_base).unsqueeze(0)
        tau_mod  = torch.sigmoid(self.tau_mod(x_t))
        tau      = (tau_base * tau_mod).clamp(min=self.dt)

        gate = torch.sigmoid(self.gate(torch.cat([x_t, hidden], dim=1)))

        f_t = torch.tanh(input_proj + rec_proj)
        dh  = ((-hidden / tau) + gate * f_t) * self.dt
        return (hidden + dh).clamp(-10.0, 10.0)


# ---------------------------------------------------------------------------
# Physics-Informed Advanced LNN Model
# ---------------------------------------------------------------------------

class PhysicsInformedAdvancedLiquidNetworkModel(nn.Module):
    """
    Stacked AdvancedLiquidNetworkModel encoder (num_layers layers, inter-layer
    LayerNorm) -> per-appliance linear output heads.

    The shared encoder produces a single hidden state that all appliance heads
    read from. The physics constraint is applied to the concatenated output at
    training time.
    """

    def __init__(self, input_size, hidden_size, n_appliances,
                 num_layers=2, dt=0.1):
        super().__init__()
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.n_appliances = n_appliances

        # Stacked AdvancedLiquidTimeLayer cells
        self.liquid_layers = nn.ModuleList([
            AdvancedLiquidTimeLayer(
                input_size if i == 0 else hidden_size,
                hidden_size,
                dt,
            ) for i in range(num_layers)
        ])

        # Inter-layer LayerNorm (normalises hidden state crossing layer boundaries)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])

        # Per-appliance output heads
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(n_appliances)
        ])

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            out: (batch, n_appliances)
        """
        batch_size, seq_len, _ = x.size()
        hidden_states = [None] * self.num_layers

        for t in range(seq_len):
            x_t = x[:, t, :]
            for i in range(self.num_layers):
                # Inter-layer: normalise signal coming from previous layer
                inp = x_t if i == 0 else self.layer_norms[i - 1](hidden_states[i - 1])
                hidden_states[i] = self.liquid_layers[i](inp, hidden_states[i])

        # Normalise final hidden state before heads
        h = self.layer_norms[-1](hidden_states[-1])

        # Each head: (batch, 1) -> concatenate to (batch, n_appliances)
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

def load_data(dataset_dir=DATASET_DIR):
    """Load train / validation / test from APR-new-House1-dataset/ CSV files."""
    print(f"Loading data from: {os.path.abspath(dataset_dir)}")

    def _load(filename):
        path = os.path.join(dataset_dir, filename)
        df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
        return df.rename(columns={'aggregate': 'main'})

    train_data = _load('UKDALE_HF_train.csv')
    val_data   = _load('UKDALE_HF_validation.csv')
    test_data  = _load('UKDALE_HF_test.csv')

    for name, df in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
        print(f"  {name}: {df.index.min()} to {df.index.max()}  ({len(df):,} rows)")

    return {'train': train_data, 'val': val_data, 'test': test_data}


def create_sequences(data, window_size=WIN):
    """Midpoint targeting -- y[i] is the appliance values at the window centre."""
    mains    = data['main'].values
    app_vals = {app: data[app].values for app in APPLIANCES}
    X, Y = [], []
    for i in range(0, len(mains) - window_size, STRIDE):
        X.append(mains[i:i + window_size])
        mid = i + window_size // 2
        Y.append([app_vals[app][mid] for app in APPLIANCES])
    return (
        np.array(X, dtype=np.float32).reshape(-1, window_size, 1),
        np.array(Y, dtype=np.float32),   # (N, n_appliances)
    )


# ---------------------------------------------------------------------------
# Per-appliance metrics helper
# ---------------------------------------------------------------------------

def compute_per_appliance_metrics(y_true, y_pred, y_scalers):
    metrics = {}
    for i, app in enumerate(APPLIANCES):
        raw_true = y_scalers[i].inverse_transform(
            y_true[:, i:i+1]).flatten()
        raw_pred = y_scalers[i].inverse_transform(
            y_pred[:, i:i+1]).flatten()
        metrics[app] = calculate_nilm_metrics(
            raw_true, raw_pred, threshold=THRESHOLDS[app])
    return metrics


# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def train_pinn_model(data_dict, save_dir,
                     hidden_size=64, num_layers=2, dt=0.1,
                     lambda_phys=LAMBDA_PHYS, epsilon_w=EPSILON_W):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"lambda_phys={lambda_phys}  epsilon={epsilon_w} W  "
          f"hidden={hidden_size}  num_layers={num_layers}  dt={dt}")
    print(f"Thresholds: { {a: THRESHOLDS[a] for a in APPLIANCES} }")

    # -- Sequences (all appliances together) --
    X_tr, Y_tr = create_sequences(data_dict['train'], WIN)
    X_va, Y_va = create_sequences(data_dict['val'],   WIN)
    X_te, Y_te = create_sequences(data_dict['test'],  WIN)

    # -- Scaling --
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

    print(f"Train: {X_tr.shape} -> {Y_tr.shape}")
    print(f"Val:   {X_va.shape} -> {Y_va.shape}")
    print(f"Test:  {X_te.shape} -> {Y_te.shape}")

    tr_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_tr, Y_tr), batch_size=BATCH, shuffle=True,  drop_last=False)
    va_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_va, Y_va), batch_size=BATCH, shuffle=False, drop_last=False)
    te_loader = torch.utils.data.DataLoader(
        MultiApplianceDataset(X_te, Y_te), batch_size=BATCH, shuffle=False, drop_last=False)

    # -- Model + losses --
    model = PhysicsInformedAdvancedLiquidNetworkModel(
        input_size=1, hidden_size=hidden_size,
        n_appliances=len(APPLIANCES), num_layers=num_layers, dt=dt,
    ).to(device)

    mse_criterion  = nn.MSELoss()
    phys_criterion = PhysicsConsistencyLoss(
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
    print("Starting PINN-AdvancedLNN training (all appliances simultaneously)...")

    for epoch in range(EPOCHS):
        # -- Training --
        model.train()
        ep_mse = ep_phys = ep_total = 0.0
        progress_bar = tqdm(tr_loader,
                            desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for xb, yb in progress_bar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            pred = model(xb)                          # (batch, n_apps)

            mse_loss  = mse_criterion(pred, yb)
            x_mid     = xb[:, WIN // 2, 0]            # (batch,) aggregate at midpoint
            phys_loss = phys_criterion(x_mid, pred)

            loss = mse_loss + lambda_phys * phys_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ep_mse   += mse_loss.item()
            ep_phys  += phys_loss.item()
            ep_total += loss.item()
            progress_bar.set_postfix({
                'mse': f'{mse_loss.item():.5f}',
                'phys': f'{phys_loss.item():.5f}',
            })

        avg_tr_mse   = ep_mse   / len(tr_loader)
        avg_tr_phys  = ep_phys  / len(tr_loader)
        avg_tr_total = ep_total / len(tr_loader)
        history['train_mse'].append(avg_tr_mse)
        history['train_phys'].append(avg_tr_phys)
        history['train_loss'].append(avg_tr_total)

        # -- Validation --
        model.eval()
        vl_mse = vl_phys = vl_total = 0.0
        val_preds, val_trues = [], []

        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)

                mse_loss  = mse_criterion(pred, yb)
                x_mid     = xb[:, WIN // 2, 0]
                phys_loss = phys_criterion(x_mid, pred)
                loss      = mse_loss + lambda_phys * phys_loss

                vl_mse   += mse_loss.item()
                vl_phys  += phys_loss.item()
                vl_total += loss.item()
                val_preds.append(pred.cpu().numpy())
                val_trues.append(yb.cpu().numpy())

        avg_va_mse   = vl_mse   / len(va_loader)
        avg_va_phys  = vl_phys  / len(va_loader)
        avg_va_total = vl_total / len(va_loader)
        history['val_mse'].append(avg_va_mse)
        history['val_phys'].append(avg_va_phys)
        history['val_loss'].append(avg_va_total)

        # Step scheduler on MSE only -- total loss oscillates due to physics term
        scheduler.step(avg_va_mse)

        y_pred_all = np.concatenate(val_preds)
        y_true_all = np.concatenate(val_trues)

        per_app_metrics = compute_per_appliance_metrics(
            y_true_all, y_pred_all, y_scalers)
        history['val_metrics'].append(per_app_metrics)

        avg_f1  = np.mean([per_app_metrics[a]['f1']  for a in APPLIANCES])
        avg_mae = np.mean([per_app_metrics[a]['mae'] for a in APPLIANCES])

        print(
            f"  Epoch {epoch+1:3d}/{EPOCHS}  "
            f"train={avg_tr_total:.5f} (mse={avg_tr_mse:.5f} phys={avg_tr_phys:.5f})  "
            f"val={avg_va_total:.5f} (mse={avg_va_mse:.5f} phys={avg_va_phys:.5f})  "
            f"avgF1={avg_f1:.4f}  avgMAE={avg_mae:.2f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        for app in APPLIANCES:
            m = per_app_metrics[app]
            print(f"    {app:<20}  F1={m['f1']:.4f}  "
                  f"P={m['precision']:.4f}  R={m['recall']:.4f}  "
                  f"MAE={m['mae']:.2f}  SAE={m['sae']:.4f}")

        # Early stop on val MSE -- prevents physics oscillations from triggering
        # early stopping before regression has converged
        if avg_va_mse < best_val_loss:
            best_val_loss = avg_va_mse
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            counter       = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    print("Training completed!")

    # -- Test evaluation --
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

    test_metrics = compute_per_appliance_metrics(y_true_te, y_pred_te, y_scalers)

    print(f"\n{'Appliance':<22} {'F1':>8} {'Precision':>10} {'Recall':>8} "
          f"{'MAE':>8} {'SAE':>8}")
    print("-" * 70)
    for app in APPLIANCES:
        m = test_metrics[app]
        print(f"{app:<22} {m['f1']:>8.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['mae']:>8.2f} {m['sae']:>8.4f}")

    # -- Plots --
    _plot_training(history, test_metrics, save_dir)

    # -- Save JSON --
    config = {
        'dataset': 'APR-new-House1-dataset (H1 train 2014-09-08..10-07 / val 10-08..10-14 / test 10-15..10-21)',
        'model': 'PhysicsInformedAdvancedLiquidNetworkModel',
        'description': (
            f'stacked AdvancedLiquidTimeLayer encoder ({num_layers} layers, '
            f'inter-layer LayerNorm) + per-appliance heads + L_phys; all House 1, 10 W threshold'
        ),
        'loss': f'MSE + {lambda_phys} * PhysicsConsistency(epsilon={epsilon_w}W)',
        'window_size': WIN,
        'thresholds': THRESHOLDS,
        'model_params': {
            'input_size': 1, 'hidden_size': hidden_size,
            'n_appliances': len(APPLIANCES),
            'num_layers': num_layers, 'dt': dt,
        },
        'train_params': {
            'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE,
            'lambda_phys': lambda_phys, 'epsilon_w': epsilon_w,
        },
        'test_metrics': {
            app: {k: float(v) for k, v in m.items()}
            for app, m in test_metrics.items()
        },
    }
    with open(os.path.join(save_dir, 'pinn_advanced_lnn_apr_new_house1_dataset_results.json'),
              'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return test_metrics, history


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_training(history, test_metrics, save_dir):
    epochs_x = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs_x, history['train_loss'], label='Train total', color='blue')
    plt.plot(epochs_x, history['val_loss'],   label='Val total',   color='red')
    plt.title('Total Loss (MSE + lambda*Phys)')
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
    plt.savefig(os.path.join(save_dir, 'pinn_advanced_lnn_apr_new_house1_dataset_loss.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(len(APPLIANCES), 2,
                             figsize=(12, 4 * len(APPLIANCES)))
    fig.suptitle('PINN-AdvancedLNN APR-new-House1-dataset -- Per-Appliance Val Metrics', fontsize=13)

    for row, app in enumerate(APPLIANCES):
        f1_series  = [m[app]['f1']  for m in history['val_metrics']]
        mae_series = [m[app]['mae'] for m in history['val_metrics']]

        ax_f1  = axes[row][0]
        ax_mae = axes[row][1]

        ax_f1.plot(epochs_x, f1_series, color='blue', linewidth=1.5)
        ax_f1.axhline(test_metrics[app]['f1'], color='green',
                      linestyle='--', label='Test F1')
        ax_f1.set_title(f'{app} -- F1')
        ax_f1.set_xlabel('Epoch'); ax_f1.set_ylabel('F1')
        ax_f1.legend(); ax_f1.grid(True, alpha=0.3)

        ax_mae.plot(epochs_x, mae_series, color='red', linewidth=1.5)
        ax_mae.axhline(test_metrics[app]['mae'], color='green',
                       linestyle='--', label='Test MAE')
        ax_mae.set_title(f'{app} -- MAE (W)')
        ax_mae.set_xlabel('Epoch'); ax_mae.set_ylabel('MAE (W)')
        ax_mae.legend(); ax_mae.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pinn_advanced_lnn_apr_new_house1_dataset_per_appliance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for fname in ['UKDALE_HF_train.csv', 'UKDALE_HF_validation.csv', 'UKDALE_HF_test.csv']:
        path = os.path.join(DATASET_DIR, fname)
        if not os.path.exists(path):
            print(f"Error: {path} not found!")
            sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir  = os.path.join(
        os.path.dirname(__file__), '..', 'models',
        f"pinn_advanced_lnn_apr_new_house1_dataset_{timestamp}"
    )

    data_dict = load_data()

    test_metrics, history = train_pinn_model(
        data_dict,
        save_dir     = save_dir,
        hidden_size  = 64,
        num_layers   = 2,
        dt           = 0.1,
        lambda_phys  = LAMBDA_PHYS,
        epsilon_w    = EPSILON_W,
    )

    print(f"\nResults saved to {save_dir}")
