"""
Advanced LNN + Time Embeddings — Per-Appliance Cross-House Fine-Tuning

Single addition over test_advanced_lnn_finetune.py: cyclical hour-of-day
features appended to the aggregate input at every timestep.

Standard NILM models receive only aggregate power:
    x_t = [P_t]

This version extends the input to:
    x_t = [P_t,  sin(2π·h_t/24),  cos(2π·h_t/24)]

where h_t is the fractional hour of the day (0–24) at timestep t.

Why cyclical encoding?
  - sin/cos pair maps midnight (h=0) and near-midnight (h=23.9) to the
    same neighbourhood in feature space; a plain hour integer does not.
  - Each appliance has strong time-of-day priors:
      Dishwasher      → post-meal peaks (morning/evening)
      Microwave       → meal-time spikes
      Fridge          → compressor cycles follow ambient temperature curve
      Washing machine → morning/weekend bias
  - The LNN's adaptive tau can now distinguish "short mains spike at 7am"
    (likely microwave) from "short spike at 3pm" (less likely), improving
    both precision and recall without adding any separate time classifier.

Why the LNN in particular benefits:
    tau_mod = σ(W_τ x_t)   now receives time features directly.
    gate    = σ(W_g [x_t, h_t]) also conditioned on time.
    This means the model can learn to extend or shrink its time constants
    based on where in the day it currently is — a natural inductive bias
    for appliance disaggregation.

Implementation notes:
  - Hour is extracted from the DataFrame index (parsed as datetime).
    Falls back to a 'timestamp' column, then to zeros if neither is present.
  - Only channel 0 (aggregate power) is MinMax-scaled; sin/cos are already
    in [-1, 1] and must NOT be scaled.
  - input_size changes from 1 → 3; everything else (model class, phases,
    scalers, metrics, plots, JSON) is identical to test_advanced_lnn_finetune.py.
"""

import sys
import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Source Code'))
from models import AdvancedLiquidNetworkModel
from utils  import calculate_nilm_metrics, save_model


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_DATASET_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'fine_tuning_dataset')

APPLIANCES = ['dishwasher', 'fridge', 'microwave', 'washing_machine']
THRESHOLD  = 10.0

EPOCHS    = 80;  PATIENCE    = 20;  LR    = 1e-3
EPOCHS_FT = 30;  PATIENCE_FT = 10;  LR_FT = 1e-4
BATCH     = 32;  WIN         = 100;  STRIDE = 5

INPUT_SIZE = 3   # [P_t, sin(2πh/24), cos(2πh/24)]


# ---------------------------------------------------------------------------
# Time features
# ---------------------------------------------------------------------------

def _hour_of_day(df):
    """
    Return fractional hour (0.0–24.0) for each row, preferring the DataFrame
    index, then a 'timestamp' column.  Falls back to zeros if neither parses.
    """
    try:
        dt = pd.to_datetime(df.index)
        if dt.isna().all():
            raise ValueError
        return (dt.hour + dt.minute / 60.0).astype(np.float32).values
    except Exception:
        pass
    for col in ('timestamp', 'time', 'datetime'):
        if col in df.columns:
            try:
                dt = pd.to_datetime(df[col])
                return (dt.dt.hour + dt.dt.minute / 60.0).astype(np.float32).values
            except Exception:
                continue
    return np.zeros(len(df), dtype=np.float32)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class UKDALEDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self):          return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def load_splits(dataset_dir):
    splits = {}
    for name in ('pretrain', 'validation', 'finetune', 'test'):
        path = os.path.join(dataset_dir, f'UKDALE_HF_{name}.csv')
        splits[name] = pd.read_csv(path, index_col=0)
        print(f"  {name:12s}: {len(splits[name]):6,} rows")
    return splits


def create_sequences(df, appliance):
    """
    Returns X of shape (N, WIN, 3): [normalised_power, sin_hour, cos_hour]
    and y of shape (N, 1): midpoint appliance power.

    Note: only channel 0 (power) needs MinMax scaling — caller handles that.
    """
    mains = df['aggregate'].values.astype(np.float32)
    tgts  = df[appliance].values.astype(np.float32)
    hours = _hour_of_day(df)
    sin_h = np.sin(2.0 * np.pi * hours / 24.0)
    cos_h = np.cos(2.0 * np.pi * hours / 24.0)

    X, y = [], []
    for i in range(0, len(mains) - WIN, STRIDE):
        # (WIN, 3): [power, sin, cos]
        X.append(np.stack([mains[i:i + WIN],
                           sin_h[i:i + WIN],
                           cos_h[i:i + WIN]], axis=1))
        y.append(tgts[i + WIN // 2])

    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32).reshape(-1, 1))


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total = 0.0
    outs, tgts = [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if train:
                optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item()
            outs.append(out.detach().cpu().numpy())
            tgts.append(yb.cpu().numpy())
    return total / len(loader), np.concatenate(outs), np.concatenate(tgts)


def _metrics(raw_true, raw_pred):
    return calculate_nilm_metrics(raw_true, raw_pred, threshold=THRESHOLD)


def _aggregates(history, test_metrics):
    vm = history['val_metrics']
    return {
        'train_loss_mean': float(np.mean(history['train_loss'])),
        'train_loss_var':  float(np.var(history['train_loss'])),
        'val_loss_mean':   float(np.mean(history['val_loss'])),
        'val_loss_var':    float(np.var(history['val_loss'])),
        'val_f1_mean':     float(np.mean([m['f1']  for m in vm])),
        'val_f1_var':      float(np.var( [m['f1']  for m in vm])),
        'val_mae_mean':    float(np.mean([m['mae'] for m in vm])),
        'val_mae_var':     float(np.var( [m['mae'] for m in vm])),
        'val_sae_mean':    float(np.mean([m['sae'] for m in vm])),
        'val_sae_var':     float(np.var( [m['sae'] for m in vm])),
        'test_f1':         float(test_metrics['f1']),
        'test_mae':        float(test_metrics['mae']),
        'test_sae':        float(test_metrics['sae']),
        'test_precision':  float(test_metrics['precision']),
        'test_recall':     float(test_metrics['recall']),
    }


# ---------------------------------------------------------------------------
# Per-appliance pipeline
# ---------------------------------------------------------------------------

def train_on_appliance(splits, appliance, dataset_dir=DEFAULT_DATASET_DIR,
                       hidden_size=64, num_layers=2, dt=0.1,
                       save_dir='models/advanced_lnn_te_finetune'):

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nAppliance: {appliance}  |  device: {device}")

    X_pre, y_pre = create_sequences(splits['pretrain'],   appliance)
    X_val, y_val = create_sequences(splits['validation'], appliance)
    X_ft,  y_ft  = create_sequences(splits['finetune'],   appliance)
    X_te,  y_te  = create_sequences(splits['test'],       appliance)

    print(f"  Sequences — pre:{X_pre.shape}  val:{X_val.shape}  "
          f"ft:{X_ft.shape}  te:{X_te.shape}")
    print(f"  sin/cos channels active: "
          f"sin range [{X_pre[:,:,1].min():.2f}, {X_pre[:,:,1].max():.2f}]")

    # Scale only channel 0 (aggregate power); sin/cos already in [-1, 1]
    xs = MinMaxScaler()
    n_pre = X_pre.shape[0]; n_val = X_val.shape[0]
    n_ft  = X_ft.shape[0];  n_te  = X_te.shape[0]
    X_pre[:, :, 0] = xs.fit_transform(
        X_pre[:, :, 0].reshape(-1, 1)).reshape(n_pre, WIN)
    X_val[:, :, 0] = xs.transform(
        X_val[:, :, 0].reshape(-1, 1)).reshape(n_val, WIN)
    X_ft[:, :, 0]  = xs.transform(
        X_ft[:, :, 0].reshape(-1, 1)).reshape(n_ft, WIN)
    X_te[:, :, 0]  = xs.transform(
        X_te[:, :, 0].reshape(-1, 1)).reshape(n_te, WIN)

    ys = MinMaxScaler()
    y_pre = ys.fit_transform(y_pre); y_val = ys.transform(y_val)
    y_ft  = ys.transform(y_ft);     y_te  = ys.transform(y_te)

    mk_loader = lambda X, y, shuf: torch.utils.data.DataLoader(
        UKDALEDataset(X, y), batch_size=BATCH, shuffle=shuf)
    pre_loader = mk_loader(X_pre, y_pre, True)
    val_loader = mk_loader(X_val, y_val, False)
    ft_loader  = mk_loader(X_ft,  y_ft,  True)
    te_loader  = mk_loader(X_te,  y_te,  False)

    model = AdvancedLiquidNetworkModel(
        input_size=INPUT_SIZE, hidden_size=hidden_size,
        output_size=1, num_layers=num_layers, dt=dt).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}  (input_size={INPUT_SIZE})")
    criterion = torch.nn.MSELoss()

    # -- Phase 1: Pretrain ----------------------------------------------------
    print("  Phase 1: Pretrain")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)
    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    best_val  = float('inf'); best_state = None; counter = 0

    pretrain_start = time.time()
    for epoch in range(EPOCHS):
        ep_start = time.time()
        tr_loss, _, _   = _run_epoch(model, pre_loader, criterion, optimizer, device, True)
        va_loss, vo, vt = _run_epoch(model, val_loader, criterion, optimizer, device, False)
        scheduler.step(va_loss)
        raw_t = ys.inverse_transform(vt).flatten()
        raw_o = ys.inverse_transform(vo).flatten()
        m = _metrics(raw_t, raw_o)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['val_metrics'].append(m)
        ep_time = time.time() - ep_start
        print(f"    Ep {epoch+1:3d}  train={tr_loss:.5f}  val={va_loss:.5f}  "
              f"F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  "
              f"MAE={m['mae']:.2f}  SAE={m['sae']:.4f}  "
              f"TP={m['TP']}  FP={m['FP']}  TN={m['TN']}  FN={m['FN']}  "
              f"time={ep_time:.1f}s")
        if va_loss < best_val:
            best_val = va_loss; counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            save_model(model,
                       {'input_size': INPUT_SIZE, 'output_size': 1,
                        'hidden_size': hidden_size, 'num_layers': num_layers, 'dt': dt},
                       {'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE,
                        'appliance': appliance},
                       m, os.path.join(save_dir, f'pretrain_{appliance}_best.pth'))
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"    Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    print(f"  Phase 1 total: {(time.time()-pretrain_start)/60:.1f} min")

    # Before fine-tune test
    _, to, tt = _run_epoch(model, te_loader, criterion, optimizer, device, False)
    pre_ft_metrics = _metrics(ys.inverse_transform(tt).flatten(),
                               ys.inverse_transform(to).flatten())
    print(f"  Test BEFORE fine-tune: "
          f"F1={pre_ft_metrics['f1']:.4f}  P={pre_ft_metrics['precision']:.4f}  "
          f"R={pre_ft_metrics['recall']:.4f}  MAE={pre_ft_metrics['mae']:.2f}  "
          f"SAE={pre_ft_metrics['sae']:.4f}  "
          f"TP={pre_ft_metrics['TP']}  FP={pre_ft_metrics['FP']}  "
          f"TN={pre_ft_metrics['TN']}  FN={pre_ft_metrics['FN']}")

    # -- Phase 2: Fine-tune ---------------------------------------------------
    print("  Phase 2: Fine-tune")
    ft_optimizer = torch.optim.Adam(model.parameters(), lr=LR_FT)
    best_ft = float('inf'); best_ft_state = None; ft_counter = 0
    ft_history = {'train_loss': []}

    ft_start = time.time()
    for epoch in range(EPOCHS_FT):
        ep_start = time.time()
        tr_loss, _, _ = _run_epoch(model, ft_loader, criterion, ft_optimizer, device, True)
        ft_history['train_loss'].append(tr_loss)
        ep_time = time.time() - ep_start
        print(f"    FT Ep {epoch+1:2d}  loss={tr_loss:.5f}  time={ep_time:.1f}s")
        if tr_loss < best_ft:
            best_ft = tr_loss; ft_counter = 0
            best_ft_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            ft_counter += 1
            if ft_counter >= PATIENCE_FT:
                print(f"    FT early stopping at epoch {epoch+1}"); break

    model.load_state_dict(best_ft_state)
    print(f"  Phase 2 total: {(time.time()-ft_start)/60:.1f} min")

    # -- Phase 3: Test --------------------------------------------------------
    _, to, tt = _run_epoch(model, te_loader, criterion, ft_optimizer, device, False)
    test_metrics = _metrics(ys.inverse_transform(tt).flatten(),
                             ys.inverse_transform(to).flatten())
    print(f"  Test AFTER  fine-tune: "
          f"F1={test_metrics['f1']:.4f}  P={test_metrics['precision']:.4f}  "
          f"R={test_metrics['recall']:.4f}  MAE={test_metrics['mae']:.2f}  "
          f"SAE={test_metrics['sae']:.4f}  "
          f"TP={test_metrics['TP']}  FP={test_metrics['FP']}  "
          f"TN={test_metrics['TN']}  FN={test_metrics['FN']}")

    # -- Plot -----------------------------------------------------------------
    ep = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(18, 10))
    plt.subplot(2, 3, 1)
    plt.plot(ep, history['train_loss'], label='Train', color='blue')
    plt.plot(ep, history['val_loss'],   label='Val',   color='red')
    plt.title(f'Pretrain Loss — {appliance}'); plt.xlabel('Epoch')
    plt.legend(); plt.grid(alpha=0.3)
    plt.subplot(2, 3, 2)
    plt.plot(ep, [m['mae'] for m in history['val_metrics']], color='red', label='Val MAE')
    plt.axhline(pre_ft_metrics['mae'], color='steelblue',  linestyle='--', label='Test pre-FT')
    plt.axhline(test_metrics['mae'],   color='darkorange', linestyle='--', label='Test post-FT')
    plt.title(f'MAE — {appliance}'); plt.xlabel('Epoch'); plt.legend(); plt.grid(alpha=0.3)
    plt.subplot(2, 3, 3)
    plt.plot(ep, [m['sae'] for m in history['val_metrics']], color='purple', label='Val SAE')
    plt.axhline(pre_ft_metrics['sae'], color='steelblue',  linestyle='--', label='Test pre-FT')
    plt.axhline(test_metrics['sae'],   color='darkorange', linestyle='--', label='Test post-FT')
    plt.title(f'SAE — {appliance}'); plt.xlabel('Epoch'); plt.legend(); plt.grid(alpha=0.3)
    plt.subplot(2, 3, 4)
    plt.plot(ep, [m['f1'] for m in history['val_metrics']], color='red', label='Val F1')
    plt.axhline(pre_ft_metrics['f1'],  color='steelblue',  linestyle='--', label='Test pre-FT')
    plt.axhline(test_metrics['f1'],    color='darkorange', linestyle='--', label='Test post-FT')
    plt.title(f'F1 — {appliance}'); plt.xlabel('Epoch'); plt.legend(); plt.grid(alpha=0.3)
    plt.subplot(2, 3, 5)
    ft_ep = range(1, len(ft_history['train_loss']) + 1)
    plt.plot(ft_ep, ft_history['train_loss'], color='green', label='FT train loss')
    plt.title(f'Fine-tune Loss — {appliance}'); plt.xlabel('FT Epoch')
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'advanced_lnn_te_{appliance}_metrics.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # -- JSON -----------------------------------------------------------------
    config = {
        'appliance': appliance,
        'dataset':   'fine_tuning_dataset',
        'model':     'AdvancedLiquidNetworkModel + time embeddings',
        'time_embedding': {
            'features':  ['sin(2pi*h/24)', 'cos(2pi*h/24)'],
            'input_size': INPUT_SIZE,
            'note': 'cyclical encoding; sin/cos not scaled (already in [-1,1])',
        },
        'model_params': {'input_size': INPUT_SIZE, 'hidden_size': hidden_size,
                         'num_layers': num_layers, 'dt': dt},
        'pretrain_params': {'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE},
        'finetune_params': {'lr': LR_FT, 'epochs': EPOCHS_FT, 'patience': PATIENCE_FT},
        'test_metrics_before_finetune': {k: float(v) for k, v in pre_ft_metrics.items()},
        'test_metrics_after_finetune':  {k: float(v) for k, v in test_metrics.items()},
        'aggregates': _aggregates(history, test_metrics),
    }
    with open(os.path.join(save_dir, f'advanced_lnn_te_{appliance}_results.json'),
              'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return model, history, test_metrics, pre_ft_metrics


# ---------------------------------------------------------------------------
# Run all appliances
# ---------------------------------------------------------------------------

def run_all(dataset_dir=DEFAULT_DATASET_DIR, hidden_size=64, num_layers=2, dt=0.1):
    print("Loading fine_tuning_dataset splits...")
    splits    = load_splits(dataset_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir  = f'models/advanced_lnn_te_finetune_{timestamp}'
    all_results = {}
    wall_start  = time.time()

    for app in APPLIANCES:
        print(f"\n{'='*60}\nAdvanced LNN + Time Embeddings — {app}\n{'='*60}")
        app_dir = os.path.join(base_dir, app)
        try:
            _, _, after, before = train_on_appliance(
                splits, app, dataset_dir=dataset_dir,
                hidden_size=hidden_size, num_layers=num_layers,
                dt=dt, save_dir=app_dir)
            all_results[app] = {
                'before_finetune': {k: float(v) for k, v in before.items()},
                'after_finetune':  {k: float(v) for k, v in after.items()},
            }
        except Exception as e:
            print(f"Error on {app}: {e}")
            import traceback; traceback.print_exc()

    os.makedirs(base_dir, exist_ok=True)
    summary = {
        'timestamp': timestamp,
        'model':     'AdvancedLiquidNetworkModel + time embeddings',
        'dataset':   'fine_tuning_dataset',
        'time_embedding': 'x_t = [P_t, sin(2pi*h/24), cos(2pi*h/24)]',
        'model_params': {'input_size': INPUT_SIZE, 'hidden_size': hidden_size,
                         'num_layers': num_layers, 'dt': dt},
        'results': all_results,
    }
    with open(os.path.join(base_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    print(f"\nAdvanced LNN + TE complete.  Results → {base_dir}")
    for app, r in all_results.items():
        print(f"  {app:<20}  F1  {r['before_finetune']['f1']:.4f} → "
              f"{r['after_finetune']['f1']:.4f}  "
              f"MAE  {r['before_finetune']['mae']:.1f} → {r['after_finetune']['mae']:.1f}")
    print(f"Total wall-clock time: {(time.time()-wall_start)/60:.1f} min")
    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Advanced LNN with cyclical time embeddings: '
                    'x_t = [P_t, sin(2pi*h/24), cos(2pi*h/24)]')
    p.add_argument('--dataset-dir',  default=DEFAULT_DATASET_DIR)
    p.add_argument('--hidden-size',  type=int,   default=64)
    p.add_argument('--num-layers',   type=int,   default=2)
    p.add_argument('--dt',           type=float, default=0.1)
    args = p.parse_args()
    run_all(args.dataset_dir, args.hidden_size, args.num_layers, args.dt)
