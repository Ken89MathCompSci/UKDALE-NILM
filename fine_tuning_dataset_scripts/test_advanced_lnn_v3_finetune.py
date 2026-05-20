"""
Advanced LNN v3 — Per-Appliance Cross-House Fine-Tuning

Single targeted improvement over test_advanced_lnn_finetune.py: fully adaptive tau.

Old tau (models.py AdvancedLiquidTimeLayer):

    τ_t = softplus(τ_base) × σ(W_τ x_t + b_τ)

Problems:
  1. σ(·) ∈ [0,1] — modulation can only SHRINK τ below softplus(τ_base).
     If the model wants a longer time constant for a given input, it is blocked.
  2. tau_mod receives only x_t — no hidden-state dependency.
     Appliance dynamics often need τ to remain high for many steps after an
     activation event (e.g. washing machine mid-cycle), which requires the
     accumulated state h_t to keep τ extended, not just the current sample.

New tau (FullyAdaptiveLiquidTimeLayer):

    τ_t = τ_min + softplus(τ_base + W_τ [x_t, h_t] + b_τ)

Benefits:
  1. Additive softplus parameterisation — τ can GROW or SHRINK from τ_base.
     The modulation Δ = W_τ [x,h] can be positive or negative.
  2. tau_mod receives [x_t, h_t] — truly state-adaptive time constants.
     After detecting an activation, h_t carries that information forward and
     can keep τ elevated across multiple timesteps without input support.
  3. τ_min > 0 provides a finite minimum decay rate; prevents τ → ∞.

Comparison table:
  Property                     | Old (models.py)       | New (v3)
  -----------------------------|----------------------|---------------------
  tau_mod depends on x_t       | yes                  | yes
  tau_mod depends on h_t       | no                   | yes
  tau can grow above tau_base  | no  (sigmoid ≤ 1)    | yes (additive)
  tau can shrink below tau_base| yes                  | yes
  minimum tau guaranteed       | clamp(min=1e-3)      | tau_min + softplus
  per-hidden-unit tau          | yes                  | yes

Everything else (3-phase pipeline, scalers, metrics, plots, JSON) is identical
to test_advanced_lnn_finetune.py.
"""

import sys
import os
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Source Code'))
from utils import calculate_nilm_metrics, save_model


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

TAU_MIN = 0.01   # minimum time constant — finite decay floor, keeps grad stable


# ---------------------------------------------------------------------------
# Model: fully adaptive tau LNN
# ---------------------------------------------------------------------------

class FullyAdaptiveLiquidTimeLayer(nn.Module):
    """
    LNN cell with hidden-and-input-adaptive tau.

    Update rule (identical to AdvancedLiquidTimeLayer except tau):

        combined  = [x_t, h_t]
        gate      = σ(W_g combined + b_g)
        f_t       = tanh(W_x x_t + W_h h_t)
        τ_t       = τ_min + softplus(τ_base + W_τ combined + b_τ)  ← key change
        h_{t+1}   = h_t + dt(-h_t/τ_t + gate·f_t)

    vs. the original:
        τ_t = softplus(τ_base) × σ(W_τ x_t + b_τ)

    The additive form lets each hidden unit's time constant adapt freely
    in both directions from τ_base, conditioned on both the current input
    and the current memory state.
    """

    def __init__(self, input_size, hidden_size, dt=0.1, tau_min=TAU_MIN):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt      = dt
        self.tau_min = tau_min

        self.input_proj  = nn.Linear(input_size, hidden_size)
        # tau_base: per-unit learnable offset; ones → initial τ ≈ tau_min + softplus(1) ≈ 1.32
        self.tau_base    = nn.Parameter(torch.ones(hidden_size))
        # tau_mod takes [x, h]: input_size + hidden_size → hidden_size
        self.tau_mod     = nn.Linear(input_size + hidden_size, hidden_size)
        self.rec_weights = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.rec_weights)
        # gate also takes [x, h] (same as AdvancedLiquidTimeLayer)
        self.gate        = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, hidden=None):
        B = x.size(0)
        if hidden is None:
            hidden = torch.zeros(B, self.hidden_size, device=x.device)

        combined = torch.cat([x, hidden], dim=1)

        inp  = self.input_proj(x)
        rec  = torch.matmul(hidden, self.rec_weights)
        gate = torch.sigmoid(self.gate(combined))

        # fully adaptive tau: unbounded above tau_base; cannot go below tau_min
        tau  = self.tau_min + F.softplus(
            self.tau_base.unsqueeze(0) + self.tau_mod(combined)
        )

        f_t = torch.tanh(inp + rec)
        dh  = ((-hidden / tau) + gate * f_t) * self.dt
        return (hidden + dh).clamp(-10.0, 10.0)


class FullyAdaptiveLNNModel(nn.Module):
    """Stacks N FullyAdaptiveLiquidTimeLayer cells; final hidden state → output."""

    def __init__(self, input_size, hidden_size, output_size,
                 num_layers=2, dt=0.1, tau_min=TAU_MIN):
        super().__init__()
        self.num_layers  = num_layers
        self.hidden_size = hidden_size

        self.liquid_layers = nn.ModuleList([
            FullyAdaptiveLiquidTimeLayer(
                input_size if i == 0 else hidden_size,
                hidden_size, dt, tau_min)
            for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        B, T, _ = x.size()
        hs = [None] * self.num_layers
        for t in range(T):
            x_t = x[:, t, :]
            for i, lyr in enumerate(self.liquid_layers):
                inp  = x_t if i == 0 else hs[i - 1]
                hs[i] = lyr(inp, hs[i])
        return self.fc(hs[-1])


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
        splits[name] = pd.read_csv(path)
        print(f"  {name:12s}: {len(splits[name]):6,} rows")
    return splits


def create_sequences(df, appliance):
    mains = df['aggregate'].values
    tgts  = df[appliance].values
    X, y  = [], []
    for i in range(0, len(mains) - WIN, STRIDE):
        X.append(mains[i:i + WIN])
        y.append(tgts[i + WIN // 2])
    return (np.array(X, dtype=np.float32).reshape(-1, WIN, 1),
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
                       hidden_size=64, num_layers=2, dt=0.1, tau_min=TAU_MIN,
                       save_dir='models/advanced_lnn_v3_finetune'):

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nAppliance: {appliance}  |  device: {device}")

    X_pre, y_pre = create_sequences(splits['pretrain'],   appliance)
    X_val, y_val = create_sequences(splits['validation'], appliance)
    X_ft,  y_ft  = create_sequences(splits['finetune'],   appliance)
    X_te,  y_te  = create_sequences(splits['test'],       appliance)

    xs = MinMaxScaler(); ys = MinMaxScaler()
    X_pre = xs.fit_transform(X_pre.reshape(-1, 1)).reshape(X_pre.shape)
    X_val = xs.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
    X_ft  = xs.transform(X_ft.reshape(-1, 1)).reshape(X_ft.shape)
    X_te  = xs.transform(X_te.reshape(-1, 1)).reshape(X_te.shape)
    y_pre = ys.fit_transform(y_pre); y_val = ys.transform(y_val)
    y_ft  = ys.transform(y_ft);     y_te  = ys.transform(y_te)

    mk_loader = lambda X, y, shuf: torch.utils.data.DataLoader(
        UKDALEDataset(X, y), batch_size=BATCH, shuffle=shuf)
    pre_loader = mk_loader(X_pre, y_pre, True)
    val_loader = mk_loader(X_val, y_val, False)
    ft_loader  = mk_loader(X_ft,  y_ft,  True)
    te_loader  = mk_loader(X_te,  y_te,  False)

    model = FullyAdaptiveLNNModel(
        input_size=1, hidden_size=hidden_size, output_size=1,
        num_layers=num_layers, dt=dt, tau_min=tau_min).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")
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
                       {'input_size': 1, 'output_size': 1, 'hidden_size': hidden_size,
                        'num_layers': num_layers, 'dt': dt, 'tau_min': tau_min},
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
    plt.savefig(os.path.join(save_dir, f'advanced_lnn_v3_{appliance}_metrics.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # -- JSON -----------------------------------------------------------------
    config = {
        'appliance': appliance,
        'dataset':   'fine_tuning_dataset',
        'model':     'FullyAdaptiveLNNModel (Advanced LNN v3)',
        'tau_parameterisation': {
            'formula': 'tau_min + softplus(tau_base + W_tau[x,h] + b_tau)',
            'tau_min': tau_min,
            'vs_v1':  'softplus(tau_base) * sigmoid(W_tau x + b_tau)',
        },
        'model_params':  {'hidden_size': hidden_size, 'num_layers': num_layers,
                          'dt': dt, 'tau_min': tau_min},
        'pretrain_params': {'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE},
        'finetune_params': {'lr': LR_FT, 'epochs': EPOCHS_FT, 'patience': PATIENCE_FT},
        'test_metrics_before_finetune': {k: float(v) for k, v in pre_ft_metrics.items()},
        'test_metrics_after_finetune':  {k: float(v) for k, v in test_metrics.items()},
        'aggregates': _aggregates(history, test_metrics),
    }
    with open(os.path.join(save_dir, f'advanced_lnn_v3_{appliance}_results.json'),
              'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return model, history, test_metrics, pre_ft_metrics


# ---------------------------------------------------------------------------
# Run all appliances
# ---------------------------------------------------------------------------

def run_all(dataset_dir=DEFAULT_DATASET_DIR, hidden_size=64, num_layers=2,
            dt=0.1, tau_min=TAU_MIN):
    print("Loading fine_tuning_dataset splits...")
    splits    = load_splits(dataset_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir  = f'models/advanced_lnn_v3_finetune_{timestamp}'
    all_results = {}
    wall_start  = time.time()

    for app in APPLIANCES:
        print(f"\n{'='*60}\nAdvanced LNN v3 — {app}\n{'='*60}")
        app_dir = os.path.join(base_dir, app)
        try:
            _, _, after, before = train_on_appliance(
                splits, app, dataset_dir=dataset_dir,
                hidden_size=hidden_size, num_layers=num_layers,
                dt=dt, tau_min=tau_min, save_dir=app_dir)
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
        'model':     'FullyAdaptiveLNNModel (Advanced LNN v3)',
        'dataset':   'fine_tuning_dataset',
        'tau_parameterisation': {
            'formula': 'tau_min + softplus(tau_base + W_tau[x,h] + b_tau)',
            'tau_min': tau_min,
            'vs_v1':  'softplus(tau_base) * sigmoid(W_tau x + b_tau)',
        },
        'model_params': {'hidden_size': hidden_size, 'num_layers': num_layers,
                         'dt': dt, 'tau_min': tau_min},
        'results': all_results,
    }
    with open(os.path.join(base_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    print(f"\nAdvanced LNN v3 complete.  Results → {base_dir}")
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
        description='Advanced LNN v3: fully adaptive tau (x+h dependent, additive softplus)')
    p.add_argument('--dataset-dir',  default=DEFAULT_DATASET_DIR)
    p.add_argument('--hidden-size',  type=int,   default=64)
    p.add_argument('--num-layers',   type=int,   default=2)
    p.add_argument('--dt',           type=float, default=0.1)
    p.add_argument('--tau-min',      type=float, default=TAU_MIN,
                   help='Minimum time constant floor (default 0.01)')
    args = p.parse_args()
    run_all(args.dataset_dir, args.hidden_size, args.num_layers, args.dt, args.tau_min)
