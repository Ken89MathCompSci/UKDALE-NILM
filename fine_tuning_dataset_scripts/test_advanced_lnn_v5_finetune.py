"""
Advanced LNN v5 — Multi-Timescale Event-Aware LNN with Attentive Pooling

Four combined improvements, each motivated by v3 diagnostic observations:

  Observation 1 — Fridge: recall≈1.0, precision≈0.43
    Model predicts ON too often → hidden state stays active too long.
    Cause: single tau range that is too permissive for fridge duty cycles.

  Observation 2 — Microwave: precision≈0.95, recall≈0.40
    Model misses short bursty events → tau too slow or gate too conservative.
    Cause: same tau range shared with longer-cycle appliances.

  Observation 3 — Dishwasher: late convergence, long-memory dynamics.
    Needs a slow-timescale stream to track multi-stage cycles independently.

  Observation 4 — Washing machine: strong recall, weak precision.
    Long-cycle persistent memory — same class of issue as fridge.

Improvements:

  1. Multi-Timescale Hidden State
     Each timestep runs TWO coupled LNN streams in parallel:

       h_fast ← tau ∈ (tau_fast_min, tau_fast_max)   [transients, spikes]
       h_slow ← tau ∈ (tau_slow_min, tau_slow_max)   [cycles, long-range]

     Both streams see each other: combined = [xe, h_fast, h_slow].
     Output at each step: cat([h_fast, h_slow]) ∈ R^hidden.

     This mirrors the physical reality:
       dishwasher/washer  → dominated by slow stream
       microwave/fridge   → dominated by fast stream

  2. Appliance-Specific Tau Ranges
     Instead of a global [0.05, 5.0] for all appliances, use:

       microwave       fast=(0.01, 0.5)   slow=(0.1,  2.0)
       fridge          fast=(0.05, 2.0)   slow=(0.5,  6.0)
       dishwasher      fast=(0.01, 1.0)   slow=(0.5,  8.0)
       washing_machine fast=(0.05, 2.0)   slow=(1.0, 12.0)

     Directly targets the precision/recall imbalance observed in v3.

  3. Event-Aware Tau Modulation
     Compute e_t = |x_t - x_{t-1}| (scalar rate-of-change of aggregate).
     Feed [x_t, e_t] as joint input to both streams.
     Large aggregate transitions → e_t spikes → tau and gate both respond.
     Helps microwave (sharp ON/OFF edges in aggregate) and dishwasher.

  4. Attentive Pooling over All Hidden States
     Instead of using only the final state h_T for prediction:

       states = stack([h_1,...,h_T])          (B, T, hidden)
       α_t    = softmax(W_attn * h_t)         scalar per timestep
       context = Σ α_t * h_t                  (B, hidden)

     The label lives at the sequence midpoint (t = WIN//2).  Attention
     allows the model to focus there rather than being forced to use h_T.
     Also provides gradient access to all timesteps — helps early convergence.

Architecture summary:
    x (B, T, 1)
    → e_t = |x_t - x_{t-1}|  (B, T, 1)
    → xe = [x, e]             (B, T, 2)
    → MultiTimescaleLiquidLayer for each t:
          h_fast_t, h_slow_t  ∈ R^(hidden/2) each
    → states = cat([h_fast, h_slow])  (B, T, hidden)
    → attentive pooling → context     (B, hidden)
    → fc → output                     (B, 1)

Everything else (3-phase pipeline, scalers, metrics, plots, JSON) identical to v4.
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

# Appliance-specific tau ranges: each (fast_min, fast_max) and (slow_min, slow_max)
# Derived from v3 observations:
#   microwave → needs fast response (tiny tau), precision issue → shrink slow upper bound
#   fridge    → moderate; recall too high → tighten upper bounds vs washer
#   dishwasher→ long multi-stage cycles; generous slow upper
#   washer    → longest cycles; widest slow range
APPLIANCE_TAU = {
    'dishwasher':      {'fast': (0.01, 1.0),  'slow': (0.5,  8.0)},
    'fridge':          {'fast': (0.05, 2.0),  'slow': (0.5,  6.0)},
    'microwave':       {'fast': (0.01, 0.5),  'slow': (0.1,  2.0)},
    'washing_machine': {'fast': (0.05, 2.0),  'slow': (1.0, 12.0)},
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MultiTimescaleLiquidLayer(nn.Module):
    """
    Two coupled LNN streams (fast and slow) sharing a combined context.

    Input:  xe (B, input_size)  where xe = cat([x_t, e_t])
    State:  h_fast (B, half),  h_slow (B, half)
    Context: ctx = cat([xe, h_fast, h_slow])  (B, input_size + hidden)

    Fast update:
        gate_f = σ(W_gf ctx)
        τ_f    = τ_fmin + (τ_fmax - τ_fmin) × σ(W_τf ctx)
        f_f    = tanh(W_xf xe + h_fast @ W_rf)
        h_fast ← h_fast + dt(-h_fast/τ_f + gate_f·f_f)

    Slow update: same structure, different tau range.
    """

    def __init__(self, input_size, hidden_size, dt=0.1,
                 tau_fast=(0.01, 1.0), tau_slow=(0.5, 10.0)):
        super().__init__()
        assert hidden_size % 2 == 0, "hidden_size must be even"
        self.half = hidden_size // 2
        self.dt   = dt
        self.tau_fast_min, self.tau_fast_max = tau_fast
        self.tau_slow_min, self.tau_slow_max = tau_slow

        ctx_size = input_size + hidden_size   # xe + h_fast + h_slow

        # Fast stream
        self.fast_proj = nn.Linear(input_size, self.half)
        self.fast_rec  = nn.Parameter(torch.empty(self.half, self.half))
        self.fast_tau  = nn.Linear(ctx_size, self.half)
        self.fast_gate = nn.Linear(ctx_size, self.half)
        nn.init.xavier_uniform_(self.fast_rec)

        # Slow stream
        self.slow_proj = nn.Linear(input_size, self.half)
        self.slow_rec  = nn.Parameter(torch.empty(self.half, self.half))
        self.slow_tau  = nn.Linear(ctx_size, self.half)
        self.slow_gate = nn.Linear(ctx_size, self.half)
        nn.init.xavier_uniform_(self.slow_rec)

    def forward(self, xe, h_fast=None, h_slow=None):
        B = xe.size(0)
        if h_fast is None: h_fast = torch.zeros(B, self.half, device=xe.device)
        if h_slow is None: h_slow = torch.zeros(B, self.half, device=xe.device)

        ctx = torch.cat([xe, h_fast, h_slow], dim=1)   # (B, ctx_size)

        # Fast stream
        f_inp  = self.fast_proj(xe)
        f_rec  = torch.matmul(h_fast, self.fast_rec)
        f_gate = torch.sigmoid(self.fast_gate(ctx))
        f_tau  = (self.tau_fast_min
                  + (self.tau_fast_max - self.tau_fast_min)
                  * torch.sigmoid(self.fast_tau(ctx)))
        dh_f   = ((-h_fast / f_tau) + f_gate * torch.tanh(f_inp + f_rec)) * self.dt
        h_fast_new = (h_fast + dh_f).clamp(-10.0, 10.0)

        # Slow stream
        s_inp  = self.slow_proj(xe)
        s_rec  = torch.matmul(h_slow, self.slow_rec)
        s_gate = torch.sigmoid(self.slow_gate(ctx))
        s_tau  = (self.tau_slow_min
                  + (self.tau_slow_max - self.tau_slow_min)
                  * torch.sigmoid(self.slow_tau(ctx)))
        dh_s   = ((-h_slow / s_tau) + s_gate * torch.tanh(s_inp + s_rec)) * self.dt
        h_slow_new = (h_slow + dh_s).clamp(-10.0, 10.0)

        return h_fast_new, h_slow_new


class MultiTimescaleLNNModel(nn.Module):
    """
    Multi-timescale LNN with event signal and attentive pooling.

    input_size: dimension of each x_t (typically 1 for mains)
    hidden_size: total hidden dim; split equally as half fast, half slow
    tau_fast / tau_slow: (min, max) bounds for each stream's time constant
    """

    def __init__(self, input_size=1, hidden_size=64, output_size=1, dt=0.1,
                 tau_fast=(0.01, 1.0), tau_slow=(0.5, 10.0)):
        super().__init__()
        self.hidden_size = hidden_size
        # +1 for the event signal e_t appended to x_t
        self.lnn  = MultiTimescaleLiquidLayer(
            input_size + 1, hidden_size, dt, tau_fast, tau_slow)
        # Scalar attentive pooling: one weight per timestep
        self.attn = nn.Linear(hidden_size, 1)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """x: (B, T, input_size)"""
        B, T, _ = x.size()

        # Event signal: rate-of-change of aggregate; e_0 = 0
        e = torch.zeros_like(x)
        e[:, 1:, :] = (x[:, 1:, :] - x[:, :-1, :]).abs()

        h_fast = h_slow = None
        states = []
        for t in range(T):
            xe = torch.cat([x[:, t, :], e[:, t, :]], dim=1)   # (B, input_size+1)
            h_fast, h_slow = self.lnn(xe, h_fast, h_slow)
            states.append(torch.cat([h_fast, h_slow], dim=1)) # (B, hidden)

        states  = torch.stack(states, dim=1)              # (B, T, hidden)
        scores  = self.attn(states)                       # (B, T, 1)
        weights = F.softmax(scores, dim=1)                # (B, T, 1)
        context = (weights * states).sum(dim=1)           # (B, hidden)
        return self.fc(context)


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
                       hidden_size=64, dt=0.1, save_dir='models/advanced_lnn_v5_finetune'):

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tau_cfg  = APPLIANCE_TAU[appliance]
    tau_fast = tau_cfg['fast']
    tau_slow = tau_cfg['slow']
    print(f"\nAppliance: {appliance}  |  device: {device}  "
          f"tau_fast={tau_fast}  tau_slow={tau_slow}")

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

    model = MultiTimescaleLNNModel(
        input_size=1, hidden_size=hidden_size, output_size=1, dt=dt,
        tau_fast=tau_fast, tau_slow=tau_slow).to(device)
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
                        'dt': dt, 'tau_fast': tau_fast, 'tau_slow': tau_slow},
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
    plt.savefig(os.path.join(save_dir, f'advanced_lnn_v5_{appliance}_metrics.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # -- JSON -----------------------------------------------------------------
    config = {
        'appliance': appliance,
        'dataset':   'fine_tuning_dataset',
        'model':     'MultiTimescaleLNNModel (Advanced LNN v5)',
        'architecture': {
            'multi_timescale': 'fast + slow hidden streams, each (hidden/2)',
            'event_aware_tau': 'e_t = |x_t - x_{t-1}| fed into tau and gate',
            'attentive_pooling': 'scalar softmax attention over all T states',
            'appliance_specific_tau': True,
        },
        'tau': {'fast': list(tau_fast), 'slow': list(tau_slow)},
        'model_params': {'hidden_size': hidden_size, 'dt': dt,
                         'tau_fast': tau_fast, 'tau_slow': tau_slow},
        'pretrain_params': {'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE},
        'finetune_params': {'lr': LR_FT, 'epochs': EPOCHS_FT, 'patience': PATIENCE_FT},
        'test_metrics_before_finetune': {k: float(v) for k, v in pre_ft_metrics.items()},
        'test_metrics_after_finetune':  {k: float(v) for k, v in test_metrics.items()},
        'aggregates': _aggregates(history, test_metrics),
    }
    with open(os.path.join(save_dir, f'advanced_lnn_v5_{appliance}_results.json'),
              'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return model, history, test_metrics, pre_ft_metrics


# ---------------------------------------------------------------------------
# Run all appliances
# ---------------------------------------------------------------------------

def run_all(dataset_dir=DEFAULT_DATASET_DIR, hidden_size=64, dt=0.1):
    print("Loading fine_tuning_dataset splits...")
    splits    = load_splits(dataset_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir  = f'models/advanced_lnn_v5_finetune_{timestamp}'
    all_results = {}
    wall_start  = time.time()

    for app in APPLIANCES:
        print(f"\n{'='*60}\nAdvanced LNN v5 — {app}\n{'='*60}")
        app_dir = os.path.join(base_dir, app)
        try:
            _, _, after, before = train_on_appliance(
                splits, app, dataset_dir=dataset_dir,
                hidden_size=hidden_size, dt=dt, save_dir=app_dir)
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
        'model':     'MultiTimescaleLNNModel (Advanced LNN v5)',
        'dataset':   'fine_tuning_dataset',
        'improvements': [
            'multi_timescale_hidden_state (fast + slow streams)',
            'appliance_specific_tau_ranges',
            'event_aware_tau (e_t = |x_t - x_{t-1}|)',
            'attentive_pooling_over_all_states',
        ],
        'appliance_tau': {k: {'fast': list(v['fast']), 'slow': list(v['slow'])}
                          for k, v in APPLIANCE_TAU.items()},
        'model_params': {'hidden_size': hidden_size, 'dt': dt},
        'results': all_results,
    }
    with open(os.path.join(base_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    print(f"\nAdvanced LNN v5 complete.  Results → {base_dir}")
    for app, r in all_results.items():
        tau = APPLIANCE_TAU[app]
        print(f"  {app:<20}  F1  {r['before_finetune']['f1']:.4f} → "
              f"{r['after_finetune']['f1']:.4f}  "
              f"MAE  {r['before_finetune']['mae']:.1f} → {r['after_finetune']['mae']:.1f}  "
              f"tau_fast={tau['fast']}  tau_slow={tau['slow']}")
    print(f"Total wall-clock time: {(time.time()-wall_start)/60:.1f} min")
    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Advanced LNN v5: multi-timescale + event-aware tau + attention')
    p.add_argument('--dataset-dir',  default=DEFAULT_DATASET_DIR)
    p.add_argument('--hidden-size',  type=int,   default=64,
                   help='Total hidden size; split equally between fast and slow streams')
    p.add_argument('--dt',           type=float, default=0.1)
    args = p.parse_args()
    run_all(args.dataset_dir, args.hidden_size, args.dt)
