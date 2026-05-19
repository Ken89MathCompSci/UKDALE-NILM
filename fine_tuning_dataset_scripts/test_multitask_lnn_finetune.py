"""
Multi-Task LNN — Cross-House Fine-Tuning on fine_tuning_dataset/

Architecture:
    Aggregate Power
         ↓
    LNN Encoder  (full hidden-state sequence, not just last step)
         ↓
    Lightweight Attention  (last state queries full sequence → context)
         ↓
    Shared Latent  z(t)  = LayerNorm(ReLU(Linear(context)))
         ↓
    ┌──────────────────────────────────────────┐
    │  Power regression    sigmoid → [0,1]     │
    │  ON/OFF classifier   sigmoid → {0,1}     │
    │  Transition predictor sigmoid → {0,1}    │
    └──────────────────────────────────────────┘

Loss (all four appliances simultaneously):
    L = L_reg + 0.5·L_cls + 0.1·L_smooth + 0.05·L_cons

    L_reg    = Huber(pred_power, y_power_scaled)
    L_cls    = BCE(pred_class, y_class)          binary ON/OFF
    L_smooth = 0.5·BCE(pred_trans, y_trans)      transition head
               + 0.5·L1(pred[1:], pred[:-1])     within-batch TV (sequential loader)
    L_cons   = mean(ReLU(Σpred_raw − agg × 1.1)) conservation

Three-phase strategy (identical splits to other fine-tuning scripts):
    Phase 1 — Pretrain  : House 1 data  (pretrain.csv + validation.csv)
    Phase 2 — Fine-tune : House 5 first 2h  (finetune.csv)
    Phase 3 — Test      : House 5 remaining 22h  (test.csv)

Unlike other scripts, a single shared model handles all four appliances.
Per-appliance metrics from the power-regression head allow direct comparison
with single-appliance scripts.
"""

import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Source Code'))
from utils import calculate_nilm_metrics, save_model


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_DATASET_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'fine_tuning_dataset')

APPLIANCES = ['dishwasher', 'fridge', 'microwave', 'washing_machine']
THRESHOLD  = 10.0          # W — ON/OFF decision boundary
WIN    = 100;  STRIDE = 5;  BATCH = 32

EPOCHS    = 80;  PATIENCE    = 20;  LR    = 1e-3
EPOCHS_FT = 30;  PATIENCE_FT = 10;  LR_FT = 1e-4

# Loss weights
W_REG    = 1.0
W_CLS    = 0.5
W_SMOOTH = 0.1
W_CONS   = 0.05


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LNNMultiTaskModel(nn.Module):
    """
    LNN full-sequence encoder → lightweight attention → shared latent z(t)
    → three parallel heads (regression, classification, transition).
    """

    def __init__(self, input_size=1, hidden_size=64, n_appliances=4,
                 n_heads=2, dt=0.1):
        super().__init__()
        self.hidden_size  = hidden_size
        self.n_appliances = n_appliances
        self.dt           = dt

        # ── LNN Encoder ───────────────────────────────────────────────────
        self.input_proj  = nn.Linear(input_size, hidden_size)
        self.tau_base    = nn.Parameter(torch.ones(hidden_size))
        self.tau_mod     = nn.Linear(input_size, hidden_size)
        self.rec_weights = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.rec_weights)
        self.gate        = nn.Linear(input_size + hidden_size, hidden_size)
        self.norm_enc    = nn.LayerNorm(hidden_size)

        # ── Lightweight Attention ─────────────────────────────────────────
        # hidden_size must be divisible by n_heads (default 64/2=32 per head)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=n_heads,
            dropout=0.1, batch_first=True)
        self.norm_attn = nn.LayerNorm(hidden_size)

        # ── Shared Latent ─────────────────────────────────────────────────
        self.z_proj = nn.Linear(hidden_size, hidden_size)
        self.norm_z  = nn.LayerNorm(hidden_size)

        # ── Multi-task Heads ──────────────────────────────────────────────
        self.power_head = nn.Linear(hidden_size, n_appliances)   # regression
        self.class_head = nn.Linear(hidden_size, n_appliances)   # ON/OFF
        self.trans_head = nn.Linear(hidden_size, n_appliances)   # transition

    def _encode_sequence(self, x):
        """LNN forward for full sequence: (B, T, 1) → (B, T, hidden)."""
        B, T, _ = x.size()
        h = torch.zeros(B, self.hidden_size, device=x.device)
        states = []
        for t in range(T):
            x_t      = x[:, t, :]
            inp      = self.input_proj(x_t)
            rec      = torch.matmul(h, self.rec_weights)
            tau_base = F.softplus(self.tau_base).unsqueeze(0)
            tau_mod  = torch.sigmoid(self.tau_mod(x_t))
            tau      = (tau_base * tau_mod).clamp(min=self.dt)
            gate     = torch.sigmoid(self.gate(torch.cat([x_t, h], dim=1)))
            f_t      = torch.tanh(inp + rec)
            dh       = ((-h / tau) + gate * f_t) * self.dt
            h        = (h + dh).clamp(-10.0, 10.0)
            states.append(h)
        return self.norm_enc(torch.stack(states, dim=1))  # (B, T, hidden)

    def forward(self, x):
        seq = self._encode_sequence(x)           # (B, T, hidden)

        # Attention: last state queries full sequence
        query   = seq[:, -1:, :]                 # (B, 1, hidden)
        context, _ = self.attention(query, seq, seq)   # (B, 1, hidden)
        context = self.norm_attn(context.squeeze(1) + seq[:, -1, :])  # residual

        # Shared latent
        z = self.norm_z(F.relu(self.z_proj(context)))  # (B, hidden)

        pred_power = torch.sigmoid(self.power_head(z))   # (B, n_apps)
        pred_class = torch.sigmoid(self.class_head(z))   # (B, n_apps)
        pred_trans = torch.sigmoid(self.trans_head(z))   # (B, n_apps)
        return pred_power, pred_class, pred_trans


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y_power, Y_class, Y_trans, Agg):
        self.X       = torch.FloatTensor(X)
        self.Y_power = torch.FloatTensor(Y_power)
        self.Y_class = torch.FloatTensor(Y_class)
        self.Y_trans = torch.FloatTensor(Y_trans)
        self.Agg     = torch.FloatTensor(Agg)

    def __len__(self):         return len(self.X)
    def __getitem__(self, i):
        return (self.X[i], self.Y_power[i], self.Y_class[i],
                self.Y_trans[i], self.Agg[i])


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_splits(dataset_dir):
    splits = {}
    for name in ('pretrain', 'validation', 'finetune', 'test'):
        path = os.path.join(dataset_dir, f'UKDALE_HF_{name}.csv')
        splits[name] = pd.read_csv(path)
        print(f"  {name:12s}: {len(splits[name]):6,} rows")
    return splits


def create_sequences(df):
    """
    Returns
    -------
    X       : (N, WIN, 1)  scaled aggregate windows (caller must scale)
    Y_power : (N, 4)       raw mid-point appliance Watts
    Y_class : (N, 4)       binary ON/OFF (from raw, before scaling)
    Y_trans : (N, 4)       binary is-transition vs previous window
    Agg     : (N,)         raw mid-point aggregate Watts
    """
    mains    = df['aggregate'].values.astype(np.float32)
    app_vals = {a: df[a].values.astype(np.float32) for a in APPLIANCES}
    X, Y_power, Agg = [], [], []

    for i in range(0, len(mains) - WIN, STRIDE):
        X.append(mains[i: i + WIN])
        mid = i + WIN // 2
        Y_power.append([app_vals[a][mid] for a in APPLIANCES])
        Agg.append(mains[mid])

    X       = np.array(X,       dtype=np.float32).reshape(-1, WIN, 1)
    Y_power = np.array(Y_power, dtype=np.float32)
    Agg     = np.array(Agg,     dtype=np.float32)

    # Binary class labels from raw power (before scaling)
    Y_class = (Y_power > THRESHOLD).astype(np.float32)

    # Transition labels: 1 if state changed from previous window midpoint
    Y_trans = np.zeros_like(Y_class)
    Y_trans[1:] = (Y_class[1:] != Y_class[:-1]).astype(np.float32)

    return X, Y_power, Y_class, Y_trans, Agg


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _run_epoch(model, loader, optimizer, y_mins_t, y_ranges_t,
               device, train=True):
    model.train() if train else model.eval()
    tot = tot_reg = tot_cls = tot_smooth = tot_cons = 0.0
    all_pred, all_true = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb_pw, yb_cl, yb_tr, agg in loader:
            xb    = xb.to(device)
            yb_pw = yb_pw.to(device)    # scaled power
            yb_cl = yb_cl.to(device)    # binary class
            yb_tr = yb_tr.to(device)    # binary transition
            agg   = agg.to(device)      # raw aggregate Watts

            pred_power, pred_class, pred_trans = model(xb)

            L_reg = F.smooth_l1_loss(pred_power, yb_pw)
            L_cls = F.binary_cross_entropy(pred_class, yb_cl)

            # TV smoothness (within sequential batch) + transition BCE
            if pred_power.shape[0] > 1:
                L_tv = F.l1_loss(pred_power[1:], pred_power[:-1].detach())
            else:
                L_tv = pred_power.new_tensor(0.0)
            L_trans  = F.binary_cross_entropy(pred_trans, yb_tr)
            L_smooth = 0.5 * L_tv + 0.5 * L_trans

            # Conservation: Σpred_raw ≤ agg × 1.1
            pred_raw = pred_power * y_ranges_t + y_mins_t  # Watts
            excess   = F.relu(pred_raw.sum(dim=1) - agg * 1.1)
            L_cons   = excess.mean()

            loss = (W_REG * L_reg + W_CLS * L_cls
                    + W_SMOOTH * L_smooth + W_CONS * L_cons)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            tot        += loss.item()
            tot_reg    += L_reg.item()
            tot_cls    += L_cls.item()
            tot_smooth += L_smooth.item()
            tot_cons   += L_cons.item()
            all_pred.append(pred_power.detach().cpu().numpy())
            all_true.append(yb_pw.cpu().numpy())

    n = len(loader)
    return (tot/n, tot_reg/n, tot_cls/n, tot_smooth/n, tot_cons/n,
            np.concatenate(all_pred), np.concatenate(all_true))


def _metrics_per_appliance(pred_scaled, true_scaled, y_scalers):
    results = {}
    for i, app in enumerate(APPLIANCES):
        raw_pred = y_scalers[i].inverse_transform(
            pred_scaled[:, i:i+1]).flatten()
        raw_true = y_scalers[i].inverse_transform(
            true_scaled[:, i:i+1]).flatten()
        m = calculate_nilm_metrics(raw_true, raw_pred, threshold=THRESHOLD)
        m['TP'] = int(((raw_true > THRESHOLD) & (raw_pred > THRESHOLD)).sum())
        m['FP'] = int(((raw_true <= THRESHOLD) & (raw_pred > THRESHOLD)).sum())
        m['TN'] = int(((raw_true <= THRESHOLD) & (raw_pred <= THRESHOLD)).sum())
        m['FN'] = int(((raw_true > THRESHOLD) & (raw_pred <= THRESHOLD)).sum())
        results[app] = m
    return results


def _print_metrics(label, metrics):
    print(f"  {label}")
    print(f"    {'App':<22} {'F1':>6} {'P':>6} {'R':>6} "
          f"{'MAE':>7} {'SAE':>7} {'TP':>6} {'FP':>6} {'TN':>6} {'FN':>6}")
    for app, m in metrics.items():
        print(f"    {app:<22} {m['f1']:>6.4f} {m['precision']:>6.4f} "
              f"{m['recall']:>6.4f} {m['mae']:>7.2f} {m['sae']:>7.4f} "
              f"{m['TP']:>6} {m['FP']:>6} {m['TN']:>6} {m['FN']:>6}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_all(dataset_dir=DEFAULT_DATASET_DIR, hidden_size=64, n_heads=2,
            dt=0.1, save_dir=None):

    print("Loading fine_tuning_dataset splits...")
    splits    = load_splits(dataset_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if save_dir is None:
        save_dir = f'models/multitask_lnn_finetune_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  hidden={hidden_size}  n_heads={n_heads}  dt={dt}")
    print(f"Loss weights: reg={W_REG}  cls={W_CLS}  "
          f"smooth={W_SMOOTH}  cons={W_CONS}")

    wall_start = time.time()

    # ── Sequences ─────────────────────────────────────────────────────────
    X_pre, Yp_pre, Yc_pre, Yt_pre, Ag_pre = create_sequences(splits['pretrain'])
    X_val, Yp_val, Yc_val, Yt_val, Ag_val = create_sequences(splits['validation'])
    X_ft,  Yp_ft,  Yc_ft,  Yt_ft,  Ag_ft  = create_sequences(splits['finetune'])
    X_te,  Yp_te,  Yc_te,  Yt_te,  Ag_te  = create_sequences(splits['test'])

    # ── Scale ─────────────────────────────────────────────────────────────
    xs = MinMaxScaler()
    X_pre = xs.fit_transform(X_pre.reshape(-1,1)).reshape(X_pre.shape)
    X_val = xs.transform(X_val.reshape(-1,1)).reshape(X_val.shape)
    X_ft  = xs.transform(X_ft.reshape(-1,1)).reshape(X_ft.shape)
    X_te  = xs.transform(X_te.reshape(-1,1)).reshape(X_te.shape)

    y_scalers = []
    for i in range(len(APPLIANCES)):
        ys = MinMaxScaler()
        Yp_pre[:, i:i+1] = ys.fit_transform(Yp_pre[:, i:i+1])
        Yp_val[:, i:i+1] = ys.transform(Yp_val[:, i:i+1])
        Yp_ft[:, i:i+1]  = ys.transform(Yp_ft[:, i:i+1])
        Yp_te[:, i:i+1]  = ys.transform(Yp_te[:, i:i+1])
        y_scalers.append(ys)

    y_mins   = np.array([float(s.data_min_[0])   for s in y_scalers])
    y_ranges = np.array([float(s.data_range_[0]) for s in y_scalers])
    y_mins_t   = torch.FloatTensor(y_mins).to(device)
    y_ranges_t = torch.FloatTensor(y_ranges).to(device)

    print(f"  Train: {X_pre.shape}  Val: {X_val.shape}  "
          f"FT: {X_ft.shape}  Test: {X_te.shape}")

    mk_loader = lambda X, Yp, Yc, Yt, Ag, shuf: \
        torch.utils.data.DataLoader(
            MultiTaskDataset(X, Yp, Yc, Yt, Ag),
            batch_size=BATCH, shuffle=shuf)

    # Sequential (no shuffle) so within-batch TV loss is meaningful
    pre_loader = mk_loader(X_pre, Yp_pre, Yc_pre, Yt_pre, Ag_pre, False)
    val_loader = mk_loader(X_val, Yp_val, Yc_val, Yt_val, Ag_val, False)
    ft_loader  = mk_loader(X_ft,  Yp_ft,  Yc_ft,  Yt_ft,  Ag_ft,  False)
    te_loader  = mk_loader(X_te,  Yp_te,  Yc_te,  Yt_te,  Ag_te,  False)

    # ── Model ─────────────────────────────────────────────────────────────
    model = LNNMultiTaskModel(
        input_size=1, hidden_size=hidden_size, n_appliances=len(APPLIANCES),
        n_heads=n_heads, dt=dt).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # ── Phase 1: Pretrain ─────────────────────────────────────────────────
    print(f"\n{'='*60}\nPhase 1: Pretrain  ({EPOCHS} epochs max)\n{'='*60}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4)

    history = {k: [] for k in
               ('train_loss','val_loss','train_reg','train_cls','train_smooth','train_cons',
                'val_reg','val_cls','val_smooth','val_cons','val_metrics')}
    best_val   = float('inf'); best_state = None; counter = 0
    pretrain_start = time.time()

    for epoch in range(EPOCHS):
        ep_start = time.time()
        tr = _run_epoch(model, pre_loader, optimizer, y_mins_t, y_ranges_t, device, True)
        va = _run_epoch(model, val_loader, None,      y_mins_t, y_ranges_t, device, False)
        scheduler.step(va[0])

        vm = _metrics_per_appliance(va[5], va[6], y_scalers)
        avg_f1 = np.mean([vm[a]['f1']  for a in APPLIANCES])
        avg_mae= np.mean([vm[a]['mae'] for a in APPLIANCES])

        history['train_loss'].append(tr[0]);  history['val_loss'].append(va[0])
        history['train_reg'].append(tr[1]);   history['val_reg'].append(va[1])
        history['train_cls'].append(tr[2]);   history['val_cls'].append(va[2])
        history['train_smooth'].append(tr[3]);history['val_smooth'].append(va[3])
        history['train_cons'].append(tr[4]);  history['val_cons'].append(va[4])
        history['val_metrics'].append(vm)

        ep_time = time.time() - ep_start
        print(f"  Ep {epoch+1:3d}  "
              f"tr={tr[0]:.5f}(reg={tr[1]:.4f} cls={tr[2]:.4f} sm={tr[3]:.4f} cs={tr[4]:.4f})  "
              f"val={va[0]:.5f}  avgF1={avg_f1:.4f}  avgMAE={avg_mae:.2f}  time={ep_time:.1f}s")
        for app in APPLIANCES:
            m = vm[app]
            print(f"    {app:<22}  F1={m['f1']:.4f}  P={m['precision']:.4f}  "
                  f"R={m['recall']:.4f}  MAE={m['mae']:.2f}  SAE={m['sae']:.4f}  "
                  f"TP={m['TP']}  FP={m['FP']}  TN={m['TN']}  FN={m['FN']}")

        if va[0] < best_val:
            best_val = va[0]; counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            save_model(model,
                       {'input_size':1,'hidden_size':hidden_size,'n_heads':n_heads,'dt':dt},
                       {'lr':LR,'epochs':EPOCHS,'patience':PATIENCE},
                       {'avg_f1':float(avg_f1),'avg_mae':float(avg_mae)},
                       os.path.join(save_dir,'pretrain_best.pth'))
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}"); break

    print(f"  Phase 1 total: {(time.time()-pretrain_start)/60:.1f} min")
    model.load_state_dict(best_state)

    # Evaluate before fine-tune
    pre_ft = _run_epoch(model, te_loader, None, y_mins_t, y_ranges_t, device, False)
    pre_ft_metrics = _metrics_per_appliance(pre_ft[5], pre_ft[6], y_scalers)
    _print_metrics("Test BEFORE fine-tune:", pre_ft_metrics)

    # ── Phase 2: Fine-tune ────────────────────────────────────────────────
    print(f"\n{'='*60}\nPhase 2: Fine-tune  ({EPOCHS_FT} epochs max)\n{'='*60}")
    ft_optimizer = torch.optim.Adam(model.parameters(), lr=LR_FT)
    best_ft   = float('inf'); best_ft_state = None; ft_counter = 0
    ft_history = {'train_loss': []}
    ft_start  = time.time()

    for epoch in range(EPOCHS_FT):
        ep_start = time.time()
        tr = _run_epoch(model, ft_loader, ft_optimizer, y_mins_t, y_ranges_t, device, True)
        ft_history['train_loss'].append(tr[0])
        ep_time = time.time() - ep_start
        print(f"  FT Ep {epoch+1:2d}  loss={tr[0]:.5f}  "
              f"(reg={tr[1]:.4f} cls={tr[2]:.4f} sm={tr[3]:.4f} cs={tr[4]:.4f})  "
              f"time={ep_time:.1f}s")
        if tr[0] < best_ft:
            best_ft = tr[0]; ft_counter = 0
            best_ft_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            ft_counter += 1
            if ft_counter >= PATIENCE_FT:
                print(f"  FT early stopping at epoch {epoch+1}"); break

    print(f"  Phase 2 total: {(time.time()-ft_start)/60:.1f} min")
    model.load_state_dict(best_ft_state)

    # ── Phase 3: Test ─────────────────────────────────────────────────────
    te = _run_epoch(model, te_loader, None, y_mins_t, y_ranges_t, device, False)
    test_metrics = _metrics_per_appliance(te[5], te[6], y_scalers)
    _print_metrics("Test AFTER fine-tune:", test_metrics)

    # ── Plots ─────────────────────────────────────────────────────────────
    ep = range(1, len(history['train_loss']) + 1)

    # Pretrain: 5-panel loss breakdown
    plt.figure(figsize=(22, 4))
    for i, (tr_key, va_key, title, color) in enumerate([
        ('train_loss',  'val_loss',   'Total Loss',  'blue'),
        ('train_reg',   'val_reg',    'Regression',  'steelblue'),
        ('train_cls',   'val_cls',    'Classification', 'orange'),
        ('train_smooth','val_smooth', 'Smooth/Trans','green'),
        ('train_cons',  'val_cons',   'Conservation','purple'),
    ], 1):
        plt.subplot(1, 5, i)
        plt.plot(ep, history[tr_key], label='Train', color=color)
        plt.plot(ep, history[va_key], label='Val',   color=color, linestyle='--', alpha=0.7)
        plt.title(title); plt.xlabel('Epoch'); plt.legend(); plt.grid(alpha=0.3)
    plt.suptitle('Phase 1: Pretrain — Multi-Task LNN Loss Components', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pretrain_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Per-appliance F1/MAE curves
    fig, axes = plt.subplots(len(APPLIANCES), 2, figsize=(12, 4 * len(APPLIANCES)))
    fig.suptitle('Multi-Task LNN — Per-Appliance Val Metrics', fontsize=13)
    for row, app in enumerate(APPLIANCES):
        f1s  = [m[app]['f1']  for m in history['val_metrics']]
        maes = [m[app]['mae'] for m in history['val_metrics']]
        for col, vals, label, ytest, color in [
            (0, f1s,  'F1',     test_metrics[app]['f1'],  'steelblue'),
            (1, maes, 'MAE(W)', test_metrics[app]['mae'], 'red'),
        ]:
            axes[row][col].plot(ep, vals, color=color)
            axes[row][col].axhline(pre_ft_metrics[app]['f1' if col==0 else 'mae'],
                                   color='steelblue', linestyle='--', label='Test pre-FT')
            axes[row][col].axhline(ytest, color='darkorange', linestyle='--', label='Test post-FT')
            axes[row][col].set_title(f'{app} — {label}')
            axes[row][col].legend(); axes[row][col].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_appliance_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Fine-tune loss
    ft_ep = range(1, len(ft_history['train_loss']) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(ft_ep, ft_history['train_loss'], color='green')
    plt.title('Phase 2: Fine-tune Loss'); plt.xlabel('FT Epoch')
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'finetune_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── JSON summary ──────────────────────────────────────────────────────
    config = {
        'model': 'LNNMultiTaskModel',
        'dataset': 'fine_tuning_dataset',
        'model_params': {'hidden_size': hidden_size, 'n_heads': n_heads, 'dt': dt,
                         'n_params': n_params},
        'loss_weights': {'reg': W_REG, 'cls': W_CLS, 'smooth': W_SMOOTH, 'cons': W_CONS},
        'pretrain_params': {'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE},
        'finetune_params': {'lr': LR_FT, 'epochs': EPOCHS_FT, 'patience': PATIENCE_FT},
        'test_metrics_before_finetune': {
            app: {k: (int(v) if isinstance(v,(int,np.integer)) else float(v))
                  for k,v in m.items()}
            for app, m in pre_ft_metrics.items()},
        'test_metrics_after_finetune': {
            app: {k: (int(v) if isinstance(v,(int,np.integer)) else float(v))
                  for k,v in m.items()}
            for app, m in test_metrics.items()},
    }
    with open(os.path.join(save_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    print(f"\nMulti-Task LNN fine-tuning complete.  Results → {save_dir}")
    print(f"\n  {'App':<22} {'F1 pre':>8} {'F1 post':>8}  {'MAE pre':>8} {'MAE post':>8}")
    for app in APPLIANCES:
        print(f"  {app:<22} "
              f"{pre_ft_metrics[app]['f1']:>8.4f} {test_metrics[app]['f1']:>8.4f}  "
              f"{pre_ft_metrics[app]['mae']:>8.1f} {test_metrics[app]['mae']:>8.1f}")
    print(f"\nTotal wall-clock time: {(time.time()-wall_start)/60:.1f} min")
    return test_metrics, pre_ft_metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Multi-Task LNN fine-tuning on UKDALE')
    p.add_argument('--dataset-dir', default=DEFAULT_DATASET_DIR)
    p.add_argument('--hidden-size', type=int,   default=64)
    p.add_argument('--n-heads',     type=int,   default=2,
                   help='Attention heads (must divide hidden-size)')
    p.add_argument('--dt',          type=float, default=0.1)
    args = p.parse_args()
    run_all(args.dataset_dir, args.hidden_size, args.n_heads, args.dt)
