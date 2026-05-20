"""
Advanced LNN v2 — Multi-Task Cross-House Fine-Tuning

Extends test_advanced_lnn_finetune.py with four targeted improvements:

  1. Multi-Resolution TCN Frontend
     Dilated convolutions (rates 1/2/4) capture appliance signatures at
     multiple temporal scales before the liquid dynamics integrate them.
     Helps dishwasher (long-state) and microwave (bursty) simultaneously.

  2. Appliance-Aware Dynamic Tau
     Each appliance embedding is used as the attention QUERY over the LNN
     sequence.  A learned tau_app layer scales each query, biasing each
     appliance to attend to slow vs fast temporal patterns.  This gives
     each appliance its own effective time constant without running 4
     separate LNN encoders.  Functionally: τ_i = τ_shared * σ(τ_app(e_i)).

  3. Hybrid Event + Power Branches
     Separate classification head (ON/OFF, Focal BCE) and regression head
     (wattage, Smooth-L1) share the appliance-conditioned latent z_i.
     Separating the tasks lets each head specialise: event head focuses
     on activation structure, power head on magnitude calibration.

  4. Energy Conservation Loss
     L_cons = mean(relu(Σ pred_raw_i - agg * 1.1))
     Penalises windows where the sum of predicted appliance loads exceeds
     the measured aggregate.  Directly attacks SAE / energy calibration.

Phase structure (identical to v1):
    Phase 1 — Supervised pretrain : House 1  (pretrain + validation splits)
    Phase 2 — Fine-tune           : House 5  (finetune split, frozen encoder)
    Phase 3 — Test                : House 5  (test split)
"""

import sys
import os
import time
import json
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
WIN = 100;  STRIDE = 5;  BATCH = 32

# Phase 1: supervised pretrain
EPOCHS = 80;  PATIENCE = 20;  LR = 1e-3

# Phase 2: fine-tune
EPOCHS_FT = 30;  PATIENCE_FT = 10;  LR_FT = 1e-4

# Loss weights  (higher W_CONS vs v1 to improve SAE / energy calibration)
W_POWER  = 1.0
W_EVENT  = 0.5
W_TRANS  = 0.1
W_CONS   = 0.05

# Improvement 3: focal loss for event/transition heads
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.35

# Per-appliance thresholds for evaluation
CLASS_THRESHOLDS = {
    'dishwasher':      0.35,
    'fridge':          0.55,
    'microwave':       0.30,
    'washing_machine': 0.35,
}

# Improvement 1: multi-resolution TCN
TCN_HIDDEN = 32;  TCN_LAYERS = 3   # dilations: 1, 2, 4

HIDDEN_SIZE = 64;  N_HEADS = 2;  DT = 0.1


# ---------------------------------------------------------------------------
# Improvement 3: Focal BCE (per-sample alpha)
# ---------------------------------------------------------------------------

def focal_bce(pred, target):
    eps     = 1e-6
    pred    = pred.clamp(eps, 1.0 - eps)
    bce     = F.binary_cross_entropy(pred, target, reduction='none')
    p_t     = torch.where(target == 1.0, pred, 1.0 - pred)
    alpha_t = torch.where(target == 1.0,
                          torch.full_like(target, FOCAL_ALPHA),
                          torch.full_like(target, 1.0 - FOCAL_ALPHA))
    return (alpha_t * (1.0 - p_t).pow(FOCAL_GAMMA) * bce).mean()


# ---------------------------------------------------------------------------
# Improvement 1: Multi-Resolution TCN Frontend
# ---------------------------------------------------------------------------

class DilatedTCN(nn.Module):
    """3-layer dilated TCN; preserves sequence length via padding=dilation."""
    def __init__(self, in_channels=1, hidden=TCN_HIDDEN, n_layers=TCN_LAYERS):
        super().__init__()
        layers = []
        ch = in_channels
        for i in range(n_layers):
            d = 2 ** i
            layers += [nn.Conv1d(ch, hidden, kernel_size=3, padding=d, dilation=d),
                       nn.ReLU(), nn.BatchNorm1d(hidden)]
            ch = hidden
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.permute(0, 2, 1)).permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Model: EventPowerLNN
# ---------------------------------------------------------------------------

class EventPowerLNN(nn.Module):
    """
    DilatedTCN → Shared LNN encoder → Appliance-Specific Attention
    → Dual heads: power regression + event classification.

    Appliance-Aware Dynamic Tau (Improvement 2):
        Each appliance embedding e_i is used as the attention QUERY.
        A learned tau_app layer scales e_i, biasing each appliance to
        attend to different temporal patterns in the shared LNN sequence.
        Different appliances therefore see different effective time scales
        without running 4 separate LNN encoders.

    Hybrid Event + Power (Improvement 3):
        power_head → Smooth-L1 regression (wattage)
        event_head → Focal BCE classification (ON/OFF)
        trans_head → Focal BCE (state transitions)
    """

    def __init__(self, input_size=1, hidden_size=HIDDEN_SIZE,
                 n_appliances=4, n_heads=N_HEADS, dt=DT,
                 tcn_hidden=TCN_HIDDEN):
        super().__init__()
        self.hidden_size  = hidden_size
        self.n_appliances = n_appliances
        self.dt = dt

        # Improvement 1: multi-resolution frontend
        self.tcn = DilatedTCN(in_channels=input_size, hidden=tcn_hidden)

        # ── Shared LNN encoder ──────────────────────────────────────────
        self.input_proj  = nn.Linear(tcn_hidden, hidden_size)
        self.tau_base    = nn.Parameter(torch.ones(hidden_size))
        self.tau_mod     = nn.Linear(tcn_hidden, hidden_size)
        self.rec_weights = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.rec_weights)
        self.gate        = nn.Linear(tcn_hidden + hidden_size, hidden_size)
        self.norm_enc    = nn.LayerNorm(hidden_size)

        # Improvement 2: appliance embeddings + tau_app
        self.appliance_emb = nn.Embedding(n_appliances, hidden_size)
        # tau_app: per-appliance query scaling — biases attention towards
        # appliance-specific temporal frequencies (slow fridge, fast microwave)
        self.tau_app = nn.Linear(hidden_size, hidden_size, bias=False)

        # Appliance-specific attention over LNN sequence
        self.app_attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=n_heads,
            dropout=0.1, batch_first=True)
        self.norm_attn = nn.LayerNorm(hidden_size)

        # Latent projection
        self.z_proj = nn.Linear(hidden_size, hidden_size)
        self.norm_z  = nn.LayerNorm(hidden_size)

        # Improvement 3: dual prediction heads (shared, conditioned on z_i)
        self.power_head = nn.Linear(hidden_size, 1)  # regression
        self.event_head = nn.Linear(hidden_size, 1)  # ON/OFF classification
        self.trans_head = nn.Linear(hidden_size, 1)  # transition detection

    def _encode(self, x):
        """x: (B, T, 1) → seq: (B, T, hidden)"""
        tcn_out = self.tcn(x)
        B, T, _ = tcn_out.size()
        h = torch.zeros(B, self.hidden_size, device=x.device)
        states = []
        for t in range(T):
            xt    = tcn_out[:, t, :]
            inp   = self.input_proj(xt)
            rec   = torch.matmul(h, self.rec_weights)
            tau   = (F.softplus(self.tau_base).unsqueeze(0)
                     * torch.sigmoid(self.tau_mod(xt))).clamp(min=self.dt)
            gate  = torch.sigmoid(self.gate(torch.cat([xt, h], dim=1)))
            dh    = ((-h / tau) + gate * torch.tanh(inp + rec)) * self.dt
            h     = (h + dh).clamp(-10.0, 10.0)
            states.append(h)
        return self.norm_enc(torch.stack(states, dim=1))  # (B, T, hidden)

    def forward(self, x):
        seq = self._encode(x)                             # (B, T, hidden)
        B   = seq.size(0)

        # Appliance-specific attention (Improvement 2 — appliance-aware tau)
        app_idx = torch.arange(self.n_appliances, device=x.device)
        emb     = self.appliance_emb(app_idx)             # (n_apps, hidden)

        # τ_app scales each embedding: biases which temporal patterns each
        # appliance attends to (functionally: per-appliance time constant bias)
        queries = torch.sigmoid(self.tau_app(emb))        # (n_apps, hidden)
        queries = (emb * queries).unsqueeze(0).expand(B, -1, -1)  # (B, n_apps, hidden)

        context, _ = self.app_attention(queries, seq, seq)         # (B, n_apps, hidden)
        context    = self.norm_attn(context + emb.unsqueeze(0))    # residual

        z_cond = self.norm_z(F.relu(self.z_proj(context)))         # (B, n_apps, hidden)

        pred_power = torch.sigmoid(self.power_head(z_cond)).squeeze(-1)  # (B, n_apps)
        pred_event = torch.sigmoid(self.event_head(z_cond)).squeeze(-1)
        pred_trans = torch.sigmoid(self.trans_head(z_cond)).squeeze(-1)
        return pred_power, pred_event, pred_trans


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

    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.Y_power[i], self.Y_class[i], self.Y_trans[i], self.Agg[i]


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
    mains    = df['aggregate'].values.astype(np.float32)
    app_vals = {a: df[a].values.astype(np.float32) for a in APPLIANCES}
    X, Y_power, Agg = [], [], []
    for i in range(0, len(mains) - WIN, STRIDE):
        X.append(mains[i:i + WIN])
        mid = i + WIN // 2
        Y_power.append([app_vals[a][mid] for a in APPLIANCES])
        Agg.append(mains[mid])
    X       = np.array(X,       dtype=np.float32).reshape(-1, WIN, 1)
    Y_power = np.array(Y_power, dtype=np.float32)
    Agg     = np.array(Agg,     dtype=np.float32)
    Y_class = (Y_power > THRESHOLD).astype(np.float32)
    Y_trans = np.zeros_like(Y_class)
    Y_trans[1:] = (Y_class[1:] != Y_class[:-1]).astype(np.float32)
    return X, Y_power, Y_class, Y_trans, Agg


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _run_epoch(model, loader, optimizer, y_mins_t, y_ranges_t, device, train=True):
    model.train() if train else model.eval()
    tot = tot_pw = tot_ev = tot_sm = tot_co = 0.0
    all_pred_power, all_pred_event = [], []
    all_true_power, all_true_class = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb_pw, yb_cl, yb_tr, agg in loader:
            xb    = xb.to(device);    yb_pw = yb_pw.to(device)
            yb_cl = yb_cl.to(device); yb_tr = yb_tr.to(device)
            agg   = agg.to(device)

            pred_power, pred_event, pred_trans = model(xb)

            L_power = F.smooth_l1_loss(pred_power, yb_pw)
            L_event = focal_bce(pred_event, yb_cl)
            L_trans = focal_bce(pred_trans, yb_tr)

            # Improvement 4: energy conservation — penalise excess aggregate
            pred_raw = pred_power * y_ranges_t + y_mins_t
            L_cons   = F.relu(pred_raw.sum(dim=1) - agg * 1.1).mean()

            loss = (W_POWER * L_power + W_EVENT * L_event
                    + W_TRANS * L_trans + W_CONS * L_cons)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            tot    += loss.item();     tot_pw += L_power.item()
            tot_ev += L_event.item();  tot_sm += L_trans.item()
            tot_co += L_cons.item()
            all_pred_power.append(pred_power.detach().cpu().numpy())
            all_pred_event.append(pred_event.detach().cpu().numpy())
            all_true_power.append(yb_pw.cpu().numpy())
            all_true_class.append(yb_cl.cpu().numpy())

    n = len(loader)
    return (tot/n, tot_pw/n, tot_ev/n, tot_sm/n, tot_co/n,
            np.concatenate(all_pred_power), np.concatenate(all_pred_event),
            np.concatenate(all_true_power), np.concatenate(all_true_class))


def _metrics_per_appliance(pred_power_sc, pred_event_arr,
                           true_power_sc, true_class_arr, y_scalers):
    """
    Classification metrics (F1/P/R/TP/FP/TN/FN) from event head.
    Regression metrics (MAE/SAE) from power head (inverse-scaled).
    SAE uses WIN-sample chunked formula matching utils.calculate_nilm_metrics.
    """
    results = {}
    for i, app in enumerate(APPLIANCES):
        raw_pred = y_scalers[i].inverse_transform(
            pred_power_sc[:, i:i+1]).flatten()
        raw_true = y_scalers[i].inverse_transform(
            true_power_sc[:, i:i+1]).flatten()

        mae          = float(np.abs(raw_pred - raw_true).mean())
        energy_error = abs(raw_pred.sum() - raw_true.sum())
        sae_abs      = float(energy_error)
        sae_ratio    = float(energy_error / (raw_true.sum() + 1e-8))
        n_chunks     = len(raw_pred) // WIN
        if n_chunks > 0:
            pc = raw_pred[:n_chunks * WIN].reshape(n_chunks, WIN)
            tc = raw_true[:n_chunks * WIN].reshape(n_chunks, WIN)
            sae_avg = float(np.abs(pc.sum(1) - tc.sum(1)).sum() / (n_chunks * WIN))
        else:
            sae_avg = float(energy_error / max(len(raw_pred), 1))

        pred_on = pred_event_arr[:, i] > CLASS_THRESHOLDS[app]
        true_on = true_class_arr[:, i] > 0.5
        tp = int(( pred_on &  true_on).sum())
        fp = int(( pred_on & ~true_on).sum())
        tn = int((~pred_on & ~true_on).sum())
        fn = int((~pred_on &  true_on).sum())
        prec  = tp / (tp + fp + 1e-8)
        rec   = tp / (tp + fn + 1e-8)
        f1    = 2.0 * prec * rec / (prec + rec + 1e-8)

        results[app] = {
            'f1': float(f1), 'precision': float(prec), 'recall': float(rec),
            'mae': mae, 'sae_abs': sae_abs, 'sae_avg': sae_avg, 'sae_ratio': sae_ratio,
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        }
    return results


def _print_metrics(label, metrics):
    print(f"  {label}")
    print(f"    {'App':<22} {'F1':>6} {'P':>6} {'R':>6} "
          f"{'MAE':>7} {'SAE_avg':>9} {'SAE_%':>7} {'TP':>6} {'FP':>6} {'TN':>6} {'FN':>6}")
    for app, m in metrics.items():
        print(f"    {app:<22} {m['f1']:>6.4f} {m['precision']:>6.4f} "
              f"{m['recall']:>6.4f} {m['mae']:>7.2f} {m['sae_avg']:>9.2f} "
              f"{m['sae_ratio']:>7.4f} {m['TP']:>6} {m['FP']:>6} {m['TN']:>6} {m['FN']:>6}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_all(dataset_dir=DEFAULT_DATASET_DIR, hidden_size=HIDDEN_SIZE,
            n_heads=N_HEADS, dt=DT, save_dir=None):

    print("Loading fine_tuning_dataset splits...")
    splits    = load_splits(dataset_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if save_dir is None:
        save_dir = f'models/advanced_lnn_v2_finetune_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  hidden={hidden_size}  n_heads={n_heads}  dt={dt}")
    print(f"TCN: {TCN_LAYERS} layers, hidden={TCN_HIDDEN}, dilations=1/2/4")
    print(f"Focal: gamma={FOCAL_GAMMA}, alpha={FOCAL_ALPHA}")
    print(f"Loss weights: power={W_POWER}  event={W_EVENT}  "
          f"trans={W_TRANS}  cons={W_CONS}")

    wall_start = time.time()

    # ── Sequences ──────────────────────────────────────────────────────────
    X_pre, Yp_pre, Yc_pre, Yt_pre, Ag_pre = create_sequences(splits['pretrain'])
    X_val, Yp_val, Yc_val, Yt_val, Ag_val = create_sequences(splits['validation'])
    X_ft,  Yp_ft,  Yc_ft,  Yt_ft,  Ag_ft  = create_sequences(splits['finetune'])
    X_te,  Yp_te,  Yc_te,  Yt_te,  Ag_te  = create_sequences(splits['test'])

    print("\nClass balance (fraction ON per appliance):")
    print(f"  {'App':<22} {'pretrain':>10} {'val':>10} {'finetune':>10} {'test':>10}")
    for i, app in enumerate(APPLIANCES):
        print(f"  {app:<22} {Yc_pre[:,i].mean():>10.4f} {Yc_val[:,i].mean():>10.4f} "
              f"{Yc_ft[:,i].mean():>10.4f} {Yc_te[:,i].mean():>10.4f}")

    # ── Scale ──────────────────────────────────────────────────────────────
    xs = MinMaxScaler()
    X_pre = xs.fit_transform(X_pre.reshape(-1, 1)).reshape(X_pre.shape)
    X_val = xs.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
    X_ft  = xs.transform(X_ft.reshape(-1, 1)).reshape(X_ft.shape)
    X_te  = xs.transform(X_te.reshape(-1, 1)).reshape(X_te.shape)

    y_scalers = []
    for i in range(len(APPLIANCES)):
        ys = MinMaxScaler()
        Yp_pre[:, i:i+1] = ys.fit_transform(Yp_pre[:, i:i+1])
        Yp_val[:, i:i+1] = ys.transform(Yp_val[:, i:i+1])
        Yp_ft[:, i:i+1]  = ys.transform(Yp_ft[:, i:i+1])
        Yp_te[:, i:i+1]  = ys.transform(Yp_te[:, i:i+1])
        y_scalers.append(ys)

    y_mins_t   = torch.FloatTensor([s.data_min_[0]   for s in y_scalers]).to(device)
    y_ranges_t = torch.FloatTensor([s.data_range_[0] for s in y_scalers]).to(device)

    print(f"  Train: {X_pre.shape}  Val: {X_val.shape}  "
          f"FT: {X_ft.shape}  Test: {X_te.shape}")

    def mk_loader(X, Yp, Yc, Yt, Ag):
        return torch.utils.data.DataLoader(
            MultiTaskDataset(X, Yp, Yc, Yt, Ag),
            batch_size=BATCH, shuffle=False)

    pre_loader = mk_loader(X_pre, Yp_pre, Yc_pre, Yt_pre, Ag_pre)
    val_loader = mk_loader(X_val, Yp_val, Yc_val, Yt_val, Ag_val)
    ft_loader  = mk_loader(X_ft,  Yp_ft,  Yc_ft,  Yt_ft,  Ag_ft)
    te_loader  = mk_loader(X_te,  Yp_te,  Yc_te,  Yt_te,  Ag_te)

    # ── Model ──────────────────────────────────────────────────────────────
    model = EventPowerLNN(
        input_size=1, hidden_size=hidden_size, n_appliances=len(APPLIANCES),
        n_heads=n_heads, dt=dt).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # ── Phase 1: Supervised Pretrain ───────────────────────────────────────
    print(f"\n{'='*60}\nPhase 1: Supervised Pretrain  ({EPOCHS} epochs max)\n{'='*60}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4)

    history   = {k: [] for k in ('train_loss', 'val_loss', 'train_power', 'train_event',
                                   'train_cons', 'val_power', 'val_event',
                                   'val_cons', 'val_metrics')}
    best_score = -float('inf');  best_state = None;  counter = 0
    pretrain_start = time.time()

    for epoch in range(EPOCHS):
        ep_start = time.time()
        tr = _run_epoch(model, pre_loader, optimizer, y_mins_t, y_ranges_t, device, True)
        va = _run_epoch(model, val_loader, None,      y_mins_t, y_ranges_t, device, False)
        scheduler.step(va[0])

        vm     = _metrics_per_appliance(va[5], va[6], va[7], va[8], y_scalers)
        avg_f1 = float(np.mean([vm[a]['f1']  for a in APPLIANCES]))
        avg_mae= float(np.mean([vm[a]['mae'] for a in APPLIANCES]))

        history['train_loss'].append(tr[0]);  history['val_loss'].append(va[0])
        history['train_power'].append(tr[1]); history['val_power'].append(va[1])
        history['train_event'].append(tr[2]); history['val_event'].append(va[2])
        history['train_cons'].append(tr[4]);  history['val_cons'].append(va[4])
        history['val_metrics'].append(vm)

        ep_time = time.time() - ep_start
        print(f"  Ep {epoch+1:3d}  "
              f"tr={tr[0]:.5f}(pw={tr[1]:.4f} ev={tr[2]:.4f} "
              f"sm={tr[3]:.4f} co={tr[4]:.4f})  "
              f"val={va[0]:.5f}  avgF1={avg_f1:.4f}  avgMAE={avg_mae:.2f}  "
              f"time={ep_time:.1f}s")
        for app in APPLIANCES:
            m = vm[app]
            print(f"    {app:<22}  F1={m['f1']:.4f}  P={m['precision']:.4f}  "
                  f"R={m['recall']:.4f}  MAE={m['mae']:.2f}  "
                  f"SAE_avg={m['sae_avg']:.2f}W  "
                  f"TP={m['TP']}  FP={m['FP']}  TN={m['TN']}  FN={m['FN']}")

        score = avg_f1   # checkpoint by F1 — NILM papers optimise classification first
        if score > best_score:
            best_score = score;  counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            save_model(model,
                       {'input_size': 1, 'hidden_size': hidden_size,
                        'n_heads': n_heads, 'dt': dt},
                       {'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE},
                       {'avg_f1': avg_f1, 'avg_mae': avg_mae},
                       os.path.join(save_dir, 'pretrain_best.pth'))
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}");  break

    print(f"  Phase 1 total: {(time.time()-pretrain_start)/60:.1f} min")
    model.load_state_dict(best_state)

    pre_ft = _run_epoch(model, te_loader, None, y_mins_t, y_ranges_t, device, False)
    pre_ft_met = _metrics_per_appliance(pre_ft[5], pre_ft[6], pre_ft[7], pre_ft[8], y_scalers)
    _print_metrics("Test BEFORE fine-tune:", pre_ft_met)

    # ── Phase 2: Fine-tune (frozen encoder) ────────────────────────────────
    print(f"\n{'='*60}\nPhase 2: Fine-tune  ({EPOCHS_FT} epochs max)\n{'='*60}")
    for p in model.parameters():
        p.requires_grad = False
    for module in [model.appliance_emb, model.tau_app,
                   model.power_head, model.event_head, model.trans_head]:
        for p in module.parameters():
            p.requires_grad = True

    ft_opt        = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_FT)
    best_ft_score = -float('inf');  best_ft_state = None;  ft_counter = 0
    ft_history    = {'train_loss': []}
    ft_start      = time.time()

    for epoch in range(EPOCHS_FT):
        ep_start = time.time()
        tr = _run_epoch(model, ft_loader, ft_opt, y_mins_t, y_ranges_t, device, True)
        ft_history['train_loss'].append(tr[0])

        ft_m   = _metrics_per_appliance(tr[5], tr[6], tr[7], tr[8], y_scalers)
        avg_f1 = float(np.mean([ft_m[a]['f1']  for a in APPLIANCES]))
        avg_mae= float(np.mean([ft_m[a]['mae'] for a in APPLIANCES]))
        score  = avg_f1

        ep_time = time.time() - ep_start
        print(f"  FT Ep {epoch+1:2d}  loss={tr[0]:.5f}  "
              f"(pw={tr[1]:.4f} ev={tr[2]:.4f} co={tr[4]:.4f})  "
              f"avgF1={avg_f1:.4f}  avgMAE={avg_mae:.2f}  time={ep_time:.1f}s")

        if score > best_ft_score:
            best_ft_score = score;  ft_counter = 0
            best_ft_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            ft_counter += 1
            if ft_counter >= PATIENCE_FT:
                print(f"  FT early stopping at epoch {epoch+1}");  break

    print(f"  Phase 2 total: {(time.time()-ft_start)/60:.1f} min")
    model.load_state_dict(best_ft_state)

    # ── Phase 3: Test ──────────────────────────────────────────────────────
    te       = _run_epoch(model, te_loader, None, y_mins_t, y_ranges_t, device, False)
    test_met = _metrics_per_appliance(te[5], te[6], te[7], te[8], y_scalers)
    _print_metrics("Test AFTER fine-tune:", test_met)

    # ── Plots ──────────────────────────────────────────────────────────────
    ep = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(20, 4))
    for i, (tr_k, va_k, title, color) in enumerate([
        ('train_loss',  'val_loss',  'Total Loss',   'blue'),
        ('train_power', 'val_power', 'Power Reg',    'steelblue'),
        ('train_event', 'val_event', 'Event Focal',  'orange'),
        ('train_cons',  'val_cons',  'Conservation', 'purple'),
    ], 1):
        plt.subplot(1, 4, i)
        plt.plot(ep, history[tr_k], label='Train', color=color)
        plt.plot(ep, history[va_k], label='Val',   color=color,
                 linestyle='--', alpha=0.7)
        plt.title(title);  plt.xlabel('Epoch');  plt.legend();  plt.grid(alpha=0.3)
    plt.suptitle('Advanced LNN v2 — Loss Components', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pretrain_loss.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(len(APPLIANCES), 2, figsize=(12, 4 * len(APPLIANCES)))
    fig.suptitle('Advanced LNN v2 — Per-Appliance Val Metrics', fontsize=13)
    for row, app in enumerate(APPLIANCES):
        f1s  = [m[app]['f1']  for m in history['val_metrics']]
        maes = [m[app]['mae'] for m in history['val_metrics']]
        for col, vals, label, y_pre, y_post, color in [
            (0, f1s,  'F1',     pre_ft_met[app]['f1'],  test_met[app]['f1'],  'steelblue'),
            (1, maes, 'MAE(W)', pre_ft_met[app]['mae'], test_met[app]['mae'], 'red'),
        ]:
            axes[row][col].plot(ep, vals, color=color)
            axes[row][col].axhline(y_pre,  color='steelblue',   linestyle='--', label='Test pre-FT')
            axes[row][col].axhline(y_post, color='darkorange',  linestyle='--', label='Test post-FT')
            axes[row][col].set_title(f'{app} — {label}')
            axes[row][col].legend();  axes[row][col].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_appliance_metrics.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    ft_ep = range(1, len(ft_history['train_loss']) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(ft_ep, ft_history['train_loss'], color='green')
    plt.title('Phase 2: Fine-tune Loss');  plt.xlabel('FT Epoch')
    plt.grid(alpha=0.3);  plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'finetune_loss.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── JSON summary ───────────────────────────────────────────────────────
    def _floatify(d):
        return {k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
                for k, v in d.items()}

    config = {
        'model': 'EventPowerLNN (Advanced LNN v2)',
        'dataset': 'fine_tuning_dataset',
        'improvements': [
            'multi_resolution_tcn (3 layers, dilations=1/2/4, hidden=32)',
            'appliance_aware_tau (tau_app * appliance_embedding as attention query)',
            'hybrid_event_power_heads (focal_bce event + smooth_l1 power)',
            'energy_conservation_loss (relu(sum_pred - agg*1.1).mean, w=0.05)',
        ],
        'model_params': {'hidden_size': hidden_size, 'n_heads': n_heads, 'dt': dt,
                         'tcn_hidden': TCN_HIDDEN, 'tcn_layers': TCN_LAYERS,
                         'n_params': n_params},
        'loss_weights': {'power': W_POWER, 'event': W_EVENT,
                         'trans': W_TRANS, 'cons': W_CONS},
        'focal': {'gamma': FOCAL_GAMMA, 'alpha': FOCAL_ALPHA},
        'pretrain_params': {'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE},
        'finetune_params': {'lr': LR_FT, 'epochs': EPOCHS_FT, 'patience': PATIENCE_FT},
        'test_metrics_before_finetune': {a: _floatify(m) for a, m in pre_ft_met.items()},
        'test_metrics_after_finetune':  {a: _floatify(m) for a, m in test_met.items()},
    }
    with open(os.path.join(save_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    print(f"\nAdvanced LNN v2 complete.  Results → {save_dir}")
    print(f"\n  {'App':<22} {'F1 pre':>8} {'F1 post':>8}  {'MAE pre':>8} {'MAE post':>8}")
    for app in APPLIANCES:
        print(f"  {app:<22} "
              f"{pre_ft_met[app]['f1']:>8.4f} {test_met[app]['f1']:>8.4f}  "
              f"{pre_ft_met[app]['mae']:>8.1f} {test_met[app]['mae']:>8.1f}")
    print(f"\nTotal wall-clock time: {(time.time()-wall_start)/60:.1f} min")
    return test_met, pre_ft_met


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Advanced LNN v2: event+power heads, appliance-aware tau, conservation loss')
    p.add_argument('--dataset-dir',  default=DEFAULT_DATASET_DIR)
    p.add_argument('--hidden-size',  type=int,   default=HIDDEN_SIZE)
    p.add_argument('--n-heads',      type=int,   default=N_HEADS)
    p.add_argument('--dt',           type=float, default=DT)
    p.add_argument('--save-dir',     default=None)
    args = p.parse_args()
    run_all(args.dataset_dir, args.hidden_size, args.n_heads, args.dt, args.save_dir)
