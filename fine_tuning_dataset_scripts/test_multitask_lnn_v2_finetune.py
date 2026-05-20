"""
Multi-Task LNN v2 — Cross-House Fine-Tuning on fine_tuning_dataset/

Four improvements over v1 (test_multitask_lnn_finetune.py):

  Improvement 1 — Focal BCE
    L_cls = -α(1-p_t)^γ log(p_t),  γ=2.0, α=0.25
    NILM is heavily imbalanced (appliances OFF most of the time).
    Focal loss down-weights easy negatives and focuses on hard positives,
    typically boosting F1 substantially on rare ON states.

  Improvement 2 — Appliance Embeddings
    Instead of n_appliances separate linear heads, learn e_i ∈ R^hidden
    for each appliance.  All appliances share the same MLP heads, but
    receive different conditioning:  z_i = z + e_i.
    Shared representations → fewer parameters, better generalisation.

  Improvement 3 — Dilated TCN Frontend
    x(B,T,1) → DilatedTCN(dilations=1,2,4) → (B,T,32) → LNN → ...
    TCN captures local transients (sharp ON/OFF edges) at multiple
    temporal scales before the LNN integrates long-range dynamics.

  Improvement 4 — Masked Pretraining (BERT-style, Phase 0)
    Phase 0: Randomly mask 15% of aggregate timesteps with a learned
    mask token; reconstruct originals via MSE on masked positions only.
    Self-supervised — requires no appliance labels.  Gives the encoder
    rich multi-scale representations before supervised training begins.

Phase structure:
    Phase 0 — Masked pretrain   : aggregate only, no appliance labels
    Phase 1 — Supervised pretrain : House 1  (pretrain + validation splits)
    Phase 2 — Fine-tune         : House 5 first 2h   (finetune split)
    Phase 3 — Test              : House 5 remaining 22h (test split)
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
from utils import save_model


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_DATASET_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'fine_tuning_dataset')

APPLIANCES = ['dishwasher', 'fridge', 'microwave', 'washing_machine']
THRESHOLD  = 10.0
WIN = 100;  STRIDE = 5;  BATCH = 32

# Phase 0: masked pretraining
MASK_EPOCHS = 10;  MASK_LR = 5e-4;  MASK_RATIO = 0.15

# Phase 1: supervised pretraining
EPOCHS = 80;  PATIENCE = 20;  LR = 1e-3

# Phase 2: fine-tuning
EPOCHS_FT = 30;  PATIENCE_FT = 10;  LR_FT = 1e-4

# House 5 interleaved re-split (prevents all-ON/OFF pathological splits)
FT_CHUNK_MINUTES = 5    # per cycle: assigned to fine-tuning
TE_CHUNK_MINUTES = 30   # per cycle: assigned to test

# Loss weights
W_REG = 1.0;  W_CLS = 0.5;  W_SMOOTH = 0.1;  W_CONS = 0.05

# Improvement 1: focal loss
FOCAL_GAMMA = 2.0;  FOCAL_ALPHA = 0.75   # high alpha → up-weights rare ON samples

# Per-appliance classification thresholds (tuned for typical ON-rate differences)
CLASS_THRESHOLDS = {
    'dishwasher':      0.35,
    'fridge':          0.65,
    'microwave':       0.25,
    'washing_machine': 0.40,
}

# Improvement 3: TCN
TCN_HIDDEN = 32;  TCN_LAYERS = 3   # dilations: 1, 2, 4


# ---------------------------------------------------------------------------
# Improvement 1: Focal BCE
# ---------------------------------------------------------------------------

def focal_bce(pred, target):
    """Focal loss with per-sample alpha: FOCAL_ALPHA for ON, 1-FOCAL_ALPHA for OFF."""
    eps     = 1e-6
    pred    = pred.clamp(eps, 1.0 - eps)
    bce     = F.binary_cross_entropy(pred, target, reduction='none')
    p_t     = torch.where(target == 1.0, pred, 1.0 - pred)
    alpha_t = torch.where(
        target == 1.0,
        torch.full_like(target, FOCAL_ALPHA),
        torch.full_like(target, 1.0 - FOCAL_ALPHA),
    )
    return (alpha_t * (1.0 - p_t).pow(FOCAL_GAMMA) * bce).mean()


# ---------------------------------------------------------------------------
# Improvement 3: Dilated TCN Frontend
# ---------------------------------------------------------------------------

class DilatedTCN(nn.Module):
    """3-layer dilated temporal convolution; preserves sequence length."""
    def __init__(self, in_channels=1, hidden=TCN_HIDDEN, n_layers=TCN_LAYERS):
        super().__init__()
        layers = []
        ch = in_channels
        for i in range(n_layers):
            d = 2 ** i
            layers += [
                nn.Conv1d(ch, hidden, kernel_size=3, padding=d, dilation=d),
                nn.ReLU(),
                nn.BatchNorm1d(hidden),
            ]
            ch = hidden
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, T, in_channels) → (B, T, hidden)
        return self.net(x.permute(0, 2, 1)).permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LNNMultiTaskModelV2(nn.Module):
    """
    DilatedTCN → LNN encoder → Attention → z(t)
    → appliance-embedding-conditioned shared heads.
    Reconstruction head supports masked pretraining (Phase 0).
    """

    def __init__(self, input_size=1, hidden_size=64, n_appliances=4,
                 n_heads=2, dt=0.1, tcn_hidden=TCN_HIDDEN):
        super().__init__()
        self.hidden_size  = hidden_size
        self.n_appliances = n_appliances
        self.dt           = dt

        # Improvement 4: learned mask token (scalar, in normalised input space)
        self.mask_token = nn.Parameter(torch.zeros(1))

        # Improvement 3: TCN frontend
        self.tcn = DilatedTCN(in_channels=input_size, hidden=tcn_hidden)

        # ── LNN Encoder (input from TCN, size tcn_hidden) ─────────────────
        self.input_proj  = nn.Linear(tcn_hidden, hidden_size)
        self.tau_base    = nn.Parameter(torch.ones(hidden_size))
        self.tau_mod     = nn.Linear(tcn_hidden, hidden_size)
        self.rec_weights = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.rec_weights)
        self.gate        = nn.Linear(tcn_hidden + hidden_size, hidden_size)
        self.norm_enc    = nn.LayerNorm(hidden_size)

        # ── Lightweight Attention ─────────────────────────────────────────
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=n_heads,
            dropout=0.1, batch_first=True)
        self.norm_attn = nn.LayerNorm(hidden_size)

        # ── Shared Latent ─────────────────────────────────────────────────
        self.z_proj = nn.Linear(hidden_size, hidden_size)
        self.norm_z  = nn.LayerNorm(hidden_size)

        # Improvement 2: appliance embeddings
        self.appliance_emb = nn.Embedding(n_appliances, hidden_size)

        # Shared heads conditioned on z_i = z + e_i  (output size 1, not n_apps)
        self.power_head = nn.Linear(hidden_size, 1)
        self.class_head = nn.Linear(hidden_size, 1)
        self.trans_head = nn.Linear(hidden_size, 1)

        # Improvement 4: reconstruction head for masked pretraining
        self.recon_head = nn.Linear(hidden_size, 1)

    def _encode_sequence(self, x):
        """x: (B, T, 1) → seq: (B, T, hidden)."""
        tcn_out = self.tcn(x)              # (B, T, tcn_hidden)
        B, T, _ = tcn_out.size()
        h = torch.zeros(B, self.hidden_size, device=x.device)
        states = []
        for t in range(T):
            xt    = tcn_out[:, t, :]
            inp   = self.input_proj(xt)
            rec   = torch.matmul(h, self.rec_weights)
            tau_b = F.softplus(self.tau_base).unsqueeze(0)
            tau_m = torch.sigmoid(self.tau_mod(xt))
            tau   = (tau_b * tau_m).clamp(min=self.dt)
            gate  = torch.sigmoid(self.gate(torch.cat([xt, h], dim=1)))
            f_t   = torch.tanh(inp + rec)
            dh    = ((-h / tau) + gate * f_t) * self.dt
            h     = (h + dh).clamp(-10.0, 10.0)
            states.append(h)
        return self.norm_enc(torch.stack(states, dim=1))   # (B, T, hidden)

    def _context_and_z(self, seq):
        query      = seq[:, -1:, :]
        context, _ = self.attention(query, seq, seq)
        context    = self.norm_attn(context.squeeze(1) + seq[:, -1, :])
        return self.norm_z(F.relu(self.z_proj(context)))   # (B, hidden)

    def forward(self, x):
        seq = self._encode_sequence(x)
        z   = self._context_and_z(seq)                     # (B, hidden)

        # Improvement 2: z_i = z + appliance_embedding_i  → (B, n_apps, hidden)
        app_idx = torch.arange(self.n_appliances, device=x.device)
        emb     = self.appliance_emb(app_idx)              # (n_apps, hidden)
        z_cond  = z.unsqueeze(1) + emb.unsqueeze(0)        # (B, n_apps, hidden)

        pred_power = torch.sigmoid(self.power_head(z_cond)).squeeze(-1)  # (B, n_apps)
        pred_class = torch.sigmoid(self.class_head(z_cond)).squeeze(-1)
        pred_trans = torch.sigmoid(self.trans_head(z_cond)).squeeze(-1)
        return pred_power, pred_class, pred_trans


# ---------------------------------------------------------------------------
# Dataset  (identical to v1)
# ---------------------------------------------------------------------------

class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y_power, Y_class, Y_trans, Agg):
        self.X       = torch.FloatTensor(X)
        self.Y_power = torch.FloatTensor(Y_power)
        self.Y_class = torch.FloatTensor(Y_class)
        self.Y_trans = torch.FloatTensor(Y_trans)
        self.Agg     = torch.FloatTensor(Agg)

    def __len__(self):        return len(self.X)
    def __getitem__(self, i):
        return (self.X[i], self.Y_power[i], self.Y_class[i],
                self.Y_trans[i], self.Agg[i])


# ---------------------------------------------------------------------------
# Data helpers  (identical to v1)
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
        X.append(mains[i: i + WIN])
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


def _resplit_house5(ft_df, te_df):
    """
    Concatenate House 5 finetune+test splits then re-split with interleaved chunks.

    Each cycle: FT_CHUNK_MINUTES → fine-tuning, TE_CHUNK_MINUTES → test.
    Ensures every appliance's ON/OFF events appear in both splits, preventing
    the degenerate case where one appliance is always ON (or always OFF) in a split.
    """
    house5    = pd.concat([ft_df, te_df], ignore_index=True)
    X, Yp, Yc, Yt, Ag = create_sequences(house5)
    N         = len(X)
    cycle_seq = max(1, int((FT_CHUNK_MINUTES + TE_CHUNK_MINUTES) * 60 / STRIDE))
    ft_seq    = max(1, int(FT_CHUNK_MINUTES * 60 / STRIDE))
    ft_idx, te_idx = [], []
    for start in range(0, N, cycle_seq):
        for j in range(start, min(start + cycle_seq, N)):
            if j - start < ft_seq:
                ft_idx.append(j)
            else:
                te_idx.append(j)
    print(f"  House 5 re-split: {len(ft_idx)} FT windows, {len(te_idx)} test windows "
          f"({FT_CHUNK_MINUTES}min FT / {TE_CHUNK_MINUTES}min test per cycle)")
    return (X[ft_idx], Yp[ft_idx], Yc[ft_idx], Yt[ft_idx], Ag[ft_idx],
            X[te_idx], Yp[te_idx], Yc[te_idx], Yt[te_idx], Ag[te_idx])


# ---------------------------------------------------------------------------
# Improvement 4: Masked pretraining epoch
# ---------------------------------------------------------------------------

def _run_mask_epoch(model, loader, optimizer, device, train=True):
    """Phase 0: reconstruct randomly masked aggregate positions."""
    model.train() if train else model.eval()
    tot = 0.0;  n = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, _, _, _, _ in loader:
            xb = xb.to(device)                           # (B, T, 1), normalised
            mask   = torch.rand(xb.shape[0], xb.shape[1], device=device) < MASK_RATIO
            target = xb.clone()

            # Differentiable replacement — keeps mask_token in compute graph
            mask3d    = mask.unsqueeze(-1).float()       # (B, T, 1)
            xb_masked = xb * (1.0 - mask3d) + model.mask_token * mask3d

            seq    = model._encode_sequence(xb_masked)  # (B, T, hidden)
            recon  = model.recon_head(seq)               # (B, T, 1)

            masked_count = mask3d.sum()
            if masked_count > 0:
                loss = (((recon - target) ** 2) * mask3d).sum() / masked_count
            else:
                loss = recon.new_tensor(0.0)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            tot += loss.item();  n += 1
    return tot / max(n, 1)


# ---------------------------------------------------------------------------
# Supervised training helpers
# ---------------------------------------------------------------------------

def _run_epoch(model, loader, optimizer, y_mins_t, y_ranges_t, device, train=True):
    """Phase 1/2: multi-task supervised training with focal BCE."""
    model.train() if train else model.eval()
    tot = tot_reg = tot_cls = tot_smooth = tot_cons = 0.0
    all_pred_power, all_pred_class = [], []
    all_true_power, all_true_class = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb_pw, yb_cl, yb_tr, agg in loader:
            xb    = xb.to(device)
            yb_pw = yb_pw.to(device)
            yb_cl = yb_cl.to(device)
            yb_tr = yb_tr.to(device)
            agg   = agg.to(device)

            pred_power, pred_class, pred_trans = model(xb)

            L_reg = F.smooth_l1_loss(pred_power, yb_pw)
            L_cls = focal_bce(pred_class, yb_cl)         # Improvement 1

            if pred_power.shape[0] > 1:
                L_tv = F.l1_loss(pred_power[1:], pred_power[:-1].detach())
            else:
                L_tv = pred_power.new_tensor(0.0)
            L_trans  = focal_bce(pred_trans, yb_tr)      # focal for transitions too
            L_smooth = 0.5 * L_tv + 0.5 * L_trans

            pred_raw = pred_power * y_ranges_t + y_mins_t
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
            all_pred_power.append(pred_power.detach().cpu().numpy())
            all_pred_class.append(pred_class.detach().cpu().numpy())
            all_true_power.append(yb_pw.cpu().numpy())
            all_true_class.append(yb_cl.cpu().numpy())

    n = len(loader)
    return (tot/n, tot_reg/n, tot_cls/n, tot_smooth/n, tot_cons/n,
            np.concatenate(all_pred_power), np.concatenate(all_pred_class),
            np.concatenate(all_true_power), np.concatenate(all_true_class))


def _metrics_per_appliance(pred_power_sc, pred_class_arr,
                           true_power_sc, true_class_arr, y_scalers):
    """
    F1 / P / R / TP / FP / TN / FN  ← classification head  (pred_class > 0.5)
    MAE / SAE                         ← regression head     (inverse-scaled power)
    """
    results = {}
    for i, app in enumerate(APPLIANCES):
        # Regression metrics
        raw_pred = y_scalers[i].inverse_transform(
            pred_power_sc[:, i:i+1]).flatten()
        raw_true = y_scalers[i].inverse_transform(
            true_power_sc[:, i:i+1]).flatten()
        mae       = float(np.abs(raw_pred - raw_true).mean())
        sae_abs   = float(abs(raw_pred.sum() - raw_true.sum()))
        sae_ratio = float(abs(raw_pred.sum() - raw_true.sum()) / (raw_true.sum() + 1e-8))

        # Classification metrics from the dedicated class head
        pred_on = pred_class_arr[:, i] > CLASS_THRESHOLDS[app]
        true_on = true_class_arr[:, i] > 0.5
        tp = int(( pred_on &  true_on).sum())
        fp = int(( pred_on & ~true_on).sum())
        tn = int((~pred_on & ~true_on).sum())
        fn = int((~pred_on &  true_on).sum())
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2.0 * precision * recall / (precision + recall + 1e-8)

        results[app] = {
            'f1': float(f1), 'precision': float(precision), 'recall': float(recall),
            'mae': mae, 'sae_abs': sae_abs, 'sae_ratio': sae_ratio,
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
        }
    return results


def _print_metrics(label, metrics):
    print(f"  {label}")
    print(f"    {'App':<22} {'F1':>6} {'P':>6} {'R':>6} "
          f"{'MAE':>7} {'SAE_abs':>10} {'SAE_%':>7} {'TP':>6} {'FP':>6} {'TN':>6} {'FN':>6}")
    for app, m in metrics.items():
        print(f"    {app:<22} {m['f1']:>6.4f} {m['precision']:>6.4f} "
              f"{m['recall']:>6.4f} {m['mae']:>7.2f} {m['sae_abs']:>10.1f} "
              f"{m['sae_ratio']:>7.4f} {m['TP']:>6} {m['FP']:>6} {m['TN']:>6} {m['FN']:>6}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_all(dataset_dir=DEFAULT_DATASET_DIR, hidden_size=64, n_heads=2,
            dt=0.1, save_dir=None):

    print("Loading fine_tuning_dataset splits...")
    splits    = load_splits(dataset_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if save_dir is None:
        save_dir = f'models/multitask_lnn_v2_finetune_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  hidden={hidden_size}  n_heads={n_heads}  dt={dt}")
    print(f"TCN: {TCN_LAYERS} layers, hidden={TCN_HIDDEN}, dilations=1/2/4")
    print(f"Focal: gamma={FOCAL_GAMMA}, alpha={FOCAL_ALPHA}")
    print(f"Mask: ratio={MASK_RATIO}, epochs={MASK_EPOCHS}")
    print(f"Loss weights: reg={W_REG}  cls={W_CLS}  smooth={W_SMOOTH}  cons={W_CONS}")

    wall_start = time.time()

    # ── Sequences ─────────────────────────────────────────────────────────
    X_pre, Yp_pre, Yc_pre, Yt_pre, Ag_pre = create_sequences(splits['pretrain'])
    X_val, Yp_val, Yc_val, Yt_val, Ag_val = create_sequences(splits['validation'])
    (X_ft, Yp_ft, Yc_ft, Yt_ft, Ag_ft,
     X_te, Yp_te, Yc_te, Yt_te, Ag_te) = _resplit_house5(
        splits['finetune'], splits['test'])

    print("\nClass balance (fraction ON per appliance):")
    print(f"  {'App':<22} {'pretrain':>10} {'val':>10} {'finetune':>10} {'test':>10}")
    for i, app in enumerate(APPLIANCES):
        print(f"  {app:<22} {Yc_pre[:, i].mean():>10.4f} {Yc_val[:, i].mean():>10.4f} "
              f"{Yc_ft[:, i].mean():>10.4f} {Yc_te[:, i].mean():>10.4f}")

    # ── Scale ─────────────────────────────────────────────────────────────
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

    y_mins   = np.array([float(s.data_min_[0])   for s in y_scalers])
    y_ranges = np.array([float(s.data_range_[0]) for s in y_scalers])
    y_mins_t   = torch.FloatTensor(y_mins).to(device)
    y_ranges_t = torch.FloatTensor(y_ranges).to(device)

    print(f"  Train: {X_pre.shape}  Val: {X_val.shape}  "
          f"FT: {X_ft.shape}  Test: {X_te.shape}")

    def mk_loader(X, Yp, Yc, Yt, Ag):
        return torch.utils.data.DataLoader(
            MultiTaskDataset(X, Yp, Yc, Yt, Ag),
            batch_size=BATCH, shuffle=False)   # no shuffle: TV loss needs order

    pre_loader = mk_loader(X_pre, Yp_pre, Yc_pre, Yt_pre, Ag_pre)
    val_loader = mk_loader(X_val, Yp_val, Yc_val, Yt_val, Ag_val)
    ft_loader  = mk_loader(X_ft,  Yp_ft,  Yc_ft,  Yt_ft,  Ag_ft)
    te_loader  = mk_loader(X_te,  Yp_te,  Yc_te,  Yt_te,  Ag_te)

    # ── Model ─────────────────────────────────────────────────────────────
    model = LNNMultiTaskModelV2(
        input_size=1, hidden_size=hidden_size, n_appliances=len(APPLIANCES),
        n_heads=n_heads, dt=dt).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # ── Phase 0: Masked Pretraining ───────────────────────────────────────
    print(f"\n{'='*60}\nPhase 0: Masked Pretraining  ({MASK_EPOCHS} epochs)\n{'='*60}")
    mask_opt     = torch.optim.Adam(model.parameters(), lr=MASK_LR)
    mask_history = []
    mask_start   = time.time()

    for epoch in range(MASK_EPOCHS):
        ep_start = time.time()
        loss = _run_mask_epoch(model, pre_loader, mask_opt, device, train=True)
        mask_history.append(loss)
        ep_time = time.time() - ep_start
        print(f"  Mask Ep {epoch+1:2d}/{MASK_EPOCHS}  recon_loss={loss:.6f}  "
              f"time={ep_time:.1f}s")
    print(f"  Phase 0 total: {(time.time()-mask_start)/60:.1f} min")

    # ── Phase 1: Supervised Pretrain ──────────────────────────────────────
    print(f"\n{'='*60}\nPhase 1: Supervised Pretrain  ({EPOCHS} epochs max)\n{'='*60}")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4)

    history = {k: [] for k in
               ('train_loss', 'val_loss', 'train_reg', 'train_cls',
                'train_smooth', 'train_cons',
                'val_reg', 'val_cls', 'val_smooth', 'val_cons', 'val_metrics')}
    best_score = -float('inf');  best_state = None;  counter = 0
    pretrain_start = time.time()

    for epoch in range(EPOCHS):
        ep_start = time.time()
        tr = _run_epoch(model, pre_loader, optimizer, y_mins_t, y_ranges_t, device, True)
        va = _run_epoch(model, val_loader, None,      y_mins_t, y_ranges_t, device, False)
        scheduler.step(va[0])

        vm     = _metrics_per_appliance(va[5], va[6], va[7], va[8], y_scalers)
        avg_f1 = np.mean([vm[a]['f1']  for a in APPLIANCES])
        avg_mae= np.mean([vm[a]['mae'] for a in APPLIANCES])

        history['train_loss'].append(tr[0]);   history['val_loss'].append(va[0])
        history['train_reg'].append(tr[1]);    history['val_reg'].append(va[1])
        history['train_cls'].append(tr[2]);    history['val_cls'].append(va[2])
        history['train_smooth'].append(tr[3]); history['val_smooth'].append(va[3])
        history['train_cons'].append(tr[4]);   history['val_cons'].append(va[4])
        history['val_metrics'].append(vm)

        ep_time = time.time() - ep_start
        print(f"  Ep {epoch+1:3d}  "
              f"tr={tr[0]:.5f}(reg={tr[1]:.4f} cls={tr[2]:.4f} "
              f"sm={tr[3]:.4f} cs={tr[4]:.4f})  "
              f"val={va[0]:.5f}  avgF1={avg_f1:.4f}  avgMAE={avg_mae:.2f}  "
              f"time={ep_time:.1f}s")
        for app in APPLIANCES:
            m = vm[app]
            print(f"    {app:<22}  F1={m['f1']:.4f}  P={m['precision']:.4f}  "
                  f"R={m['recall']:.4f}  MAE={m['mae']:.2f}  "
                  f"SAE_abs={m['sae_abs']:.1f}W  SAE%={m['sae_ratio']:.4f}  "
                  f"TP={m['TP']}  FP={m['FP']}  TN={m['TN']}  FN={m['FN']}")

        score = avg_f1 - 0.001 * avg_mae
        if score > best_score:
            best_score = score;  counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            save_model(model,
                       {'input_size': 1, 'hidden_size': hidden_size,
                        'n_heads': n_heads, 'dt': dt},
                       {'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE},
                       {'avg_f1': float(avg_f1), 'avg_mae': float(avg_mae)},
                       os.path.join(save_dir, 'pretrain_best.pth'))
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}");  break

    print(f"  Phase 1 total: {(time.time()-pretrain_start)/60:.1f} min")
    model.load_state_dict(best_state)

    pre_ft     = _run_epoch(model, te_loader, None, y_mins_t, y_ranges_t, device, False)
    pre_ft_met = _metrics_per_appliance(pre_ft[5], pre_ft[6], pre_ft[7], pre_ft[8], y_scalers)
    _print_metrics("Test BEFORE fine-tune:", pre_ft_met)

    # ── Phase 2: Fine-tune ────────────────────────────────────────────────
    print(f"\n{'='*60}\nPhase 2: Fine-tune  ({EPOCHS_FT} epochs max)\n{'='*60}")
    # Freeze shared encoder; adapt only output heads to the small fine-tune set
    for p in model.parameters():
        p.requires_grad = False
    for module in [model.appliance_emb, model.power_head,
                   model.class_head, model.trans_head]:
        for p in module.parameters():
            p.requires_grad = True

    ft_opt     = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_FT)
    best_ft    = float('inf');  best_ft_state = None;  ft_counter = 0
    ft_history = {'train_loss': []}
    ft_start   = time.time()

    for epoch in range(EPOCHS_FT):
        ep_start = time.time()
        tr = _run_epoch(model, ft_loader, ft_opt, y_mins_t, y_ranges_t, device, True)
        ft_history['train_loss'].append(tr[0])
        ep_time = time.time() - ep_start
        print(f"  FT Ep {epoch+1:2d}  loss={tr[0]:.5f}  "
              f"(reg={tr[1]:.4f} cls={tr[2]:.4f} sm={tr[3]:.4f} cs={tr[4]:.4f})  "
              f"time={ep_time:.1f}s")
        if tr[0] < best_ft:
            best_ft = tr[0];  ft_counter = 0
            best_ft_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            ft_counter += 1
            if ft_counter >= PATIENCE_FT:
                print(f"  FT early stopping at epoch {epoch+1}");  break

    print(f"  Phase 2 total: {(time.time()-ft_start)/60:.1f} min")
    model.load_state_dict(best_ft_state)

    # ── Phase 3: Test ─────────────────────────────────────────────────────
    te       = _run_epoch(model, te_loader, None, y_mins_t, y_ranges_t, device, False)
    test_met = _metrics_per_appliance(te[5], te[6], te[7], te[8], y_scalers)
    _print_metrics("Test AFTER fine-tune:", test_met)

    # ── Plots ─────────────────────────────────────────────────────────────
    # Phase 0: masked pretraining reconstruction loss
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(mask_history) + 1), mask_history, color='teal')
    plt.title('Phase 0: Masked Pretraining — Reconstruction MSE')
    plt.xlabel('Epoch');  plt.ylabel('MSE');  plt.grid(alpha=0.3);  plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mask_pretrain_loss.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Phase 1: 5-panel loss breakdown
    ep = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(22, 4))
    for i, (tr_key, va_key, title, color) in enumerate([
        ('train_loss',   'val_loss',   'Total Loss',   'blue'),
        ('train_reg',    'val_reg',    'Regression',   'steelblue'),
        ('train_cls',    'val_cls',    'Focal CLS',    'orange'),
        ('train_smooth', 'val_smooth', 'Smooth/Trans', 'green'),
        ('train_cons',   'val_cons',   'Conservation', 'purple'),
    ], 1):
        plt.subplot(1, 5, i)
        plt.plot(ep, history[tr_key], label='Train', color=color)
        plt.plot(ep, history[va_key], label='Val',   color=color,
                 linestyle='--', alpha=0.7)
        plt.title(title);  plt.xlabel('Epoch');  plt.legend();  plt.grid(alpha=0.3)
    plt.suptitle('Phase 1: Pretrain — Multi-Task LNN v2 Loss Components', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pretrain_loss.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Per-appliance F1/MAE curves
    fig, axes = plt.subplots(len(APPLIANCES), 2, figsize=(12, 4 * len(APPLIANCES)))
    fig.suptitle('Multi-Task LNN v2 — Per-Appliance Val Metrics', fontsize=13)
    for row, app in enumerate(APPLIANCES):
        f1s  = [m[app]['f1']  for m in history['val_metrics']]
        maes = [m[app]['mae'] for m in history['val_metrics']]
        for col, vals, label, ytest, color in [
            (0, f1s,  'F1',     test_met[app]['f1'],  'steelblue'),
            (1, maes, 'MAE(W)', test_met[app]['mae'], 'red'),
        ]:
            axes[row][col].plot(ep, vals, color=color)
            axes[row][col].axhline(
                pre_ft_met[app]['f1' if col == 0 else 'mae'],
                color='steelblue', linestyle='--', label='Test pre-FT')
            axes[row][col].axhline(
                ytest, color='darkorange', linestyle='--', label='Test post-FT')
            axes[row][col].set_title(f'{app} — {label}')
            axes[row][col].legend();  axes[row][col].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_appliance_metrics.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Fine-tune loss
    ft_ep = range(1, len(ft_history['train_loss']) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(ft_ep, ft_history['train_loss'], color='green')
    plt.title('Phase 2: Fine-tune Loss');  plt.xlabel('FT Epoch')
    plt.grid(alpha=0.3);  plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'finetune_loss.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── JSON summary ──────────────────────────────────────────────────────
    config = {
        'model': 'LNNMultiTaskModelV2',
        'dataset': 'fine_tuning_dataset',
        'improvements': [
            'focal_bce (gamma=2, alpha=0.25)',
            'appliance_embeddings (shared heads conditioned on z+e_i)',
            'dilated_tcn_frontend (3 layers, dilations=1/2/4, hidden=32)',
            'masked_pretraining (BERT-style, ratio=0.15, epochs=10)',
        ],
        'model_params': {
            'hidden_size': hidden_size, 'n_heads': n_heads, 'dt': dt,
            'tcn_hidden': TCN_HIDDEN, 'tcn_layers': TCN_LAYERS,
            'n_params': n_params,
        },
        'loss_weights': {'reg': W_REG, 'cls': W_CLS, 'smooth': W_SMOOTH, 'cons': W_CONS},
        'focal': {'gamma': FOCAL_GAMMA, 'alpha': FOCAL_ALPHA},
        'mask': {'ratio': MASK_RATIO, 'epochs': MASK_EPOCHS, 'lr': MASK_LR},
        'pretrain_params': {'lr': LR, 'epochs': EPOCHS, 'patience': PATIENCE},
        'finetune_params': {'lr': LR_FT, 'epochs': EPOCHS_FT, 'patience': PATIENCE_FT},
        'test_metrics_before_finetune': {
            app: {k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
                  for k, v in m.items()}
            for app, m in pre_ft_met.items()},
        'test_metrics_after_finetune': {
            app: {k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
                  for k, v in m.items()}
            for app, m in test_met.items()},
    }
    with open(os.path.join(save_dir, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    print(f"\nMulti-Task LNN v2 complete.  Results → {save_dir}")
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
        description='Multi-Task LNN v2 fine-tuning (4 improvements)')
    p.add_argument('--dataset-dir',  default=DEFAULT_DATASET_DIR)
    p.add_argument('--hidden-size',  type=int,   default=64)
    p.add_argument('--n-heads',      type=int,   default=2,
                   help='Attention heads (must divide hidden-size)')
    p.add_argument('--dt',           type=float, default=0.1)
    p.add_argument('--mask-epochs',  type=int,   default=MASK_EPOCHS,
                   help='Phase 0 masked pretraining epochs (default 10)')
    p.add_argument('--mask-ratio',   type=float, default=MASK_RATIO,
                   help='Fraction of timesteps to mask (default 0.15)')
    p.add_argument('--focal-gamma',  type=float, default=FOCAL_GAMMA,
                   help='Focal loss focusing parameter (default 2.0)')
    p.add_argument('--focal-alpha',  type=float, default=FOCAL_ALPHA,
                   help='Focal loss balance parameter (default 0.25)')
    args = p.parse_args()

    FOCAL_GAMMA = args.focal_gamma
    FOCAL_ALPHA = args.focal_alpha
    MASK_EPOCHS = args.mask_epochs
    MASK_RATIO  = args.mask_ratio

    run_all(args.dataset_dir, args.hidden_size, args.n_heads, args.dt)
