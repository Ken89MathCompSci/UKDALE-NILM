"""
analyse_representations.py
===========================
Post-hoc internal-representation analysis for FixedPINNLNN.

Analyses performed
------------------
1. Weight Analysis
   - CNN kernel singular-value spectra  (effective rank, information spread)
   - LiquidCell recurrent-weight eigenvalue spectrum  (stability, memory)
   - Power/event head norms per appliance  (relative confidence)
   - tau_base distribution  (time-constant distribution per hidden unit)

2. Activation / Feature-Space Analysis
   - CNN output statistics per split  (mean, std, kurtosis)
   - Hidden-state statistics per split
   - Representational drift: L2 distance between split centroids in CNN / hidden space
   - Centered Kernel Alignment (CKA) between splits

3. Prediction Distribution Analysis
   - Per-appliance prediction histograms across splits
   - Decision-boundary occupancy: what fraction of predictions fall in each
     threshold zone  (below train-thr, between train-thr and test-thr, above test-thr)
   - This directly explains the train/val→test F1 collapse for microwave & washing_machine

4. Linear Probing
   - Logistic regression on CNN output and on hidden states
   - Measures how linearly decodable each appliance's ON/OFF label is
   - Per-split probe accuracy reveals where the encoding breaks down

5. Gradient × Input Attribution
   - For FP cases (predicted ON, actually OFF) on the test split:
     which of the 8 input channels drove the false alarm?
   - Aggregated per appliance

Usage
-----
  python analyse_representations.py --model-dir models/fixed_pinn_lnn_TIMESTAMP

  Options:
    --dataset-dir  PATH     (default: dataset/)
    --hidden       INT      (default: 64, must match saved model)
    --n-probe      INT      number of windows to use for probing  (default: 2000)
    --device       cuda|cpu
"""

import sys, os, argparse, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, StandardScaler as SKStd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

# ── Import architecture and helpers from training script ─────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from fixed_pinn_lnn_ukdale import (
    APPLIANCES, AGG_COL, WIN, STRIDE, DATA_DIR,
    compute_adaptive_thresholds, compute_event_thresholds,
    compute_features, create_sequences,
    FixedPINNLNN, LiquidCell,
)
from sklearn.preprocessing import MinMaxScaler


# ============================================================================
# 1.  DATA LOADING & PREPROCESSING  (mirrors training script exactly)
# ============================================================================

def load_and_preprocess(dataset_dir: str):
    """Returns X/Y arrays (scaled) + metadata needed for analysis."""
    file_map = {
        'train': os.path.join(dataset_dir, 'UKDALE_HF_train.csv'),
        'val':   os.path.join(dataset_dir, 'UKDALE_HF_validation.csv'),
        'test':  os.path.join(dataset_dir, 'UKDALE_HF_test.csv'),
    }
    splits = {k: pd.read_csv(v, index_col='timestamp', parse_dates=True)
              for k, v in file_map.items()}

    tr_thr = compute_adaptive_thresholds(splits['train'])
    va_thr = compute_adaptive_thresholds(splits['val'])
    te_thr = compute_adaptive_thresholds(splits['test'])

    X_tr, Y_tr = create_sequences(splits['train'],  tr_thr, STRIDE)
    X_va, Y_va = create_sequences(splits['val'],    va_thr, STRIDE)
    X_te, Y_te = create_sequences(splits['test'],   te_thr, WIN)

    n_feat = X_tr.shape[2]
    feat_scalers = []
    for ch in range(n_feat):
        sc = StandardScaler()
        X_tr[:, :, ch] = sc.fit_transform(X_tr[:, :, ch].reshape(-1, 1)).reshape(-1, WIN)
        X_va[:, :, ch] = sc.transform(    X_va[:, :, ch].reshape(-1, 1)).reshape(-1, WIN)
        X_te[:, :, ch] = sc.transform(    X_te[:, :, ch].reshape(-1, 1)).reshape(-1, WIN)
        feat_scalers.append(sc)

    y_scalers = []
    for i in range(len(APPLIANCES)):
        ys = MinMaxScaler()
        Y_tr[:, :, i] = ys.fit_transform(Y_tr[:, :, i].reshape(-1, 1)).reshape(-1, WIN)
        Y_va[:, :, i] = ys.transform(    Y_va[:, :, i].reshape(-1, 1)).reshape(-1, WIN)
        Y_te[:, :, i] = ys.transform(    Y_te[:, :, i].reshape(-1, 1)).reshape(-1, WIN)
        y_scalers.append(ys)

    return (X_tr, Y_tr, X_va, Y_va, X_te, Y_te,
            feat_scalers, y_scalers,
            tr_thr, va_thr, te_thr, splits)


# ============================================================================
# 2.  ACTIVATION HOOKS
# ============================================================================

class ActivationCollector:
    """Wraps FixedPINNLNN and collects CNN output + final hidden states."""

    def __init__(self, model: FixedPINNLNN, device):
        self.model  = model
        self.device = device
        self._cnn_out  = []   # (batch, WIN, hidden)
        self._h_fwd    = []   # (batch, hidden)  — last fwd state
        self._h_bwd    = []   # (batch, hidden)  — last bwd state (= t=0 bwd pass)
        self._h_all    = []   # (batch, WIN, hidden*2)  — full BiLNN hidden states
        self._power    = []   # (batch, WIN, n_apps)
        self._event    = []   # (batch, WIN, n_apps)

    def _forward_with_hooks(self, x: torch.Tensor):
        model = self.model
        feat  = model.cnn(x)
        batch, T, _ = feat.shape

        h_f = torch.zeros(batch, model.hidden, device=x.device)
        fwd = []
        for t in range(T):
            h_f = model.fwd_cell(feat[:, t, :], h_f)
            fwd.append(h_f)

        h_b  = torch.zeros(batch, model.hidden, device=x.device)
        bwd  = [None] * T
        for t in reversed(range(T)):
            h_b    = model.bwd_cell(feat[:, t, :], h_b)
            bwd[t] = h_b

        power_list, event_list, h_all = [], [], []
        for t in range(T):
            h_t = model.norm(torch.cat([fwd[t], bwd[t]], dim=1))
            h_all.append(h_t)
            power_list.append(
                torch.cat([torch.sigmoid(head(h_t)) for head in model.power_heads], dim=1))
            event_list.append(
                torch.cat([head(h_t) for head in model.event_heads], dim=1))

        self._cnn_out.append(feat.cpu())
        self._h_fwd.append(fwd[-1].cpu())
        self._h_bwd.append(bwd[0].cpu())
        self._h_all.append(torch.stack(h_all, dim=1).cpu())
        self._power.append(torch.stack(power_list, dim=1).cpu())
        self._event.append(torch.stack(event_list, dim=1).cpu())

        return torch.stack(power_list, dim=1), torch.stack(event_list, dim=1)

    def collect(self, X: np.ndarray, batch_size: int = 32):
        self._cnn_out.clear(); self._h_fwd.clear(); self._h_bwd.clear()
        self._h_all.clear();  self._power.clear(); self._event.clear()
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = torch.FloatTensor(X[i:i+batch_size]).to(self.device)
                self._forward_with_hooks(xb)
        return {
            'cnn_out': torch.cat(self._cnn_out, 0).numpy(),    # (N,WIN,H)
            'h_fwd':   torch.cat(self._h_fwd,   0).numpy(),    # (N,H)
            'h_bwd':   torch.cat(self._h_bwd,   0).numpy(),    # (N,H)
            'h_all':   torch.cat(self._h_all,   0).numpy(),    # (N,WIN,2H)
            'power':   torch.cat(self._power,   0).numpy(),    # (N,WIN,A)
            'event':   torch.cat(self._event,   0).numpy(),    # (N,WIN,A)
        }


# ============================================================================
# 3.  WEIGHT ANALYSIS
# ============================================================================

def analyse_weights(model: FixedPINNLNN, out_dir: str):
    print("\n[1] Weight Analysis")
    results = {}

    # ── 3a: CNN kernel singular values ──────────────────────────────────────
    cnn_ranks = {}
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("CNN Layer Singular Value Spectra")
    layer_idx = 0
    for module in model.cnn.net:
        if isinstance(module, nn.Conv1d):
            # Weight shape: (out, in, k) → flatten to (out, in*k)
            W = module.weight.detach().cpu().numpy()
            W2d = W.reshape(W.shape[0], -1)
            sv  = np.linalg.svd(W2d, compute_uv=False)
            # Effective rank = exp(entropy of normalised sv²)
            sv2 = sv ** 2; sv2 /= sv2.sum()
            eff_rank = np.exp(-np.sum(sv2 * np.log(sv2 + 1e-12)))
            cnn_ranks[f'conv{layer_idx}'] = {
                'sv_top5': sv[:5].tolist(),
                'effective_rank': float(eff_rank),
                'shape': list(W.shape),
            }
            ax = axes[layer_idx]
            ax.plot(sv / sv[0], 'o-', ms=3)
            ax.set_title(f"Conv{layer_idx}  eff.rank={eff_rank:.1f}")
            ax.set_xlabel("Singular value index")
            ax.set_ylabel("Normalised σ")
            ax.grid(alpha=0.3)
            layer_idx += 1
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'weight_cnn_sv.png'), dpi=150)
    plt.close()
    results['cnn_singular_values'] = cnn_ranks
    print(f"   CNN layers: {list(cnn_ranks.keys())}")
    for k, v in cnn_ranks.items():
        print(f"     {k}  eff.rank={v['effective_rank']:.2f}  "
              f"top-5σ={[f'{x:.3f}' for x in v['sv_top5']]}")

    # ── 3b: LiquidCell recurrent weight eigenvalues ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (cell_name, cell) in zip(axes,
            [('Forward', model.fwd_cell), ('Backward', model.bwd_cell)]):
        W_rec = cell.rec_weights.detach().cpu().numpy()
        eigs  = np.linalg.eigvals(W_rec)
        spectral_radius = float(np.max(np.abs(eigs)))
        ax.scatter(eigs.real, eigs.imag, s=10, alpha=0.7)
        circle = plt.Circle((0, 0), 1.0, fill=False, color='red',
                             linestyle='--', linewidth=1.5, label='|λ|=1')
        ax.add_patch(circle)
        ax.set_aspect('equal')
        ax.set_title(f"{cell_name} LNN recurrent weights\nSpectral radius={spectral_radius:.4f}")
        ax.set_xlabel("Re(λ)"); ax.set_ylabel("Im(λ)")
        ax.legend(); ax.grid(alpha=0.3)
        results[f'rec_{cell_name.lower()}_spectral_radius'] = spectral_radius
        tau = F.softplus(cell.tau_base).detach().cpu().numpy()
        results[f'tau_{cell_name.lower()}'] = {
            'min': float(tau.min()), 'max': float(tau.max()),
            'mean': float(tau.mean()), 'median': float(np.median(tau))
        }
        print(f"   {cell_name} LNN: spectral_radius={spectral_radius:.4f}  "
              f"τ=[{tau.min():.3f}, {tau.max():.3f}] (mean={tau.mean():.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'weight_rec_eigenvalues.png'), dpi=150)
    plt.close()

    # ── 3c: Power and event head norms ──────────────────────────────────────
    ph_norms = [head.weight.norm().item() for head in model.power_heads]
    eh_norms = [head.weight.norm().item() for head in model.event_heads]
    results['head_norms'] = {
        'power': dict(zip(APPLIANCES, ph_norms)),
        'event': dict(zip(APPLIANCES, eh_norms)),
    }
    print("   Head weight norms:")
    for app, pn, en in zip(APPLIANCES, ph_norms, eh_norms):
        print(f"     {app:<18}  power={pn:.4f}  event={en:.4f}")

    return results


# ============================================================================
# 4.  ACTIVATION / FEATURE-SPACE ANALYSIS
# ============================================================================

def activation_stats(acts: np.ndarray, name: str) -> dict:
    """Compute basic stats on a (N, ...) activation array (flattened)."""
    flat = acts.reshape(-1)
    return {
        'name': name,
        'mean':   float(flat.mean()),
        'std':    float(flat.std()),
        'kurt':   float(kurtosis(flat, fisher=True)),
        'p1':     float(np.percentile(flat, 1)),
        'p99':    float(np.percentile(flat, 99)),
        'frac_saturated': float(np.mean(np.abs(flat) > 9.0)),  # for tanh/clamp
    }


def cka(X: np.ndarray, Y: np.ndarray, seed: int = 0) -> float:
    """
    Linear Centered Kernel Alignment (kernel formulation).

    Both matrices must have the same n rows.  If they differ, the larger set
    is subsampled down to match the smaller one so the n×n kernels align.
    """
    n = min(len(X), len(Y))
    rng = np.random.default_rng(seed)
    if len(X) > n:
        X = X[rng.choice(len(X), n, replace=False)]
    if len(Y) > n:
        Y = Y[rng.choice(len(Y), n, replace=False)]
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    H   = np.eye(n) - np.ones((n, n)) / n
    Kxc = H @ (X @ X.T) @ H
    Kyc = H @ (Y @ Y.T) @ H
    return float(np.sum(Kxc * Kyc) /
                 (np.sqrt(np.sum(Kxc ** 2)) * np.sqrt(np.sum(Kyc ** 2)) + 1e-12))


def analyse_activations(acts_tr, acts_va, acts_te, out_dir: str):
    print("\n[2] Activation / Feature-Space Analysis")
    results = {}

    # ── 4a: Statistics per split per layer ──────────────────────────────────
    for layer in ('cnn_out', 'h_all'):
        print(f"\n   Layer: {layer}")
        for split_name, acts in [('train', acts_tr), ('val', acts_va), ('test', acts_te)]:
            s = activation_stats(acts[layer], f"{split_name}/{layer}")
            results[f'{split_name}_{layer}'] = s
            print(f"     {split_name:5s}  mean={s['mean']:+.4f}  std={s['std']:.4f}  "
                  f"kurt={s['kurt']:+.4f}  sat={s['frac_saturated']:.4f}")

    # ── 4b: Centroid drift (L2) ──────────────────────────────────────────────
    print("\n   Centroid L2 distances:")
    for layer in ('cnn_out', 'h_all'):
        # flatten (N, WIN, D) → (N, WIN*D) then take mean over N
        c_tr = acts_tr[layer].reshape(len(acts_tr[layer]), -1).mean(0)
        c_va = acts_va[layer].reshape(len(acts_va[layer]), -1).mean(0)
        c_te = acts_te[layer].reshape(len(acts_te[layer]), -1).mean(0)
        d_va = float(np.linalg.norm(c_tr - c_va))
        d_te = float(np.linalg.norm(c_tr - c_te))
        results[f'centroid_{layer}_train_val']  = d_va
        results[f'centroid_{layer}_train_test'] = d_te
        print(f"     {layer:10s}  train→val={d_va:.4f}  train→test={d_te:.4f}  "
              f"(ratio={d_te/(d_va+1e-9):.2f}x)")

    # ── 4c: CKA (subsample to speed up O(N²) kernel) ────────────────────────
    print("\n   CKA (linear) between splits:")
    N_sub = 500
    rng = np.random.default_rng(42)
    for layer in ('cnn_out', 'h_all'):
        # Use last timestep (midpoint of BiLNN context)
        if acts_tr[layer].ndim == 3:
            t_mid = acts_tr[layer].shape[1] // 2
            A_tr = acts_tr[layer][:, t_mid, :]
            A_va = acts_va[layer][:, t_mid, :]
            A_te = acts_te[layer][:, t_mid, :]
        else:
            A_tr = acts_tr[layer]
            A_va = acts_va[layer]
            A_te = acts_te[layer]

        idx_tr = rng.choice(len(A_tr), min(N_sub, len(A_tr)), replace=False)
        idx_va = rng.choice(len(A_va), min(N_sub, len(A_va)), replace=False)
        idx_te = rng.choice(len(A_te), min(N_sub, len(A_te)), replace=False)

        c_tv = cka(A_tr[idx_tr], A_va[idx_va])
        c_tt = cka(A_tr[idx_tr], A_te[idx_te])
        results[f'cka_{layer}_train_val']  = c_tv
        results[f'cka_{layer}_train_test'] = c_tt
        print(f"     {layer:10s}  CKA(train,val)={c_tv:.4f}  CKA(train,test)={c_tt:.4f}")

    # ── 4d: PCA projection of hidden states (coloured by appliance ON/OFF) ──
    #    We'll do this in the probing section where we have labels handy.

    return results


# ============================================================================
# 5.  PREDICTION DISTRIBUTION  (threshold-zone occupancy)
# ============================================================================

def analyse_prediction_distributions(
        acts_tr, acts_va, acts_te,
        Y_tr, Y_va, Y_te,
        y_scalers, tr_thr, va_thr, te_thr,
        out_dir: str):
    """
    For each appliance, plot histogram of predictions in W.
    Mark train-threshold and test-threshold vertical lines.
    Report what fraction of predictions in each split land in each zone:
        zone 0: < min(train_thr, test_thr)
        zone 1: between train_thr and test_thr  (the 'gap' causing F1 collapse)
        zone 2: > max(train_thr, test_thr)
    """
    print("\n[3] Prediction Distribution / Threshold-Zone Analysis")
    results = {}

    fig, axes = plt.subplots(len(APPLIANCES), 3, figsize=(14, 4 * len(APPLIANCES)))
    fig.suptitle("Power Prediction Distributions (W)\nDashed=train-thr  Dotted=test-thr",
                 fontsize=12)

    for i, app in enumerate(APPLIANCES):
        ys = y_scalers[i]
        mn, rng_ = float(ys.data_min_[0]), float(ys.data_range_[0])

        # Convert scaled → raw W
        def to_W(pred_scaled):
            return pred_scaled * rng_ + mn

        thr_tr = tr_thr[app]
        thr_te = te_thr[app]

        app_results = {}
        for col_idx, (split_name, acts, Y, thr) in enumerate([
                ('train', acts_tr, Y_tr, thr_tr),
                ('val',   acts_va, Y_va, tr_thr[app]),   # val uses train thr
                ('test',  acts_te, Y_te, thr_te)]):

            pred_scaled = acts['power'][:, :, i].flatten()
            true_scaled = Y[:, :, i].flatten()
            pred_W = to_W(pred_scaled)
            true_W = to_W(true_scaled)

            ax = axes[i][col_idx]
            ax.hist(pred_W, bins=80, density=True, alpha=0.6,
                    label='pred', color='steelblue')
            ax.hist(true_W, bins=80, density=True, alpha=0.5,
                    label='true', color='tomato')
            ax.axvline(thr_tr, color='black',  linestyle='--', lw=1.5,
                       label=f'thr_train={thr_tr:.0f}W')
            if abs(thr_te - thr_tr) > 1:
                ax.axvline(thr_te, color='purple', linestyle=':', lw=1.5,
                           label=f'thr_test={thr_te:.0f}W')
            ax.set_title(f"{app} / {split_name}")
            ax.set_xlabel("W"); ax.set_ylabel("density")
            ax.legend(fontsize=6); ax.grid(alpha=0.3)

            # Zone fractions
            lo = min(thr_tr, thr_te)
            hi = max(thr_tr, thr_te)
            z0 = float(np.mean(pred_W < lo))
            z1 = float(np.mean((pred_W >= lo) & (pred_W < hi)))
            z2 = float(np.mean(pred_W >= hi))
            app_results[split_name] = {
                'thr_used': thr,
                'pred_mean_W': float(pred_W.mean()),
                'pred_std_W':  float(pred_W.std()),
                'zone_below_lo': z0,
                'zone_gap':      z1,
                'zone_above_hi': z2,
            }

        results[app] = app_results
        print(f"\n   {app}  (train_thr={thr_tr:.0f}W  test_thr={thr_te:.0f}W  "
              f"gap={abs(thr_te-thr_tr):.0f}W)")
        for split_name, r in app_results.items():
            print(f"     {split_name:5s}  pred_mean={r['pred_mean_W']:.1f}W  "
                  f"zone_below={r['zone_below_lo']:.3f}  "
                  f"zone_gap={r['zone_gap']:.3f}  "
                  f"zone_above={r['zone_above_hi']:.3f}")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pred_distributions.png'), dpi=150)
    plt.close()
    return results


# ============================================================================
# 6.  LINEAR PROBING
# ============================================================================

def linear_probe(acts_tr, acts_va, acts_te,
                 Y_tr, Y_va, Y_te,
                 y_scalers, tr_thr, va_thr, te_thr,
                 n_probe: int, out_dir: str):
    """
    Train a logistic regression on hidden states from the train split,
    test on train / val / test.  Reports AUC-ROC per appliance per layer.
    """
    print("\n[4] Linear Probing (logistic regression on hidden states)")
    results = {}
    rng = np.random.default_rng(0)

    for layer in ('cnn_out', 'h_all'):
        print(f"\n   Layer: {layer}")
        results[layer] = {}

        for i, app in enumerate(APPLIANCES):
            ys  = y_scalers[i]
            mn  = float(ys.data_min_[0])
            rng_ = float(ys.data_range_[0])

            def label(Y, thr_W):
                thr_s = (thr_W - mn) / (rng_ + 1e-12)
                return (Y[:, :, i].flatten() > thr_s).astype(int)

            # Extract features: use midpoint of time axis for 3-D arrays
            def feat(acts):
                a = acts[layer]
                if a.ndim == 3:
                    a = a[:, a.shape[1] // 2, :]  # (N, D)
                return a

            F_tr = feat(acts_tr)
            F_va = feat(acts_va)
            F_te = feat(acts_te)

            # Labels: repeat per timestep then flatten (we used midpoint feature)
            # Actually for per-window label we use midpoint label too
            t_mid = Y_tr.shape[1] // 2
            y_tr = (Y_tr[:, t_mid, i] > (tr_thr[app] - mn) / (rng_ + 1e-12)).astype(int)
            y_va = (Y_va[:, t_mid, i] > (tr_thr[app] - mn) / (rng_ + 1e-12)).astype(int)
            y_te = (Y_te[:, t_mid, i] > (te_thr[app] - mn) / (rng_ + 1e-12)).astype(int)

            # Subsample for speed
            idx_tr = rng.choice(len(F_tr), min(n_probe, len(F_tr)), replace=False)

            scaler_p = SKStd()
            F_tr_s = scaler_p.fit_transform(F_tr[idx_tr])
            F_va_s = scaler_p.transform(F_va)
            F_te_s = scaler_p.transform(F_te)

            clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
            clf.fit(F_tr_s, y_tr[idx_tr])

            def safe_auc(y_true, y_score):
                if len(np.unique(y_true)) < 2:
                    return float('nan')
                return float(roc_auc_score(y_true, y_score))

            auc_tr = safe_auc(y_tr[idx_tr], clf.predict_proba(F_tr_s)[:, 1])
            auc_va = safe_auc(y_va, clf.predict_proba(F_va_s)[:, 1])
            auc_te = safe_auc(y_te, clf.predict_proba(F_te_s)[:, 1])
            results[layer][app] = {
                'auc_train': auc_tr, 'auc_val': auc_va, 'auc_test': auc_te,
            }
            print(f"     {app:<18}  AUC train={auc_tr:.4f}  val={auc_va:.4f}  test={auc_te:.4f}")

    return results


# ============================================================================
# 7.  GRADIENT × INPUT ATTRIBUTION  (on FP test windows)
# ============================================================================

def gradient_attribution(model: FixedPINNLNN,
                          X_te: np.ndarray, Y_te: np.ndarray,
                          y_scalers, te_thr, device,
                          out_dir: str):
    """
    For each appliance, identify test windows that are False Positives
    (model predicts ON at midpoint, ground truth is OFF).
    Compute gradient of power_pred w.r.t. input, aggregate channel-wise.
    """
    print("\n[5] Gradient × Input Attribution (test False Positives)")

    t_mid = WIN // 2
    CHANNEL_NAMES = ['raw', 'median', 'EMA', 'residual',
                     'Δraw', 'Δsmooth', 'roll_mean', 'roll_std']
    results = {}

    fig, axes = plt.subplots(len(APPLIANCES), 1,
                             figsize=(10, 4 * len(APPLIANCES)))
    fig.suptitle("FP Attribution: |grad × input| per channel (test split)")

    model.eval()
    for i, app in enumerate(APPLIANCES):
        ys  = y_scalers[i]
        mn  = float(ys.data_min_[0])
        rng_ = float(ys.data_range_[0])
        thr_s = (te_thr[app] - mn) / (rng_ + 1e-12)

        # Find FP windows: pred > thr_s, true ≤ thr_s at midpoint
        X_t = torch.FloatTensor(X_te).to(device)
        with torch.no_grad():
            pw, _ = model(X_t)
        pred_mid = pw[:, t_mid, i].cpu().numpy()
        true_mid = Y_te[:, t_mid, i]
        is_fp = (pred_mid > thr_s) & (true_mid <= thr_s)
        fp_idx = np.where(is_fp)[0]
        print(f"\n   {app}: {len(fp_idx)} FP windows "
              f"out of {len(X_te)} test windows")

        if len(fp_idx) == 0:
            results[app] = None
            continue

        # Gradient × input
        X_fp = torch.FloatTensor(X_te[fp_idx]).to(device).requires_grad_(True)
        pw_fp, _ = model(X_fp)
        # Scalar: sum of power predictions at midpoint for this appliance
        score = pw_fp[:, t_mid, i].sum()
        score.backward()
        grad_inp = (X_fp.grad * X_fp.detach()).abs().cpu().numpy()  # (N_fp, WIN, 8)
        # Mean over FP windows and WIN positions → (8,)
        channel_attr = grad_inp.mean(axis=(0, 1))

        results[app] = {
            'n_fp': int(len(fp_idx)),
            'channel_attribution': dict(zip(CHANNEL_NAMES, channel_attr.tolist()))
        }

        ax = axes[i]
        bars = ax.bar(range(8), channel_attr, color='steelblue')
        ax.set_xticks(range(8))
        ax.set_xticklabels(CHANNEL_NAMES, rotation=30, ha='right', fontsize=9)
        ax.set_title(f"{app}  (n_FP={len(fp_idx)})")
        ax.set_ylabel("|grad × input|")
        ax.grid(alpha=0.3, axis='y')
        print(f"     Top channels: " +
              "  ".join(f"{CHANNEL_NAMES[j]}={channel_attr[j]:.4f}"
                        for j in np.argsort(channel_attr)[::-1][:3]))

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fp_attribution.png'), dpi=150)
    plt.close()
    return results


# ============================================================================
# 8.  PCA VISUALISATION
# ============================================================================

def pca_hidden_states(acts_tr, acts_va, acts_te,
                      Y_tr, Y_va, Y_te,
                      y_scalers, tr_thr, te_thr,
                      out_dir: str, n_vis: int = 1000):
    print("\n[6] PCA of hidden states")
    rng = np.random.default_rng(7)

    t_mid = acts_tr['h_all'].shape[1] // 2
    H_tr  = acts_tr['h_all'][:, t_mid, :]
    H_te  = acts_te['h_all'][:, t_mid, :]

    # Fit PCA on train
    pca = PCA(n_components=2)
    pca.fit(H_tr)

    idx_tr = rng.choice(len(H_tr), min(n_vis, len(H_tr)), replace=False)
    idx_te = rng.choice(len(H_te), min(n_vis, len(H_te)), replace=False)
    Htr2 = pca.transform(H_tr[idx_tr])
    Hte2 = pca.transform(H_te[idx_te])

    fig, axes = plt.subplots(2, len(APPLIANCES),
                             figsize=(5 * len(APPLIANCES), 9))
    fig.suptitle("PCA of BiLNN hidden states (t=mid)\n"
                 "Colour = appliance ON/OFF label")

    for i, app in enumerate(APPLIANCES):
        ys  = y_scalers[i]
        mn  = float(ys.data_min_[0])
        rng_ = float(ys.data_range_[0])
        thr_tr_s = (tr_thr[app] - mn) / (rng_ + 1e-12)
        thr_te_s = (te_thr[app] - mn) / (rng_ + 1e-12)

        lbl_tr = (Y_tr[idx_tr, Y_tr.shape[1] // 2, i] > thr_tr_s).astype(int)
        lbl_te = (Y_te[idx_te, Y_te.shape[1] // 2, i] > thr_te_s).astype(int)

        for row, (label, H2, split_name) in enumerate([
                (lbl_tr, Htr2, 'train'), (lbl_te, Hte2, 'test')]):
            ax = axes[row][i]
            colors = np.where(label, 'tomato', 'steelblue')
            ax.scatter(H2[:, 0], H2[:, 1], c=colors, s=8, alpha=0.5)
            ax.set_title(f"{app} / {split_name}")
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            from matplotlib.lines import Line2D
            legend_els = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='tomato', ms=8, label='ON'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='steelblue', ms=8, label='OFF'),
            ]
            ax.legend(handles=legend_els, fontsize=8)
            ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pca_hidden.png'), dpi=150)
    plt.close()

    var_exp = pca.explained_variance_ratio_
    print(f"   PCA variance explained: PC1={var_exp[0]:.3f}  PC2={var_exp[1]:.3f}")
    return {'pca_var_pc1': float(var_exp[0]), 'pca_var_pc2': float(var_exp[1])}


# ============================================================================
# 9.  MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir',   required=True,
                        help='Directory containing best_model.pt and fixed_pinn_lnn_results.json')
    parser.add_argument('--dataset-dir', default=None)
    parser.add_argument('--hidden',      type=int,   default=64)
    parser.add_argument('--n-probe',     type=int,   default=2000,
                        help='Windows to use for linear probing')
    parser.add_argument('--device',      default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    out_dir = os.path.join(args.model_dir, 'repr_analysis')
    os.makedirs(out_dir, exist_ok=True)

    dataset_dir = args.dataset_dir or DATA_DIR
    device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Load config (thresholds etc.) from JSON ──────────────────────────────
    cfg_path = os.path.join(args.model_dir, 'fixed_pinn_lnn_results.json')
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        print(f"Loaded config from {cfg_path}")
    else:
        cfg = {}

    # ── Load data ────────────────────────────────────────────────────────────
    print("\nLoading and preprocessing data ...")
    (X_tr, Y_tr, X_va, Y_va, X_te, Y_te,
     feat_scalers, y_scalers,
     tr_thr, va_thr, te_thr, splits) = load_and_preprocess(dataset_dir)
    print(f"  X_tr={X_tr.shape}  X_va={X_va.shape}  X_te={X_te.shape}")

    # ── Load model ───────────────────────────────────────────────────────────
    ckpt_path = os.path.join(args.model_dir, 'best_model.pt')
    if not os.path.exists(ckpt_path):
        print(f"ERROR: {ckpt_path} not found.  "
              f"Re-run training (the script now saves best_model.pt automatically).")
        sys.exit(1)

    model = FixedPINNLNN(in_ch=X_tr.shape[2], hidden=args.hidden,
                         n_apps=len(APPLIANCES), dt=0.1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Loaded model from {ckpt_path}")

    # ── Collect activations ──────────────────────────────────────────────────
    print("\nCollecting activations (forward passes) ...")
    collector = ActivationCollector(model, device)
    acts_tr = collector.collect(X_tr)
    print("  train done")
    acts_va = collector.collect(X_va)
    print("  val done")
    acts_te = collector.collect(X_te)
    print("  test done")

    # ── Run analyses ─────────────────────────────────────────────────────────
    all_results = {}
    all_results['weights']      = analyse_weights(model, out_dir)
    all_results['activations']  = analyse_activations(acts_tr, acts_va, acts_te, out_dir)
    all_results['pred_dist']    = analyse_prediction_distributions(
                                      acts_tr, acts_va, acts_te,
                                      Y_tr, Y_va, Y_te,
                                      y_scalers, tr_thr, va_thr, te_thr, out_dir)
    all_results['probing']      = linear_probe(
                                      acts_tr, acts_va, acts_te,
                                      Y_tr, Y_va, Y_te,
                                      y_scalers, tr_thr, va_thr, te_thr,
                                      args.n_probe, out_dir)
    all_results['attribution']  = gradient_attribution(
                                      model, X_te, Y_te,
                                      y_scalers, te_thr, device, out_dir)
    all_results['pca']          = pca_hidden_states(
                                      acts_tr, acts_va, acts_te,
                                      Y_tr, Y_va, Y_te,
                                      y_scalers, tr_thr, te_thr, out_dir)

    # ── Save summary JSON ────────────────────────────────────────────────────
    def _json_safe(obj):
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    out_path = os.path.join(out_dir, 'repr_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(_json_safe(all_results), f, indent=2)
    print(f"\nAnalysis complete.  Results + plots saved to: {out_dir}")
    print(f"  JSON: {out_path}")

    # ── Print diagnostic summary ─────────────────────────────────────────────
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)

    print("\n>> Representational drift (CKA, hidden states):")
    h = all_results['activations']
    for k in ['cka_h_all_train_val', 'cka_h_all_train_test']:
        if k in h:
            print(f"   {k} = {h[k]:.4f}  (1.0=identical, 0.0=unrelated)")

    print("\n>> Threshold-gap analysis (explains F1 collapse on test):")
    for app, r in all_results['pred_dist'].items():
        tr_thr_v = tr_thr[app]; te_thr_v = te_thr[app]
        gap = abs(te_thr_v - tr_thr_v)
        zone_gap_te = r.get('test', {}).get('zone_gap', 0.0)
        print(f"   {app:<18}  gap={gap:.0f}W  "
              f"frac_pred_in_gap(test)={zone_gap_te:.3f}")

    print("\n>> Linear probe AUC-ROC (h_all layer):")
    if 'h_all' in all_results['probing']:
        for app, r in all_results['probing']['h_all'].items():
            print(f"   {app:<18}  train={r['auc_train']:.4f}  "
                  f"val={r['auc_val']:.4f}  test={r['auc_test']:.4f}")

    print("\n>> FP attribution (top-2 channels per appliance):")
    for app, r in all_results['attribution'].items():
        if r is None:
            print(f"   {app}: no FP windows found")
            continue
        attrs = r['channel_attribution']
        top2 = sorted(attrs, key=attrs.get, reverse=True)[:2]
        print(f"   {app:<18}  n_FP={r['n_fp']}  "
              f"top2: {top2[0]}={attrs[top2[0]]:.4f}  {top2[1]}={attrs[top2[1]]:.4f}")


if __name__ == '__main__':
    main()
