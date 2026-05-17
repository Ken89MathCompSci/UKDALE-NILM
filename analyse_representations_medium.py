"""
analyse_representations_medium.py
===================================
Post-hoc internal-representation analysis for PhysicsInformedLiquidNetworkModel
(test_pinn_lnn_ukdale_medium_dataset.py).

Analyses performed
------------------
1. Weight Analysis
   - LiquidCell recurrent-weight eigenvalue spectrum  (stability / memory)
   - tau_base distribution  (time-constant per hidden unit)
   - Per-appliance head weight norms

2. Hidden-State Analysis
   - Hidden-state statistics per split  (mean, std, kurtosis, saturation)
   - Centroid L2 drift: train→val and train→test
   - Centered Kernel Alignment (CKA) between splits

3. Prediction Distribution  (threshold-zone occupancy)
   - Per-appliance prediction histograms across splits
   - Threshold is fixed at 10 W for all appliances, so the "gap" here
     reflects drift in prediction magnitude rather than threshold shift

4. Linear Probing
   - Logistic regression on hidden state (after LayerNorm), trained on train,
     evaluated on train / val / test
   - AUC-ROC per appliance per split reveals where encoding breaks down

5. Temporal Attribution  (gradient × input on test False Positives)
   - Input is (batch, WIN=100, 1), so attribution is over time, not channels
   - Reveals which part of the 100-step window drives false alarms per appliance
   - Plots mean |∇x · x| profile over the window

6. PCA of hidden states
   - 2-D projection coloured by appliance ON/OFF label, train vs test

Usage
-----
  python analyse_representations_medium.py --model-dir models/pinn_lnn_medium_TIMESTAMP

  Options:
    --dataset-dir  PATH    (default: medium_dataset)
    --hidden-size  INT     (default: 64, must match saved model)
    --n-probe      INT     windows to use for linear probing  (default: 2000)
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
from sklearn.preprocessing import StandardScaler as SKStd, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from test_pinn_lnn_ukdale_medium_dataset import (
    APPLIANCES, WIN, STRIDE, THRESHOLDS,
    DEFAULT_DATASET_DIR,
    PhysicsInformedLiquidNetworkModel,
    load_data, create_sequences,
    apply_min_on_off, MIN_ON, MIN_OFF,
)


# ============================================================================
# 1.  DATA LOADING & PREPROCESSING  (mirrors training script exactly)
# ============================================================================

def load_and_preprocess(dataset_dir: str):
    data_dict = load_data(dataset_dir)

    X_tr, Y_tr = create_sequences(data_dict['train'],      WIN)
    X_va, Y_va = create_sequences(data_dict['validation'], WIN)
    X_te, Y_te = create_sequences(data_dict['test'],       WIN)

    x_scaler = MinMaxScaler()
    X_tr = x_scaler.fit_transform(X_tr.reshape(-1, 1)).reshape(X_tr.shape)
    X_va = x_scaler.transform(    X_va.reshape(-1, 1)).reshape(X_va.shape)
    X_te = x_scaler.transform(    X_te.reshape(-1, 1)).reshape(X_te.shape)

    y_scalers = []
    for i in range(len(APPLIANCES)):
        ys = MinMaxScaler()
        Y_tr[:, i:i+1] = ys.fit_transform(Y_tr[:, i:i+1])
        Y_va[:, i:i+1] = ys.transform(    Y_va[:, i:i+1])
        Y_te[:, i:i+1] = ys.transform(    Y_te[:, i:i+1])
        y_scalers.append(ys)

    return (X_tr, Y_tr, X_va, Y_va, X_te, Y_te,
            x_scaler, y_scalers, data_dict)


# ============================================================================
# 2.  ACTIVATION COLLECTION
# ============================================================================

class ActivationCollector:
    """
    Runs the LNN forward pass and captures the hidden state trajectory
    plus the final (post-norm) hidden state and output predictions.
    """

    def __init__(self, model: PhysicsInformedLiquidNetworkModel, device):
        self.model  = model
        self.device = device

    def _forward_with_hooks(self, x: torch.Tensor):
        """
        Returns dict of activations for a single batch.

        h_traj : (batch, WIN, hidden)  — hidden state at every timestep
        h_final: (batch, hidden)       — hidden state after LayerNorm
        pred   : (batch, n_apps)       — output predictions
        """
        m = self.model
        batch, seq_len, _ = x.size()
        h = torch.zeros(batch, m.hidden_size, device=x.device)

        h_traj = []
        for t in range(seq_len):
            x_t        = x[:, t, :]
            input_proj = m.input_proj(x_t)
            rec_proj   = torch.matmul(h, m.rec_weights)
            tau_base   = F.softplus(m.tau_base).unsqueeze(0)
            tau_mod    = torch.sigmoid(m.tau_mod(x_t))
            tau        = (tau_base * tau_mod).clamp(min=m.dt)
            gate       = torch.sigmoid(m.gate(torch.cat([x_t, h], dim=1)))
            f_t        = torch.tanh(input_proj + rec_proj)
            dh         = ((-h / tau) + gate * f_t) * m.dt
            h          = (h + dh).clamp(-10.0, 10.0)
            h_traj.append(h)

        h_normed = m.norm(h)
        pred     = torch.cat([head(h_normed) for head in m.heads], dim=1)

        return {
            'h_traj':  torch.stack(h_traj, dim=1).cpu(),  # (batch, WIN, hidden)
            'h_final': h_normed.cpu(),                     # (batch, hidden)
            'pred':    pred.cpu(),                          # (batch, n_apps)
        }

    def collect(self, X: np.ndarray, batch_size: int = 128):
        self.model.eval()
        all_h_traj, all_h_final, all_pred = [], [], []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb  = torch.FloatTensor(X[i:i+batch_size]).to(self.device)
                out = self._forward_with_hooks(xb)
                all_h_traj.append(out['h_traj'])
                all_h_final.append(out['h_final'])
                all_pred.append(out['pred'])
        return {
            'h_traj':  torch.cat(all_h_traj,  0).numpy(),  # (N, WIN, hidden)
            'h_final': torch.cat(all_h_final, 0).numpy(),  # (N, hidden)
            'pred':    torch.cat(all_pred,    0).numpy(),   # (N, n_apps)
        }


# ============================================================================
# 3.  WEIGHT ANALYSIS
# ============================================================================

def analyse_weights(model: PhysicsInformedLiquidNetworkModel, out_dir: str):
    print("\n[1] Weight Analysis")
    results = {}

    # ── 3a: Recurrent weight eigenvalue spectrum ─────────────────────────────
    W_rec = model.rec_weights.detach().cpu().numpy()
    eigs  = np.linalg.eigvals(W_rec)
    spectral_radius = float(np.max(np.abs(eigs)))
    results['rec_spectral_radius'] = spectral_radius

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(eigs.real, eigs.imag, s=12, alpha=0.7, color='steelblue')
    circle = plt.Circle((0, 0), 1.0, fill=False, color='red',
                         linestyle='--', linewidth=1.5, label='|λ|=1')
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_title(f"LNN recurrent weight eigenvalues\nSpectral radius = {spectral_radius:.4f}")
    ax.set_xlabel("Re(λ)"); ax.set_ylabel("Im(λ)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'weight_rec_eigenvalues.png'), dpi=150)
    plt.close()
    print(f"   Recurrent spectral radius: {spectral_radius:.4f}")

    # ── 3b: tau_base distribution ────────────────────────────────────────────
    tau = F.softplus(model.tau_base).detach().cpu().numpy()
    results['tau_base'] = {
        'min':    float(tau.min()),
        'max':    float(tau.max()),
        'mean':   float(tau.mean()),
        'median': float(np.median(tau)),
    }
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(tau, bins=30, color='steelblue', edgecolor='white')
    ax.set_title(f"τ_base distribution  (mean={tau.mean():.3f}  max={tau.max():.3f})")
    ax.set_xlabel("τ (time constant)"); ax.set_ylabel("count")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'weight_tau_distribution.png'), dpi=150)
    plt.close()
    print(f"   tau_base: min={tau.min():.3f}  mean={tau.mean():.3f}  max={tau.max():.3f}")

    # ── 3c: Head weight norms ────────────────────────────────────────────────
    head_norms = {app: float(model.heads[i].weight.norm())
                  for i, app in enumerate(APPLIANCES)}
    results['head_norms'] = head_norms
    print("   Head weight norms:")
    for app, n in head_norms.items():
        print(f"     {app:<18}  {n:.4f}")

    # ── 3d: Input proj + gate weight norms ───────────────────────────────────
    results['input_proj_norm'] = float(model.input_proj.weight.norm())
    results['gate_norm']       = float(model.gate.weight.norm())
    results['rec_norm']        = float(model.rec_weights.norm())
    print(f"   input_proj norm: {results['input_proj_norm']:.4f}  "
          f"gate norm: {results['gate_norm']:.4f}  "
          f"rec norm:  {results['rec_norm']:.4f}")

    return results


# ============================================================================
# 4.  HIDDEN-STATE ANALYSIS
# ============================================================================

def activation_stats(arr: np.ndarray, name: str) -> dict:
    flat = arr.reshape(-1)
    return {
        'name':           name,
        'mean':           float(flat.mean()),
        'std':            float(flat.std()),
        'kurt':           float(kurtosis(flat, fisher=True)),
        'p1':             float(np.percentile(flat, 1)),
        'p99':            float(np.percentile(flat, 99)),
        'frac_saturated': float(np.mean(np.abs(flat) > 9.0)),
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
    X = X - X.mean(0); Y = Y - Y.mean(0)
    H   = np.eye(n) - np.ones((n, n)) / n
    Kxc = H @ (X @ X.T) @ H
    Kyc = H @ (Y @ Y.T) @ H
    return float(np.sum(Kxc * Kyc) /
                 (np.sqrt(np.sum(Kxc ** 2)) * np.sqrt(np.sum(Kyc ** 2)) + 1e-12))


def analyse_hidden_states(acts_tr, acts_va, acts_te, out_dir: str):
    print("\n[2] Hidden-State Analysis")
    results = {}

    # ── 4a: Statistics ────────────────────────────────────────────────────────
    print("\n   Statistics (h_final):")
    for split_name, acts in [('train', acts_tr), ('val', acts_va), ('test', acts_te)]:
        s = activation_stats(acts['h_final'], split_name)
        results[f'{split_name}_h_final'] = s
        print(f"     {split_name:5s}  mean={s['mean']:+.4f}  std={s['std']:.4f}  "
              f"kurt={s['kurt']:+.4f}  sat={s['frac_saturated']:.4f}")

    # ── 4b: Centroid drift ────────────────────────────────────────────────────
    print("\n   Centroid L2 distances (h_final):")
    c_tr = acts_tr['h_final'].mean(0)
    c_va = acts_va['h_final'].mean(0)
    c_te = acts_te['h_final'].mean(0)
    d_va = float(np.linalg.norm(c_tr - c_va))
    d_te = float(np.linalg.norm(c_tr - c_te))
    results['centroid_train_val']  = d_va
    results['centroid_train_test'] = d_te
    print(f"     train→val={d_va:.4f}  train→test={d_te:.4f}  "
          f"(ratio={d_te/(d_va+1e-9):.2f}x)")

    # ── 4c: CKA ───────────────────────────────────────────────────────────────
    print("\n   CKA (linear) between splits:")
    rng   = np.random.default_rng(42)
    N_sub = 500
    idx_tr = rng.choice(len(acts_tr['h_final']), min(N_sub, len(acts_tr['h_final'])), replace=False)
    idx_va = rng.choice(len(acts_va['h_final']), min(N_sub, len(acts_va['h_final'])), replace=False)
    idx_te = rng.choice(len(acts_te['h_final']), min(N_sub, len(acts_te['h_final'])), replace=False)
    c_tv = cka(acts_tr['h_final'][idx_tr], acts_va['h_final'][idx_va])
    c_tt = cka(acts_tr['h_final'][idx_tr], acts_te['h_final'][idx_te])
    results['cka_train_val']  = c_tv
    results['cka_train_test'] = c_tt
    print(f"     CKA(train,val)={c_tv:.4f}  CKA(train,test)={c_tt:.4f}  "
          f"(1.0=identical)")

    # ── 4d: h_traj variance over time (how much hidden state moves) ───────────
    # Variance across the WIN dimension → (hidden,)
    print("\n   Hidden-state temporal variance (mean over hidden units):")
    for split_name, acts in [('train', acts_tr), ('val', acts_va), ('test', acts_te)]:
        tvar = acts['h_traj'].var(axis=1).mean()   # mean over N and hidden
        results[f'{split_name}_h_traj_temporal_var'] = float(tvar)
        print(f"     {split_name:5s}  mean temporal var={tvar:.5f}")

    return results


# ============================================================================
# 5.  PREDICTION DISTRIBUTION
# ============================================================================

def analyse_prediction_distributions(
        acts_tr, acts_va, acts_te,
        Y_tr, Y_va, Y_te,
        y_scalers, out_dir: str):
    print("\n[3] Prediction Distribution Analysis")
    results = {}
    THR_W = THRESHOLDS  # fixed 10 W for all appliances

    fig, axes = plt.subplots(len(APPLIANCES), 3, figsize=(14, 4 * len(APPLIANCES)))
    fig.suptitle("Power Prediction Distributions (W)\nDashed = 10 W threshold", fontsize=12)

    for i, app in enumerate(APPLIANCES):
        ys   = y_scalers[i]
        mn   = float(ys.data_min_[0])
        rng_ = float(ys.data_range_[0])
        thr  = THR_W[app]

        app_results = {}
        for col, (split_name, acts, Y) in enumerate([
                ('train', acts_tr, Y_tr),
                ('val',   acts_va, Y_va),
                ('test',  acts_te, Y_te)]):

            pred_W = acts['pred'][:, i] * rng_ + mn
            true_W = Y[:, i] * rng_ + mn

            ax = axes[i][col]
            ax.hist(pred_W, bins=80, density=True, alpha=0.6,
                    label='pred', color='steelblue')
            ax.hist(true_W, bins=80, density=True, alpha=0.5,
                    label='true', color='tomato')
            ax.axvline(thr, color='black', linestyle='--', lw=1.5,
                       label=f'thr={thr:.0f}W')
            ax.set_title(f"{app} / {split_name}")
            ax.set_xlabel("W"); ax.set_ylabel("density")
            ax.legend(fontsize=6); ax.grid(alpha=0.3)

            app_results[split_name] = {
                'pred_mean_W': float(pred_W.mean()),
                'pred_std_W':  float(pred_W.std()),
                'frac_above_thr': float(np.mean(pred_W > thr)),
                'true_frac_on':   float(np.mean(true_W > thr)),
            }

        results[app] = app_results
        print(f"\n   {app}  (thr={thr:.0f}W)")
        for split_name, r in app_results.items():
            print(f"     {split_name:5s}  pred_mean={r['pred_mean_W']:.1f}W  "
                  f"pred_frac_ON={r['frac_above_thr']:.3f}  "
                  f"true_frac_ON={r['true_frac_on']:.3f}")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pred_distributions.png'), dpi=150)
    plt.close()
    return results


# ============================================================================
# 6.  LINEAR PROBING
# ============================================================================

def linear_probe(acts_tr, acts_va, acts_te,
                 Y_tr, Y_va, Y_te,
                 y_scalers, n_probe: int, out_dir: str):
    print("\n[4] Linear Probing (logistic regression on h_final)")
    results = {}
    rng = np.random.default_rng(0)

    for i, app in enumerate(APPLIANCES):
        ys   = y_scalers[i]
        mn   = float(ys.data_min_[0])
        rng_ = float(ys.data_range_[0])
        thr_s = (THRESHOLDS[app] - mn) / (rng_ + 1e-12)

        y_tr = (Y_tr[:, i] > thr_s).astype(int)
        y_va = (Y_va[:, i] > thr_s).astype(int)
        y_te = (Y_te[:, i] > thr_s).astype(int)

        F_tr = acts_tr['h_final']
        F_va = acts_va['h_final']
        F_te = acts_te['h_final']

        idx_tr = rng.choice(len(F_tr), min(n_probe, len(F_tr)), replace=False)

        scaler = SKStd()
        F_tr_s = scaler.fit_transform(F_tr[idx_tr])
        F_va_s = scaler.transform(F_va)
        F_te_s = scaler.transform(F_te)

        clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
        clf.fit(F_tr_s, y_tr[idx_tr])

        def safe_auc(y_true, scores):
            return float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float('nan')

        auc_tr = safe_auc(y_tr[idx_tr], clf.predict_proba(F_tr_s)[:, 1])
        auc_va = safe_auc(y_va,         clf.predict_proba(F_va_s)[:, 1])
        auc_te = safe_auc(y_te,         clf.predict_proba(F_te_s)[:, 1])

        results[app] = {'auc_train': auc_tr, 'auc_val': auc_va, 'auc_test': auc_te}
        print(f"   {app:<18}  AUC train={auc_tr:.4f}  val={auc_va:.4f}  test={auc_te:.4f}")

    return results


# ============================================================================
# 7.  TEMPORAL ATTRIBUTION  (gradient × input over the WIN timesteps)
# ============================================================================

def temporal_attribution(model: PhysicsInformedLiquidNetworkModel,
                          X_te: np.ndarray, Y_te: np.ndarray,
                          y_scalers, device, out_dir: str):
    """
    For each appliance, find test FP windows (predicted ON, true OFF).
    Compute |∂pred/∂x_t · x_t| averaged over FP windows → (WIN,) profile.
    Reveals which part of the 100-step input window drives false alarms.
    """
    print("\n[5] Temporal Attribution (gradient × input on test FPs)")
    results = {}
    model.eval()

    fig, axes = plt.subplots(len(APPLIANCES), 1,
                             figsize=(10, 4 * len(APPLIANCES)))
    fig.suptitle("FP Temporal Attribution: |grad × input| over WIN\n"
                 "(which timesteps drive false alarms)", fontsize=11)

    # Step labels at 6-second resolution centred so 0 = midpoint
    t_axis = np.arange(WIN) - WIN // 2   # -50 … +49

    for i, app in enumerate(APPLIANCES):
        ys   = y_scalers[i]
        mn   = float(ys.data_min_[0])
        rng_ = float(ys.data_range_[0])
        thr_s = (THRESHOLDS[app] - mn) / (rng_ + 1e-12)

        # Full forward pass (no grad) to find FP windows
        X_t = torch.FloatTensor(X_te).to(device)
        with torch.no_grad():
            pred_all = model(X_t).cpu().numpy()
        pred_mid = pred_all[:, i]
        true_mid = Y_te[:, i]
        fp_idx   = np.where((pred_mid > thr_s) & (true_mid <= thr_s))[0]

        print(f"\n   {app}: {len(fp_idx)} FP windows")
        if len(fp_idx) == 0:
            results[app] = None
            continue

        # Gradient × input for FP windows
        X_fp = torch.FloatTensor(X_te[fp_idx]).to(device).requires_grad_(True)
        pred_fp = model(X_fp)
        score   = pred_fp[:, i].sum()
        score.backward()
        # X_fp: (n_fp, WIN, 1) → squeeze channel
        attr = (X_fp.grad * X_fp.detach()).abs().squeeze(-1).cpu().numpy()  # (n_fp, WIN)
        temporal_profile = attr.mean(axis=0)  # (WIN,)

        results[app] = {
            'n_fp': int(len(fp_idx)),
            'temporal_profile': temporal_profile.tolist(),
            'peak_timestep_relative': int(t_axis[np.argmax(temporal_profile)]),
        }
        print(f"     peak at t={t_axis[np.argmax(temporal_profile)]:+d} "
              f"(relative to midpoint)")

        ax = axes[i]
        ax.plot(t_axis, temporal_profile, color='steelblue', linewidth=1.5)
        ax.axvline(0, color='black', linestyle='--', lw=1, label='midpoint')
        ax.set_title(f"{app}  (n_FP={len(fp_idx)})")
        ax.set_xlabel("Timestep relative to midpoint (6 s each)")
        ax.set_ylabel("|grad × input|")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fp_temporal_attribution.png'), dpi=150)
    plt.close()
    return results


# ============================================================================
# 8.  PCA VISUALISATION
# ============================================================================

def pca_hidden_states(acts_tr, acts_va, acts_te,
                      Y_tr, Y_va, Y_te,
                      y_scalers, out_dir: str, n_vis: int = 1000):
    print("\n[6] PCA of hidden states (h_final)")
    rng = np.random.default_rng(7)

    pca = PCA(n_components=2)
    pca.fit(acts_tr['h_final'])

    idx_tr = rng.choice(len(acts_tr['h_final']), min(n_vis, len(acts_tr['h_final'])), replace=False)
    idx_te = rng.choice(len(acts_te['h_final']), min(n_vis, len(acts_te['h_final'])), replace=False)
    Htr2 = pca.transform(acts_tr['h_final'][idx_tr])
    Hte2 = pca.transform(acts_te['h_final'][idx_te])

    fig, axes = plt.subplots(2, len(APPLIANCES),
                             figsize=(5 * len(APPLIANCES), 9))
    fig.suptitle("PCA of LNN hidden states (h_final)\nColour = appliance ON/OFF",
                 fontsize=11)

    for i, app in enumerate(APPLIANCES):
        ys   = y_scalers[i]
        mn   = float(ys.data_min_[0])
        rng_ = float(ys.data_range_[0])
        thr_s = (THRESHOLDS[app] - mn) / (rng_ + 1e-12)

        lbl_tr = (Y_tr[idx_tr, i] > thr_s).astype(int)
        lbl_te = (Y_te[idx_te, i] > thr_s).astype(int)

        for row, (lbl, H2, split_name) in enumerate([
                (lbl_tr, Htr2, 'train'), (lbl_te, Hte2, 'test')]):
            ax = axes[row][i]
            ax.scatter(H2[:, 0], H2[:, 1],
                       c=np.where(lbl, 'tomato', 'steelblue'), s=8, alpha=0.5)
            ax.set_title(f"{app} / {split_name}")
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
            from matplotlib.lines import Line2D
            ax.legend(handles=[
                Line2D([0],[0], marker='o', color='w', markerfacecolor='tomato',  ms=8, label='ON'),
                Line2D([0],[0], marker='o', color='w', markerfacecolor='steelblue',ms=8, label='OFF'),
            ], fontsize=8)
            ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pca_hidden.png'), dpi=150)
    plt.close()

    var_exp = pca.explained_variance_ratio_
    print(f"   Variance explained: PC1={var_exp[0]:.3f}  PC2={var_exp[1]:.3f}")
    return {'pca_var_pc1': float(var_exp[0]), 'pca_var_pc2': float(var_exp[1])}


# ============================================================================
# 9.  MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir',   required=True,
                        help='Directory containing best_model.pt '
                             '(output of test_pinn_lnn_ukdale_medium_dataset.py)')
    parser.add_argument('--dataset-dir', default=DEFAULT_DATASET_DIR)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--n-probe',     type=int, default=2000)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    out_dir = os.path.join(args.model_dir, 'repr_analysis')
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Load data ────────────────────────────────────────────────────────────
    print("\nLoading and preprocessing data ...")
    (X_tr, Y_tr, X_va, Y_va, X_te, Y_te,
     x_scaler, y_scalers, _) = load_and_preprocess(args.dataset_dir)
    print(f"  X_tr={X_tr.shape}  X_va={X_va.shape}  X_te={X_te.shape}")

    # ── Load model ───────────────────────────────────────────────────────────
    ckpt_path = os.path.join(args.model_dir, 'best_model.pt')
    if not os.path.exists(ckpt_path):
        print(f"ERROR: {ckpt_path} not found. "
              f"Re-run training — best_model.pt is saved automatically.")
        sys.exit(1)

    model = PhysicsInformedLiquidNetworkModel(
        input_size=1, hidden_size=args.hidden_size,
        n_appliances=len(APPLIANCES), dt=0.1,
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"Loaded model from {ckpt_path}")

    # ── Collect activations ──────────────────────────────────────────────────
    print("\nCollecting activations ...")
    collector = ActivationCollector(model, device)
    acts_tr = collector.collect(X_tr)
    print("  train done")
    acts_va = collector.collect(X_va)
    print("  val done")
    acts_te = collector.collect(X_te)
    print("  test done")

    # ── Run analyses ─────────────────────────────────────────────────────────
    all_results = {}
    all_results['weights']     = analyse_weights(model, out_dir)
    all_results['hidden']      = analyse_hidden_states(acts_tr, acts_va, acts_te, out_dir)
    all_results['pred_dist']   = analyse_prediction_distributions(
                                     acts_tr, acts_va, acts_te,
                                     Y_tr, Y_va, Y_te, y_scalers, out_dir)
    all_results['probing']     = linear_probe(
                                     acts_tr, acts_va, acts_te,
                                     Y_tr, Y_va, Y_te,
                                     y_scalers, args.n_probe, out_dir)
    all_results['attribution'] = temporal_attribution(
                                     model, X_te, Y_te,
                                     y_scalers, device, out_dir)
    all_results['pca']         = pca_hidden_states(
                                     acts_tr, acts_va, acts_te,
                                     Y_tr, Y_va, Y_te, y_scalers, out_dir)

    # ── Save JSON ────────────────────────────────────────────────────────────
    def _safe(obj):
        if isinstance(obj, dict):  return {k: _safe(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [_safe(v) for v in obj]
        if isinstance(obj, (np.integer,)):   return int(obj)
        if isinstance(obj, (np.floating, float)):
            return None if (isinstance(obj, float) and np.isnan(obj)) else float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    out_json = os.path.join(out_dir, 'repr_analysis.json')
    with open(out_json, 'w') as f:
        json.dump(_safe(all_results), f, indent=2)
    print(f"\nAnalysis complete. Outputs saved to: {out_dir}")

    # ── Diagnostic summary ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 65)

    h = all_results['hidden']
    print(f"\n>> Representational drift (CKA, h_final):")
    print(f"   CKA(train,val)  = {h['cka_train_val']:.4f}  (1.0=identical)")
    print(f"   CKA(train,test) = {h['cka_train_test']:.4f}")
    print(f"   Centroid train→val  = {h['centroid_train_val']:.4f}")
    print(f"   Centroid train→test = {h['centroid_train_test']:.4f}")

    print(f"\n>> Spectral radius: {all_results['weights']['rec_spectral_radius']:.4f}  "
          f"(>1 → potentially unstable hidden state)")

    print("\n>> Prediction ON-fraction vs true ON-fraction (test split):")
    for app, r in all_results['pred_dist'].items():
        te = r['test']
        print(f"   {app:<18}  pred_ON={te['frac_above_thr']:.3f}  "
              f"true_ON={te['true_frac_on']:.3f}  "
              f"ratio={te['frac_above_thr']/(te['true_frac_on']+1e-9):.2f}x")

    print("\n>> Linear probe AUC-ROC (h_final):")
    for app, r in all_results['probing'].items():
        print(f"   {app:<18}  train={r['auc_train']:.4f}  "
              f"val={r['auc_val']:.4f}  test={r['auc_test']:.4f}")

    print("\n>> FP temporal attribution (peak timestep relative to midpoint):")
    for app, r in all_results['attribution'].items():
        if r is None:
            print(f"   {app}: no FP windows")
            continue
        print(f"   {app:<18}  n_FP={r['n_fp']}  "
              f"peak at t={r['peak_timestep_relative']:+d} steps")


if __name__ == '__main__':
    main()
