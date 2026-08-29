"""
nilm_sfra_svm_house2.py
========================
House 2 counterpart to APR-new-scripts/nilm_sfra_svm_house1.py -- wires
MultiLabelApplianceSVM (APR-new-scripts/nilm_sfra_svm.py) to real
APR-new-House2-dataset data instead of the synthetic demo.

IMPORTANT -- this is NOT real SFRA data.
-----------------------------------------
The Mari et al. 2023 paper's SVM operates on genuine Swept-Frequency
Response Analysis traces: a signal generator injects a frequency sweep
(kHz range) onto the mains and measures the circuit's impedance response,
which is a highly appliance-specific electrical fingerprint. UKDALE (and
this repo's UKDALE_HF_*.csv files) contains none of that -- only 6-second
active-power samples.

What this script does instead is build a *proxy* frequency-domain feature
vector by taking the FFT magnitude spectrum of short windows of the
aggregate active-power signal. This gives the SVM the same *shape* of
input the paper expects (one feature per frequency bin) but the frequency
axis here is minutes-scale power fluctuation, not kHz-scale circuit
impedance. Treat results as a structural proof-of-concept for the
SVM/metrics pipeline, not as a reproduction of the paper's accuracy --
see the House 1 run for a worked example of how weak this proxy is for
a low-duty-cycle appliance like dishwasher.

Pipeline (identical to the House 1 version):
  1. Load House 2 train/val/test splits (same CSVs as the PINN-LNN scripts).
  2. Slide a WIN_MINUTES-long window over the aggregate power signal
     (TRAIN_STRIDE_MINUTES on train for overlap augmentation, non-overlapping
     on val/test).
  3. Feature = log1p(|rFFT(window - window.mean())|) -- one value per
     frequency bin, DC-removed so the spectrum reflects fluctuation shape
     rather than absolute power level.
  4. Label per appliance per window = ON if the appliance's ON-fraction
     within the window exceeds ON_FRACTION_THRESHOLD (adaptive per-appliance
     power threshold from compute_adaptive_thresholds, computed on train
     only and reused for val/test -- same threshold-locking convention as
     combined_pinn_lnn_apr_new_house2_dataset_v2.py's FIX 1).
  5. Train one binary SVM per appliance (MultiLabelApplianceSVM) on train,
     evaluate on val and test with the paper's Eq. (8)-(16) metrics
     (per-appliance + micro/macro averaged).

NOTE on House 2's fridge threshold: House 2's fridge never reads 0W (idle
floor ~9-11W), so compute_adaptive_thresholds (imported from the v2
script, same as the PINN-LNN pipeline) uses a 20W threshold to detect
compressor cycling rather than the standard 10W. See that script's
THRESHOLDS docstring for details.

Backend / data-volume note
---------------------------
Default backend is "approx" (Nystroem kernel approximation + linear
SGDClassifier, see nilm_sfra_svm.py) trained on the FULL train-window set
(no subsampling) -- this makes it a fair apples-to-apples comparison
against the PINN-LNN, which also trains on all the data, rather than
against a small SVC subsample. Pass --backend svc to use the exact
polynomial-kernel SVC instead; that backend still needs --max-train-windows
capped (default 2000) since its training cost scales far worse than cubic
in n on the weakly-separable classes here.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

# nilm_sfra_svm.py lives in APR-new-scripts/, not this folder.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'APR-new-scripts'))
sys.path.insert(0, os.path.dirname(__file__))
from nilm_sfra_svm import MultiLabelApplianceSVM, per_appliance_metrics, micro_average, macro_average

# Reuse the (train-locked) adaptive threshold logic from the House 2 v2
# script instead of duplicating it -- same threshold-consistency lesson
# (FIX 1) as the House 1 pipeline.
from combined_pinn_lnn_apr_new_house2_dataset_v2 import compute_adaptive_thresholds, APPLIANCES, AGG_COL


DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'APR-new-House2-dataset')

STEP_SECONDS          = 6.0
WIN_MINUTES            = 6.0    # proxy-SFRA analysis window length
TRAIN_STRIDE_MINUTES   = 3.0    # 50% overlap on train (augmentation)
ON_FRACTION_THRESHOLD  = 0.05   # window labeled ON if appliance exceeds its
                                 # power threshold for >=5% of the window

DEFAULT_BACKEND        = "approx"  # "approx" trains on the full window set (no cap
                                    # needed below); "svc"/"scratch" do not scale and
                                    # need MAX_TRAIN_WINDOWS_SVC_DEFAULT.
N_COMPONENTS           = 300    # Nystroem feature-map dimensionality (approx backend only)
ALPHA                  = 1e-4   # SGDClassifier L2 regularization (approx backend only)

MAX_TRAIN_WINDOWS_SVC_DEFAULT = 2000   # fallback subsample cap, backend="svc"/"scratch" only.
                                 # NOTE: exact SVC fit time on these proxy features scales
                                 # far worse than cubic in n (House 1: 800->0.9s, 3000->205s),
                                 # driven almost entirely by whichever appliance is closest
                                 # to a balanced class split (fridge in both houses -- see
                                 # the printed per-appliance ON% and fit times before raising
                                 # this, or just use backend="approx" instead).
RANDOM_SEED            = 0


# ---------------------------------------------------------------------------
# Data loading (same file layout as the PINN-LNN House 2 scripts)
# ---------------------------------------------------------------------------

def load_data(dataset_dir: str) -> dict:
    print(f"Loading APR-new-House2-dataset CSV data from '{dataset_dir}' ...")
    file_map = {'train': 'UKDALE_HF_train.csv',
                'val':   'UKDALE_HF_validation.csv',
                'test':  'UKDALE_HF_test.csv'}
    splits = {}
    for name, fname in file_map.items():
        path = os.path.join(dataset_dir, fname)
        splits[name] = pd.read_csv(path, index_col='timestamp', parse_dates=True)
        df = splits[name]
        print(f"  {name:6s}: {len(df):>7,} rows  "
              f"{df.index.min().date()} -> {df.index.max().date()}")
    return splits


# ---------------------------------------------------------------------------
# Proxy-SFRA feature extraction + windowed labeling
# ---------------------------------------------------------------------------

def fft_feature(window_power: np.ndarray) -> np.ndarray:
    """log1p(|rFFT(window - mean)|) -- one feature per frequency bin, DC removed."""
    centered = window_power - window_power.mean()
    spec = np.abs(np.fft.rfft(centered))
    return np.log1p(spec).astype(np.float32)


def build_windows(df: pd.DataFrame, thresholds: dict, win_steps: int, stride_steps: int):
    """
    Slide a win_steps-long window over df[AGG_COL] with the given stride.

    Returns:
        X : (n_windows, win_steps//2 + 1) FFT magnitude features
        Y : (n_windows, n_appliances) binary ON/OFF labels
    """
    agg  = df[AGG_COL].values.astype(np.float32)
    apps = {app: df[app].values.astype(np.float32) for app in APPLIANCES}
    n = len(agg)

    X, Y = [], []
    for start in range(0, n - win_steps + 1, stride_steps):
        end = start + win_steps
        X.append(fft_feature(agg[start:end]))
        row = []
        for app in APPLIANCES:
            on_frac = float(np.mean(apps[app][start:end] > thresholds[app]))
            row.append(1 if on_frac >= ON_FRACTION_THRESHOLD else 0)
        Y.append(row)

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.int64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dataset_dir: str = DEFAULT_DATASET_DIR,
        degree: int = 3, coef0: float = 1.0, C: float = 1.0,
        backend: str = DEFAULT_BACKEND,
        n_components: int = N_COMPONENTS, alpha: float = ALPHA,
        class_weight: str | None = None,
        max_train_windows: int | None = None):

    rng = np.random.default_rng(RANDOM_SEED)

    data = load_data(dataset_dir)
    df_tr, df_va, df_te = data['train'], data['val'], data['test']

    win_steps    = int(round(WIN_MINUTES * 60.0 / STEP_SECONDS))
    tr_stride    = int(round(TRAIN_STRIDE_MINUTES * 60.0 / STEP_SECONDS))
    eval_stride  = win_steps  # non-overlapping on val/test

    # Thresholds computed on train only, reused for val/test (avoids the
    # moving-goalpost problem FIX 1 addressed in the PINN-LNN v2 scripts)
    thresholds = compute_adaptive_thresholds(df_tr)
    print(f"\nAdaptive thresholds (train-locked): "
          + ", ".join(f"{app}={thresholds[app]:.1f}W" for app in APPLIANCES))
    print(f"Window: {WIN_MINUTES:.1f} min ({win_steps} steps)  "
          f"train_stride={TRAIN_STRIDE_MINUTES:.1f} min  on_fraction>={ON_FRACTION_THRESHOLD:.0%}\n")

    print("Building windows ...")
    X_tr, Y_tr = build_windows(df_tr, thresholds, win_steps, tr_stride)
    X_va, Y_va = build_windows(df_va, thresholds, win_steps, eval_stride)
    X_te, Y_te = build_windows(df_te, thresholds, win_steps, eval_stride)
    print(f"  Train windows: {X_tr.shape}  Val windows: {X_va.shape}  Test windows: {X_te.shape}")

    for split_name, Y in [('train', Y_tr), ('val', Y_va), ('test', Y_te)]:
        print(f"  {split_name:5s} ON-fraction per appliance: " +
              ", ".join(f"{app}={100*Y[:, i].mean():.1f}%" for i, app in enumerate(APPLIANCES)))

    # Only the exact "svc"/"scratch" backends need subsampling (training cost
    # scales far worse than cubic in n on these weakly-separable classes).
    # "approx" trains on the full window set by default.
    if max_train_windows is None and backend != "approx":
        max_train_windows = MAX_TRAIN_WINDOWS_SVC_DEFAULT

    if max_train_windows is not None and X_tr.shape[0] > max_train_windows:
        idx = rng.choice(X_tr.shape[0], size=max_train_windows, replace=False)
        X_tr, Y_tr = X_tr[idx], Y_tr[idx]
        print(f"\n  Subsampled train windows to {max_train_windows:,} "
              f"(random, seed={RANDOM_SEED}) for {backend} training tractability.")
        print(f"  Subsampled ON-fraction: " +
              ", ".join(f"{app}={100*Y_tr[:, i].mean():.1f}%" for i, app in enumerate(APPLIANCES)))
    else:
        print(f"\n  Training on the full {X_tr.shape[0]:,} windows (no subsampling, backend={backend}).")

    # Feature standardization (zero mean / unit variance per frequency bin) --
    # fit on train only, applied to val/test.
    feat_mean = X_tr.mean(axis=0, keepdims=True)
    feat_std  = X_tr.std(axis=0, keepdims=True) + 1e-8
    X_tr = (X_tr - feat_mean) / feat_std
    X_va = (X_va - feat_mean) / feat_std
    X_te = (X_te - feat_mean) / feat_std

    if backend == "approx":
        print(f"\nTraining MultiLabelApplianceSVM "
              f"(backend=approx: Nystroem poly degree={degree} coef0={coef0} "
              f"n_components={n_components} + SGDClassifier alpha={alpha} "
              f"class_weight={class_weight}) ...")
    else:
        print(f"\nTraining MultiLabelApplianceSVM "
              f"(backend={backend}, poly degree={degree}, coef0={coef0}, C={C}) ...")
    system = MultiLabelApplianceSVM(APPLIANCES, degree=degree, coef0=coef0, C=C,
                                    backend=backend, n_components=n_components,
                                    alpha=alpha, class_weight=class_weight,
                                    random_state=RANDOM_SEED)
    t0 = time.time()
    for j, app in enumerate(APPLIANCES):
        t_app = time.time()
        model = system._make_model()
        model.fit(X_tr, Y_tr[:, j])
        system.models[app] = model
        print(f"    {app:<18} fit time: {time.time() - t_app:6.1f}s")
    print(f"  Total fit time: {time.time() - t0:.1f}s")

    def evaluate(split_name, X, Y):
        Y_pred = system.predict(X)
        per_app = per_appliance_metrics(Y, Y_pred, APPLIANCES)
        mic = micro_average(per_app)
        mac = macro_average(per_app)
        print(f"\n{split_name.upper()}  ({X.shape[0]:,} windows)"
              f"  [MAE/SAE are window-level on binary ON/OFF labels, not Watts -- see "
              f"mae_score/sae_score docstrings in nilm_sfra_svm.py]")
        print(f"  {'Appliance':<18} {'P':>6} {'R':>6} {'F1':>6} {'MAE':>6} {'SAE':>6} "
              f"{'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6}")
        for app, m in per_app.items():
            print(f"  {app:<18} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} "
                  f"{m['mae']:>6.3f} {m['sae']:>6.3f} "
                  f"{m['tp']:>6d} {m['fp']:>6d} {m['fn']:>6d} {m['tn']:>6d}")
        print(f"  {'micro-avg':<18} {mic['precision']:>6.3f} {mic['recall']:>6.3f} {mic['f1']:>6.3f}")
        print(f"  {'macro-avg':<18} {mac['precision']:>6.3f} {mac['recall']:>6.3f} {mac['f1']:>6.3f}")
        return per_app, mic, mac

    val_per_app,  val_mic,  val_mac  = evaluate('val',  X_va, Y_va)
    test_per_app, test_mic, test_mac = evaluate('test', X_te, Y_te)

    results = {
        'note': 'Proxy-SFRA (FFT of aggregate power windows), NOT real SFRA -- see module docstring.',
        'window': {'win_minutes': WIN_MINUTES, 'train_stride_minutes': TRAIN_STRIDE_MINUTES,
                   'on_fraction_threshold': ON_FRACTION_THRESHOLD},
        'thresholds_w': thresholds,
        'svm_params': {'backend': backend, 'degree': degree, 'coef0': coef0, 'C': C,
                       'kernel': 'poly', 'n_components': n_components, 'alpha': alpha,
                       'class_weight': class_weight},
        'max_train_windows': max_train_windows,
        'train_windows_used': int(X_tr.shape[0]),
        'val':  {'per_appliance': val_per_app,  'micro': val_mic,  'macro': val_mac},
        'test': {'per_appliance': test_per_app, 'micro': test_mic, 'macro': test_mac},
    }
    out_path = os.path.join(os.path.dirname(__file__), '..', 'APR-new-House2-dataset',
                            'sfra_svm_house2_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved -> {out_path}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', default=DEFAULT_DATASET_DIR)
    parser.add_argument('--backend', choices=['approx', 'svc', 'scratch'], default=DEFAULT_BACKEND,
                        help='approx trains on the full dataset (default); svc/scratch use the '
                             'exact polynomial kernel and need --max-train-windows capped.')
    parser.add_argument('--degree', type=int, default=3)
    parser.add_argument('--coef0',  type=float, default=1.0)
    parser.add_argument('--C',      type=float, default=1.0)
    parser.add_argument('--n-components', type=int, default=N_COMPONENTS,
                        help='Nystroem feature-map dimensionality (approx backend only).')
    parser.add_argument('--alpha', type=float, default=ALPHA,
                        help='SGDClassifier L2 regularization (approx backend only).')
    parser.add_argument('--class-weight', choices=['none', 'balanced'], default='none',
                        help="approx backend only. 'none' (default) matches sklearn's own "
                             "default but can fully collapse to predicting OFF on a severely "
                             "imbalanced appliance (measured: House 2 microwave -> F1=0.000). "
                             "'balanced' fixes that collapse but craters precision on the rare "
                             "classes instead (measured: 0.03-0.21 precision) -- net macro-F1 "
                             "was slightly worse on both houses in testing, not a free win. "
                             "See MultiLabelApplianceSVM's class_weight docstring for details.")
    parser.add_argument('--max-train-windows', type=int, default=None,
                        help=f'Cap on train windows (random subsample). Default: no cap for '
                             f'backend=approx; {MAX_TRAIN_WINDOWS_SVC_DEFAULT} for backend=svc/scratch.')
    args = parser.parse_args()

    main(dataset_dir=args.dataset_dir, degree=args.degree, coef0=args.coef0, C=args.C,
        backend=args.backend, n_components=args.n_components, alpha=args.alpha,
        class_weight=None if args.class_weight == 'none' else args.class_weight,
        max_train_windows=args.max_train_windows)
