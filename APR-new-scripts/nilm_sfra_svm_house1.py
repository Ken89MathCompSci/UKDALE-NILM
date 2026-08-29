"""
nilm_sfra_svm_house1.py
========================
SVM baseline for NILM on APR-new-House1-dataset (House 1, Oct 2014).

Two independent SVM heads per appliance, mirroring the classification +
regression framing of the gated dual-head PINN-LNN scripts:
  - SVC (RBF kernel)  ->  ON/OFF state
  - SVR (RBF kernel)  ->  power draw (Watts)

Features: the same 8-channel aggregate features used by the PINN-LNN scripts
(raw, median, EMA, residual, Δraw, Δsmooth, rolling_mean, rolling_std),
Z-scored with StandardScaler fit on the training split only.

Adaptive thresholds are computed on train only and reused for test
(same train-locked convention as combined_pinn_lnn_apr_new_house1_dataset.py).

Training set is subsampled to MAX_TRAIN_SAMPLES for SVC tractability;
full prediction is run on the complete test set.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


APPLIANCES  = ['dishwasher', 'fridge', 'microwave', 'washing_machine']
AGG_COL     = 'aggregate'

MEDFILT_K   = 5
EMA_SPAN    = 10
ROLL_K      = 10

THRESHOLD_DELTA   = 20.0
THRESHOLD_LOW_PCT = 0.05
THRESHOLD_MIN     = 10.0
CYCLING_P5_W      = 80.0

MAX_TRAIN_SAMPLES = 20000
RANDOM_SEED       = 42

DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'APR-new-House1-dataset')


# ---------------------------------------------------------------------------
# Feature extraction  (identical to combined_pinn_lnn_apr_new_house1_dataset.py)
# ---------------------------------------------------------------------------

def _median_filter(arr, k):
    return pd.Series(arr).rolling(k, center=True, min_periods=1).median().values.astype(np.float32)

def _ema_filter(arr, span):
    return pd.Series(arr).ewm(span=span, adjust=False).mean().values.astype(np.float32)

def _n_step_diff(arr, n):
    d = np.zeros_like(arr); d[n:] = arr[n:] - arr[:-n]; return d

def compute_features(mains: np.ndarray) -> np.ndarray:
    raw    = mains.astype(np.float32)
    med    = _median_filter(raw, MEDFILT_K)
    smooth = _ema_filter(med, EMA_SPAN)
    resid  = (raw - smooth).astype(np.float32)
    d_raw    = _n_step_diff(raw,    1)
    d_smooth = _n_step_diff(smooth, 1)
    s  = pd.Series(smooth)
    rm = s.rolling(ROLL_K, min_periods=1).mean().values.astype(np.float32)
    rs = s.rolling(ROLL_K, min_periods=1).std().fillna(0).values.astype(np.float32)
    return np.stack([raw, med, smooth, resid, d_raw, d_smooth, rm, rs], axis=1)


# ---------------------------------------------------------------------------
# Adaptive thresholds  (identical to combined_pinn_lnn_apr_new_house1_dataset.py)
# ---------------------------------------------------------------------------

def compute_adaptive_thresholds(df: pd.DataFrame) -> dict:
    thresholds = {}
    for app in APPLIANCES:
        col     = df[app]
        nonzero = col[col > 0]
        if len(nonzero) == 0:
            thresholds[app] = THRESHOLD_MIN
            continue
        p5 = float(nonzero.quantile(THRESHOLD_LOW_PCT))
        thresholds[app] = THRESHOLD_MIN if p5 > CYCLING_P5_W else max(p5 + THRESHOLD_DELTA, THRESHOLD_MIN)
    return thresholds


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(dataset_dir: str) -> dict:
    print(f"Loading APR-new-House1-dataset from '{dataset_dir}' ...")
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
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_appliance(appliance, X_test, y_state_test, y_power_test,
                        clf, reg, reg_scaler_y):
    y_state_pred = clf.predict(X_test)

    f1   = f1_score(y_state_test, y_state_pred, zero_division=0)
    prec = precision_score(y_state_test, y_state_pred, zero_division=0)
    rec  = recall_score(y_state_test, y_state_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_state_test, y_state_pred, labels=[0, 1]).ravel()

    y_power_pred_s = reg.predict(X_test)
    y_power_pred   = reg_scaler_y.inverse_transform(y_power_pred_s.reshape(-1, 1)).ravel()
    y_power_pred   = np.clip(y_power_pred, 0, None)

    mae = np.mean(np.abs(y_power_test - y_power_pred))

    N = 100
    num_period = int(len(y_power_test) / N)
    diff = 0
    for i in range(num_period):
        diff += abs(np.sum(y_power_test[i*N:(i+1)*N]) - np.sum(y_power_pred[i*N:(i+1)*N]))
    sae = diff / (N * num_period) if num_period > 0 else 0.0

    return {
        'appliance': appliance, 'f1': f1, 'prec': prec, 'rec': rec,
        'mae': mae, 'sae': sae,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }


def print_results_table(results):
    header = (f"{'Appliance':<18}{'F1':>8}{'Prec':>8}{'Rec':>8}"
              f"{'MAE':>8}{'SAE':>8}{'TP':>10}{'TN':>10}{'FP':>10}{'FN':>10}")
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['appliance']:<18}{r['f1']:>8.4f}{r['prec']:>8.4f}{r['rec']:>8.4f}"
              f"{r['mae']:>8.2f}{r['sae']:>8.4f}"
              f"{r['tp']:>10,}{r['tn']:>10,}{r['fp']:>10,}{r['fn']:>10,}")
    print("-" * len(header))
    print(f"{'MACRO AVG':<18}"
          f"{np.mean([r['f1']   for r in results]):>8.4f}"
          f"{np.mean([r['prec'] for r in results]):>8.4f}"
          f"{np.mean([r['rec']  for r in results]):>8.4f}"
          f"{np.mean([r['mae']  for r in results]):>8.2f}"
          f"{np.mean([r['sae']  for r in results]):>8.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dataset_dir: str = DEFAULT_DATASET_DIR, tune: bool = False,
         max_train_samples: int = MAX_TRAIN_SAMPLES):

    rng    = np.random.default_rng(RANDOM_SEED)
    splits = load_data(dataset_dir)
    df_tr  = splits['train']
    df_te  = splits['test']

    thresholds = compute_adaptive_thresholds(df_tr)
    print("\nAdaptive thresholds (train-locked):")
    for app in APPLIANCES:
        print(f"  {app:<18} {thresholds[app]:.1f} W")

    print("\nComputing features ...")
    X_tr_raw = compute_features(df_tr[AGG_COL].values)
    X_te_raw = compute_features(df_te[AGG_COL].values)

    # Z-score per channel, fit on train only
    feat_scaler = StandardScaler().fit(X_tr_raw)
    X_tr = feat_scaler.transform(X_tr_raw)
    X_te = feat_scaler.transform(X_te_raw)

    # Subsample train for SVC tractability
    n_tr = len(X_tr)
    if n_tr > max_train_samples:
        idx = rng.choice(n_tr, size=max_train_samples, replace=False)
    else:
        idx = np.arange(n_tr)
    X_tr_sub = X_tr[idx]
    print(f"  Train: {n_tr:,} samples -> using {len(X_tr_sub):,} for SVM")
    print(f"  Test : {len(X_te):,} samples\n")

    results = []
    for app in APPLIANCES:
        print(f"  [{app}]")
        y_power_tr_full = df_tr[app].values.astype(np.float32)
        y_power_te      = df_te[app].values.astype(np.float32)

        y_power_tr_sub = y_power_tr_full[idx]
        y_state_tr_sub = (y_power_tr_sub > thresholds[app]).astype(int)
        y_state_te     = (y_power_te     > thresholds[app]).astype(int)

        print(f"    Train ON%={100*y_state_tr_sub.mean():.1f}%  "
              f"Test ON%={100*y_state_te.mean():.1f}%")

        # SVC (class_weight='balanced' handles imbalance automatically)
        if tune:
            from sklearn.model_selection import GridSearchCV
            clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                               {'C': [1, 10, 100], 'gamma': ['scale', 0.01, 0.1]},
                               cv=3, scoring='f1', n_jobs=-1)
            clf.fit(X_tr_sub, y_state_tr_sub)
            clf = clf.best_estimator_
        else:
            clf = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced')
            clf.fit(X_tr_sub, y_state_tr_sub)

        # SVR -- StandardScaler on power target
        reg_scaler_y = StandardScaler().fit(y_power_tr_sub.reshape(-1, 1))
        y_power_tr_s = reg_scaler_y.transform(y_power_tr_sub.reshape(-1, 1)).ravel()

        if tune:
            from sklearn.model_selection import GridSearchCV
            reg = GridSearchCV(SVR(kernel='rbf'),
                               {'C': [1, 10, 100], 'gamma': ['scale', 0.01, 0.1],
                                'epsilon': [0.01, 0.1]},
                               cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
            reg.fit(X_tr_sub, y_power_tr_s)
            reg = reg.best_estimator_
        else:
            reg = SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.05)
            reg.fit(X_tr_sub, y_power_tr_s)

        results.append(evaluate_appliance(
            app, X_te, y_state_te, y_power_te, clf, reg, reg_scaler_y
        ))

    print()
    print_results_table(results)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path  = os.path.join(dataset_dir, f'svm_house1_results_{timestamp}.json')
    with open(out_path, 'w') as f:
        json.dump({
            'dataset': 'APR-new-House1-dataset',
            'thresholds': thresholds,
            'max_train_samples': max_train_samples,
            'results': [{k: float(v) if isinstance(v, (float, np.floating)) else v
                         for k, v in r.items()} for r in results],
        }, f, indent=2)
    print(f"\nResults saved -> {out_path}")
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', default=DEFAULT_DATASET_DIR)
    parser.add_argument('--tune', action='store_true',
                        help='Grid-search C/gamma/epsilon (slow)')
    parser.add_argument('--max-train-samples', type=int, default=MAX_TRAIN_SAMPLES)
    args = parser.parse_args()
    main(dataset_dir=args.dataset_dir, tune=args.tune,
         max_train_samples=args.max_train_samples)
