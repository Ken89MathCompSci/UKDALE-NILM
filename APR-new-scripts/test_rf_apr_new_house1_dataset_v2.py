"""
test_rf_apr_new_house1_dataset_v2.py
======================================
v2 of test_rf_apr_new_house1_dataset.py -- two targeted fixes for the
precision collapse in v1:

    Appliance            F1  Precision   Recall     MAE     SAE
    dishwasher       0.2871     0.1704   0.9110    6.94   6.4727
    fridge           0.7240     0.5695   0.9936   30.68  18.4420
    microwave        0.2290     0.1311   0.9063   19.67   8.0324
    washing_machine  0.8641     0.7630   0.9962   20.45   6.9240

Same "recall~1.0, precision collapses on the rare classes" signature as
every neural baseline in this folder (Transformer/ResNet/TCN v1).

FIX 1 -- Two-stage classifier gate (main fix)
    v1's own docstring documents a fix that was already tried and
    REJECTED: applying inverse-frequency sample_weight directly to the
    RandomForestRegressor made every rare appliance worse (dishwasher F1
    0.31->0.22, microwave F1 0.31->0.20), because upweighting ON samples
    in a squared-error loss inflates a small positive bias across nearly
    all OFF windows -- enough for many of them to clear the threshold.
    That's the regression analogue of the same failure this project keeps
    finding: naive imbalance correction on a model with no explicit
    detection stage backfires.

    The fix that has worked everywhere else in this project (PINN-LNN's
    gate, every Gated Transformer/ResNet/TCN v2 script) is to add an
    explicit ON/OFF detector and gate the power estimate with it, rather
    than reweighting the regressor itself. RF's natural equivalent: a
    second model, RandomForestClassifier, predicting ON/OFF, with
    class_weight='balanced' -- gating the regressor's output:
        gated_power = classifier.predict_proba(X)[:, 1] * regressor.predict(X)
    class_weight='balanced' is safe on a classifier in a way sample_weight
    was NOT safe on the regressor: tree splits there are chosen by
    weighted class purity (Gini) at each node, not by minimizing a global
    squared-error surface that a few upweighted extreme residuals can
    skew across every leaf. This is tested below, not just assumed --
    both pre-gate and post-gate metrics are printed so the effect is
    directly visible.

FIX 2 -- Adaptive per-appliance thresholds (train-locked)
    v1 used a flat 10W threshold for all four appliances (THRESHOLD_W =
    10.0). House 1's own adaptive thresholds (same formula used throughout
    this repo) are 21W / 10W / 21W / 23W for dishwasher / fridge /
    microwave / washing_machine -- 10W is only correct for fridge. Fix:
    compute_adaptive_thresholds(), train-locked, used for the classifier's
    ON/OFF training target AND the reported val/test metrics.

Everything else -- WIN=100/STRIDE=5 windowed features (mean, std, min,
max, range, median, first, last, mean|diff|, max|diff|), n_estimators=300,
max_depth=20, min_samples_leaf=5, random_state=42, unweighted regressor --
is unchanged from v1.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Source Code'))
from utils import calculate_nilm_metrics

DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'APR-new-House1-dataset')
APPLIANCES  = ['dishwasher', 'fridge', 'microwave', 'washing_machine']

# FIX 2: adaptive-threshold constants (identical to combined_pinn_lnn_apr_new_house1_dataset*.py)
THRESHOLD_DELTA   = 20.0
THRESHOLD_LOW_PCT = 0.05
THRESHOLD_MIN     = 10.0
CYCLING_P5_W      = 80.0

WIN    = 100
STRIDE = 5

N_ESTIMATORS     = 300
MAX_DEPTH        = 20
MIN_SAMPLES_LEAF = 5
RANDOM_STATE     = 42

FEATURE_NAMES = ['mean', 'std', 'min', 'max', 'range', 'median',
                  'first', 'last', 'mean_abs_diff', 'max_abs_diff']


def compute_adaptive_thresholds(df: pd.DataFrame) -> dict:
    """Identical formula to combined_pinn_lnn_apr_new_house1_dataset*.py."""
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


def load_data(dataset_dir=DATASET_DIR):
    print(f"Loading data from: {os.path.abspath(dataset_dir)}")

    def _load(filename):
        path = os.path.join(dataset_dir, filename)
        return pd.read_csv(path, index_col='timestamp', parse_dates=True)

    train_data = _load('UKDALE_HF_train.csv')
    val_data   = _load('UKDALE_HF_validation.csv')
    test_data  = _load('UKDALE_HF_test.csv')

    for name, df in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
        print(f"  {name}: {df.index.min()} to {df.index.max()}  ({len(df):,} rows)")

    return {'train': train_data, 'val': val_data, 'test': test_data}


def build_window_features(mains: np.ndarray, appliance_vals: np.ndarray,
                          win: int = WIN, stride: int = STRIDE):
    """
    Windowed statistical features from the aggregate signal, midpoint-
    targeted (same alignment convention as create_sequences() elsewhere in
    this folder). Unchanged from v1.
    """
    all_windows    = sliding_window_view(mains, win)          # (len(mains)-win+1, win)
    start_indices  = np.arange(0, len(mains) - win, stride)
    windows        = all_windows[start_indices]                # (n_windows, win)
    mid_indices    = start_indices + win // 2

    diffs = np.diff(windows, axis=1)
    feat = np.stack([
        windows.mean(axis=1),
        windows.std(axis=1),
        windows.min(axis=1),
        windows.max(axis=1),
        windows.max(axis=1) - windows.min(axis=1),
        np.median(windows, axis=1),
        windows[:, 0],
        windows[:, -1],
        np.abs(diffs).mean(axis=1),
        np.abs(diffs).max(axis=1),
    ], axis=1).astype(np.float32)

    y_power = appliance_vals[mid_indices].astype(np.float32)
    return feat, y_power


def train_rf_on_appliance(data_dict, appliance_name, thresholds, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    train_data = data_dict['train']
    val_data   = data_dict['val']
    test_data  = data_dict['test']
    threshold  = thresholds[appliance_name]  # FIX 2: train-locked adaptive threshold

    print(f"\nBuilding windowed features for {appliance_name}...  (threshold={threshold:.1f}W)")
    X_tr, y_tr = build_window_features(train_data['aggregate'].values, train_data[appliance_name].values)
    X_va, y_va = build_window_features(val_data['aggregate'].values,   val_data[appliance_name].values)
    X_te, y_te = build_window_features(test_data['aggregate'].values,  test_data[appliance_name].values)

    y_on_tr = (y_tr > threshold).astype(int)
    y_on_va = (y_va > threshold).astype(int)
    y_on_te = (y_te > threshold).astype(int)

    on_tr = y_on_tr.mean() * 100
    on_va = y_on_va.mean() * 100
    on_te = y_on_te.mean() * 100
    print(f"  Train: {X_tr.shape}  ON={on_tr:.2f}%")
    print(f"  Val:   {X_va.shape}  ON={on_va:.2f}%")
    print(f"  Test:  {X_te.shape}  ON={on_te:.2f}%")

    # -- FIX 1: regressor (unweighted, unchanged from v1 -- sample_weight
    #    already proven to backfire, see module docstring) --
    regressor = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    print(f"Fitting RandomForestRegressor ({N_ESTIMATORS} trees) for {appliance_name}...")
    regressor.fit(X_tr, y_tr)

    # -- FIX 1: gate classifier (class_weight='balanced' -- safe on a
    #    classifier in a way sample_weight was not safe on the regressor,
    #    see module docstring) --
    classifier = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced',
    )
    print(f"Fitting RandomForestClassifier gate ({N_ESTIMATORS} trees) for {appliance_name}...")
    classifier.fit(X_tr, y_on_tr)

    def gated_predict(X):
        gate_proba = classifier.predict_proba(X)[:, 1]
        power      = regressor.predict(X)
        return gate_proba * power

    val_pred_raw  = regressor.predict(X_va)
    test_pred_raw = regressor.predict(X_te)
    val_pred_gated  = gated_predict(X_va)
    test_pred_gated = gated_predict(X_te)

    val_metrics_raw   = calculate_nilm_metrics(y_va, val_pred_raw,  threshold=threshold)
    test_metrics_raw  = calculate_nilm_metrics(y_te, test_pred_raw, threshold=threshold)
    val_metrics  = calculate_nilm_metrics(y_va, val_pred_gated,  threshold=threshold)
    test_metrics = calculate_nilm_metrics(y_te, test_pred_gated, threshold=threshold)

    print(f"  Val  [pre-gate]  -- F1={val_metrics_raw['f1']:.4f}  P={val_metrics_raw['precision']:.4f}  "
          f"R={val_metrics_raw['recall']:.4f}  MAE={val_metrics_raw['mae']:.2f}  SAE={val_metrics_raw['sae']:.4f}")
    print(f"  Val  [gated]     -- F1={val_metrics['f1']:.4f}  P={val_metrics['precision']:.4f}  "
          f"R={val_metrics['recall']:.4f}  MAE={val_metrics['mae']:.2f}  SAE={val_metrics['sae']:.4f}")
    print(f"  Test [pre-gate]  -- F1={test_metrics_raw['f1']:.4f}  P={test_metrics_raw['precision']:.4f}  "
          f"R={test_metrics_raw['recall']:.4f}  MAE={test_metrics_raw['mae']:.2f}  SAE={test_metrics_raw['sae']:.4f}  "
          f"TP={test_metrics_raw['TP']:,}  TN={test_metrics_raw['TN']:,}  "
          f"FP={test_metrics_raw['FP']:,}  FN={test_metrics_raw['FN']:,}")
    print(f"  Test [gated]     -- F1={test_metrics['f1']:.4f}  P={test_metrics['precision']:.4f}  "
          f"R={test_metrics['recall']:.4f}  MAE={test_metrics['mae']:.2f}  SAE={test_metrics['sae']:.4f}  "
          f"TP={test_metrics['TP']:,}  TN={test_metrics['TN']:,}  "
          f"FP={test_metrics['FP']:,}  FN={test_metrics['FN']:,}  "
          f"(ΔF1={test_metrics['f1']-test_metrics_raw['f1']:+.4f} "
          f"ΔP={test_metrics['precision']-test_metrics_raw['precision']:+.4f})")

    # -- Feature importances plot (both models) --
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, model, title in [(axes[0], classifier, 'gate classifier'),
                             (axes[1], regressor,  'power regressor')]:
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1]
        ax.bar(range(len(FEATURE_NAMES)), importances[order], color='steelblue')
        ax.set_xticks(range(len(FEATURE_NAMES)))
        ax.set_xticklabels([FEATURE_NAMES[i] for i in order], rotation=45, ha='right')
        ax.set_title(f'{appliance_name} -- {title}')
        ax.set_ylabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'rf_apr_new_house1_v2_{appliance_name}_feature_importance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    config = {
        'appliance': appliance_name,
        'dataset': 'APR-new-House1-dataset',
        'model': 'RandomForestRegressor + RandomForestClassifier gate',
        'version': 'v2',
        'fixes': {
            'fix1_classifier_gate': 'gated_power = classifier.predict_proba(X)[:,1] * '
                                    'regressor.predict(X); classifier uses class_weight=balanced',
            'fix2_adaptive_threshold': f'train-locked adaptive threshold ({threshold:.1f}W) '
                                       'replaces v1\'s flat 10W for all appliances',
        },
        'threshold_w': threshold,
        'window_size': WIN,
        'stride': STRIDE,
        'model_params': {
            'n_estimators': N_ESTIMATORS, 'max_depth': MAX_DEPTH,
            'min_samples_leaf': MIN_SAMPLES_LEAF,
            'regressor_sample_weight': 'none (unweighted -- see docstring)',
            'classifier_class_weight': 'balanced',
            'random_state': RANDOM_STATE,
        },
        'feature_names': FEATURE_NAMES,
        'classifier_feature_importances': {FEATURE_NAMES[i]: float(classifier.feature_importances_[i])
                                           for i in range(len(FEATURE_NAMES))},
        'regressor_feature_importances': {FEATURE_NAMES[i]: float(regressor.feature_importances_[i])
                                          for i in range(len(FEATURE_NAMES))},
        'val_metrics_pre_gate':  {k: float(v) for k, v in val_metrics_raw.items()},
        'val_metrics':           {k: float(v) for k, v in val_metrics.items()},
        'test_metrics_pre_gate': {k: float(v) for k, v in test_metrics_raw.items()},
        'test_metrics':          {k: float(v) for k, v in test_metrics.items()},
    }
    with open(os.path.join(save_dir, f'rf_apr_new_house1_v2_{appliance_name}_results.json'),
              'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return test_metrics


def main():
    data_dict = load_data()

    # FIX 2: adaptive thresholds computed on train only, reused for val/test
    thresholds = compute_adaptive_thresholds(data_dict['train'])
    print("\n  Adaptive thresholds (train-locked, replaces v1's flat 10W):")
    for app in APPLIANCES:
        print(f"    {app:<16} {thresholds[app]:.1f}W")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = os.path.join(
        os.path.dirname(__file__), '..', 'models', f"rf_apr_new_house1_dataset_v2_{timestamp}")

    all_results = {}
    for appliance_name in APPLIANCES:
        print(f"\n{'='*60}")
        print(f"Training RF gate+regressor on {appliance_name}")
        print(f"{'='*60}")
        appliance_dir = os.path.join(base_save_dir, appliance_name)
        test_metrics = train_rf_on_appliance(data_dict, appliance_name, thresholds, appliance_dir)
        all_results[appliance_name] = test_metrics

    summary = {
        'timestamp': timestamp,
        'dataset': 'APR-new-House1-dataset',
        'model': 'RandomForestRegressor + RandomForestClassifier gate',
        'version': 'v2',
        'dataset_splits': {
            'training':   {'house': 1, 'start': '2014-09-08', 'end': '2014-10-07'},
            'validation': {'house': 1, 'start': '2014-10-08', 'end': '2014-10-14'},
            'testing':    {'house': 1, 'start': '2014-10-15', 'end': '2014-10-21'},
        },
        'window_size': WIN, 'stride': STRIDE, 'thresholds_w': thresholds,
        'model_params': {
            'n_estimators': N_ESTIMATORS, 'max_depth': MAX_DEPTH,
            'min_samples_leaf': MIN_SAMPLES_LEAF,
            'regressor_sample_weight': 'none (unweighted)',
            'classifier_class_weight': 'balanced',
        },
        'results': {app: {k: float(v) for k, v in m.items()} for app, m in all_results.items()},
    }
    with open(os.path.join(base_save_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    print(f"\nRandom Forest v2 (gated) APR-new-House1-dataset testing completed. Results saved to {base_save_dir}\n")
    print(f"{'Appliance':<18} {'F1':>7} {'Prec':>7} {'Rec':>7} {'MAE':>7} {'SAE':>7} "
          f"{'TP':>7} {'TN':>7} {'FP':>7} {'FN':>7}")
    print("-" * 92)
    for app in APPLIANCES:
        m = all_results[app]
        print(f"{app:<18} {m['f1']:>7.4f} {m['precision']:>7.4f} "
              f"{m['recall']:>7.4f} {m['mae']:>7.2f} {m['sae']:>7.4f} "
              f"{m['TP']:>7,d} {m['TN']:>7,d} {m['FP']:>7,d} {m['FN']:>7,d}")


if __name__ == "__main__":
    for fname in ['UKDALE_HF_train.csv', 'UKDALE_HF_validation.csv', 'UKDALE_HF_test.csv']:
        path = os.path.join(DATASET_DIR, fname)
        if not os.path.exists(path):
            print(f"Error: {path} not found!")
            sys.exit(1)

    main()
