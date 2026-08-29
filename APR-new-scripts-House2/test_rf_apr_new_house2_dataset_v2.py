"""
test_rf_apr_new_house2_dataset_v2.py
======================================
v2 of test_rf_apr_new_house2_dataset.py -- House 2 counterpart to
test_rf_apr_new_house1_dataset_v2.py, with the same two fixes.

v1 test results:

    Appliance            F1  Precision   Recall     MAE     SAE
    dishwasher       0.5484     0.4041   0.8528    4.15   3.0837
    fridge           0.9054     0.8271   1.0000   17.69  10.5238
    microwave        0.1932     0.1158   0.5818    7.54   3.7160
    washing_machine  0.8397     0.7550   0.9457    8.34   4.5319

fridge's TN=0 in the raw confusion counts is the most striking single
number in that run -- at the flat 10W threshold, essentially every sample
reads "ON" (House 2's fridge idle floor is ~9-11W, so 10W barely clears
noise, not actual compressor cycling). microwave is the worst performer by
F1 (P=0.116), the same rare-class precision collapse pattern already fixed
for House 1's RF and every neural baseline in both houses.

FIX 1 -- Two-stage classifier gate (main fix)
    Same fix as test_rf_apr_new_house1_dataset_v2.py: v1's docstring
    documents that inverse-frequency sample_weight on the regressor was
    tried and REJECTED (made every rare appliance worse) -- the regression
    analogue of the same imbalance-reweighting failure mode found
    throughout this project (SGDClassifier's class_weight='balanced'
    craterred precision in nilm_sfra_svm.py; upweighting a squared-error
    loss inflates a small positive bias across nearly all OFF windows).
    Fix: don't reweight the regressor -- add a second model,
    RandomForestClassifier(class_weight='balanced'), trained purely as an
    ON/OFF detector, and gate the regressor's output with it:
        gated_power = classifier.predict_proba(X)[:, 1] * regressor.predict(X)
    class_weight='balanced' is safe on a classifier (tree splits chosen by
    weighted Gini purity per node) in a way sample_weight was not safe on
    the regressor (global squared-error surface skewed by upweighted
    residuals). Pre-gate and post-gate metrics are both printed so the
    effect is directly visible, not just assumed.

FIX 2 -- Adaptive per-appliance thresholds (train-locked)
    v1 used a flat 10W threshold for all four appliances -- identical bug
    to House 1's RF script, and the reason fridge's TN=0 above. House 2's
    own adaptive thresholds (same formula used throughout this repo, e.g.
    combined_pinn_lnn_apr_new_house2_dataset_v2.py) are 21W / 30W / 45W /
    21W for dishwasher / fridge / microwave / washing_machine. Fix:
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

DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'APR-new-House2-dataset')
APPLIANCES  = ['dishwasher', 'fridge', 'microwave', 'washing_machine']

# FIX 2: adaptive-threshold constants (identical to combined_pinn_lnn_apr_new_house2_dataset_v2.py)
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
    """Identical formula to combined_pinn_lnn_apr_new_house2_dataset_v2.py."""
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
    plt.savefig(os.path.join(save_dir, f'rf_apr_new_house2_v2_{appliance_name}_feature_importance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    config = {
        'appliance': appliance_name,
        'dataset': 'APR-new-House2-dataset',
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
    with open(os.path.join(save_dir, f'rf_apr_new_house2_v2_{appliance_name}_results.json'),
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
        os.path.dirname(__file__), '..', 'models', f"rf_apr_new_house2_dataset_v2_{timestamp}")

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
        'dataset': 'APR-new-House2-dataset',
        'model': 'RandomForestRegressor + RandomForestClassifier gate',
        'version': 'v2',
        'dataset_splits': {
            'training':   {'house': 2, 'start': '2013-06-15', 'end': '2013-07-14'},
            'validation': {'house': 2, 'start': '2013-07-15', 'end': '2013-07-21'},
            'testing':    {'house': 2, 'start': '2013-07-22', 'end': '2013-07-28'},
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

    print(f"\nRandom Forest v2 (gated) APR-new-House2-dataset testing completed. Results saved to {base_save_dir}\n")
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
