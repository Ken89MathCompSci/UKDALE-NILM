"""
Random Forest baseline for NILM -- APR-new-House1-dataset/ splits.

Derived in spirit from fine_tuning_dataset_scripts/old-trf-scripts/ (exp_2.py,
WeightedRF.py, feature_extract.py, ser.py), which use a RandomForestClassifier
for appliance classification with cross-domain transfer learning (SER --
Structural Expansion/Reduction -- adapts a forest trained on one house/dataset
to another using a handful of "tuning" samples from the target domain).

That transfer-learning machinery does NOT apply here: APR-new-House1-dataset/
is a single house split chronologically (train/val/test), not a cross-house
or cross-dataset domain-shift scenario, so there's no second domain to adapt
into. feature_extract.py's power-factor/harmonic/THD features also can't be
reused -- they require raw voltage and current waveforms at ~30kHz, and
UKDALE only records active power at 6s resolution (no V/I waveforms).

What IS carried over is the core idea: Random Forest + engineered statistical
features, as a classical-ML baseline to compare against the LNN/GRU/LSTM/TCN/
ResNet/Transformer family in this folder. Concretely:

  - One RandomForestRegressor per appliance predicting continuous power (W)
    at the window midpoint -- regression, not classification, so it plugs
    into the same calculate_nilm_metrics() evaluation used by every other
    script in this folder and gets MAE/SAE/precision/recall/F1 all from one
    model, instead of only a class label with no Watts to measure error
    against. Same "one model per appliance" pattern as
    test_gru_apr_new_house1_dataset.py etc.
  - Features are 10 windowed statistics of the aggregate signal (mean, std,
    min, max, range, median, first, last, mean|diff|, max|diff|) computed
    over the same WIN=100/STRIDE=5 midpoint-targeted windows used by the
    other baseline scripts in this folder -- kept simple since RF doesn't
    need per-timestep sequence input the way the neural models do.
  - No sample_weight/class_weight imbalance correction (dishwasher/microwave
    ON only ~1.4-2.9% of the time). Tried an inverse-frequency sample_weight
    on the regressor (the regression analogue of class_weight='balanced')
    and it backfired: F1 dropped for every rare appliance (dishwasher
    0.31->0.22, microwave 0.31->0.20) because upweighting ON samples in a
    squared-error loss inflates a small positive bias across nearly all OFF
    windows, which is enough to clear the 10 W threshold on many of them --
    the same recall~1.0/precision-collapse failure mode seen in the PINN-LNN
    experiments earlier in this project. Left unweighted, which does better
    on every appliance. This is different from WeightedRF/SER anyway (those
    are for cross-domain adaptation, not class imbalance).
  - No feature/target scaling needed -- tree splits are scale-invariant, so
    RF operates directly on raw Watts (unlike the neural models, which need
    MinMaxScaler/StandardScaler and can suffer from outlier-spike compression).
  - Feature importances are plotted per appliance, a diagnostic specific to
    RF that isn't available for the neural models.
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
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Source Code'))
from utils import calculate_nilm_metrics

DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'APR-new-House1-dataset')
APPLIANCES  = ['dishwasher', 'fridge', 'microwave', 'washing_machine']
THRESHOLD_W = 10.0

WIN    = 100
STRIDE = 5

N_ESTIMATORS    = 300
MAX_DEPTH       = 20
MIN_SAMPLES_LEAF = 5
RANDOM_STATE    = 42

FEATURE_NAMES = ['mean', 'std', 'min', 'max', 'range', 'median',
                  'first', 'last', 'mean_abs_diff', 'max_abs_diff']


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
    this folder): X[i] summarizes mains[i*stride : i*stride+win], and
    y[i] is the appliance's continuous power (W) at that window's midpoint.
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


def train_rf_on_appliance(data_dict, appliance_name, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    train_data = data_dict['train']
    val_data   = data_dict['val']
    test_data  = data_dict['test']

    print(f"\nBuilding windowed features for {appliance_name}...")
    X_tr, y_tr = build_window_features(train_data['aggregate'].values, train_data[appliance_name].values)
    X_va, y_va = build_window_features(val_data['aggregate'].values,   val_data[appliance_name].values)
    X_te, y_te = build_window_features(test_data['aggregate'].values,  test_data[appliance_name].values)

    on_tr = (y_tr > THRESHOLD_W).mean() * 100
    on_va = (y_va > THRESHOLD_W).mean() * 100
    on_te = (y_te > THRESHOLD_W).mean() * 100
    print(f"  Train: {X_tr.shape}  ON={on_tr:.2f}%")
    print(f"  Val:   {X_va.shape}  ON={on_va:.2f}%")
    print(f"  Test:  {X_te.shape}  ON={on_te:.2f}%")

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    print(f"Fitting RandomForestRegressor ({N_ESTIMATORS} trees) for {appliance_name}...")
    model.fit(X_tr, y_tr)

    val_pred  = model.predict(X_va)
    test_pred = model.predict(X_te)
    val_metrics  = calculate_nilm_metrics(y_va, val_pred,  threshold=THRESHOLD_W)
    test_metrics = calculate_nilm_metrics(y_te, test_pred, threshold=THRESHOLD_W)

    print(f"  Val  -- F1={val_metrics['f1']:.4f}  P={val_metrics['precision']:.4f}  "
          f"R={val_metrics['recall']:.4f}  MAE={val_metrics['mae']:.2f}  SAE={val_metrics['sae']:.4f}")
    print(f"  Test -- F1={test_metrics['f1']:.4f}  P={test_metrics['precision']:.4f}  "
          f"R={test_metrics['recall']:.4f}  MAE={test_metrics['mae']:.2f}  SAE={test_metrics['sae']:.4f}  "
          f"TP={test_metrics['TP']:,}  TN={test_metrics['TN']:,}  "
          f"FP={test_metrics['FP']:,}  FN={test_metrics['FN']:,}")

    # -- Feature importances plot --
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(7, 4))
    plt.bar(range(len(FEATURE_NAMES)), importances[order], color='steelblue')
    plt.xticks(range(len(FEATURE_NAMES)), [FEATURE_NAMES[i] for i in order], rotation=45, ha='right')
    plt.title(f'{appliance_name} -- RF feature importances')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'rf_apr_new_house1_{appliance_name}_feature_importance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    config = {
        'appliance': appliance_name,
        'dataset': 'APR-new-House1-dataset',
        'model': 'RandomForestRegressor',
        'threshold_w': THRESHOLD_W,
        'window_size': WIN,
        'stride': STRIDE,
        'model_params': {
            'n_estimators': N_ESTIMATORS, 'max_depth': MAX_DEPTH,
            'min_samples_leaf': MIN_SAMPLES_LEAF, 'sample_weight': 'none (unweighted -- see docstring)',
            'random_state': RANDOM_STATE,
        },
        'feature_names': FEATURE_NAMES,
        'feature_importances': {FEATURE_NAMES[i]: float(importances[i]) for i in range(len(FEATURE_NAMES))},
        'val_metrics': {k: float(v) for k, v in val_metrics.items()},
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
    }
    with open(os.path.join(save_dir, f'rf_apr_new_house1_{appliance_name}_results.json'),
              'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return test_metrics


def main():
    data_dict = load_data()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = os.path.join(
        os.path.dirname(__file__), '..', 'models', f"rf_apr_new_house1_dataset_{timestamp}")

    all_results = {}
    for appliance_name in APPLIANCES:
        print(f"\n{'='*60}")
        print(f"Training RandomForestRegressor on {appliance_name}")
        print(f"{'='*60}")
        appliance_dir = os.path.join(base_save_dir, appliance_name)
        test_metrics = train_rf_on_appliance(data_dict, appliance_name, appliance_dir)
        all_results[appliance_name] = test_metrics

    summary = {
        'timestamp': timestamp,
        'dataset': 'APR-new-House1-dataset',
        'model': 'RandomForestRegressor',
        'dataset_splits': {
            'training':   {'house': 1, 'start': '2014-09-08', 'end': '2014-10-07'},
            'validation': {'house': 1, 'start': '2014-10-08', 'end': '2014-10-14'},
            'testing':    {'house': 1, 'start': '2014-10-15', 'end': '2014-10-21'},
        },
        'window_size': WIN, 'stride': STRIDE, 'threshold_w': THRESHOLD_W,
        'model_params': {
            'n_estimators': N_ESTIMATORS, 'max_depth': MAX_DEPTH,
            'min_samples_leaf': MIN_SAMPLES_LEAF, 'sample_weight': 'none (unweighted -- see docstring)',
        },
        'results': {app: {k: float(v) for k, v in m.items()} for app, m in all_results.items()},
    }
    with open(os.path.join(base_save_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    print(f"\nRandom Forest APR-new-House1-dataset testing completed. Results saved to {base_save_dir}\n")
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
            import sys
            sys.exit(1)

    main()
