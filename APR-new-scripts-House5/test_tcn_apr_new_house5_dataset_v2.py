"""
test_tcn_apr_new_house5_dataset_v2.py
=======================================
House 5 counterpart to test_tcn_apr_new_house1_dataset_v2.py and
test_tcn_apr_new_house2_dataset_v2.py -- ported directly to the "v2"
gated-head architecture (there is no separate un-gated House 5 TCN script
in this repo; test_gru/lnn/lstm_apr_new_house5_dataset.py serve as this
house's un-gated MSE-only baselines).

GATE HEAD
    TCNModel's usual single unbounded nn.Linear(num_channels[-1], output_size)
    head after global-average-pooling the TCN stack, trained with MSE only,
    is replaced with two heads sharing the same dilated-TCN trunk: a gate
    head predicting P(ON) via BCEWithLogits, and a power head predicting a
    sigmoid-bounded power estimate.
        gated_power = sigmoid(gate_logit) * sigmoid(power_logit)
    Loss = MSE(gated_power, y) + lambda_event * BCEWithLogits(gate_logit, y_on).

ADAPTIVE PER-APPLIANCE THRESHOLDS (train-locked)
    Same compute_adaptive_thresholds() formula used by
    combined_pinn_lnn_apr_new_house5_dataset.py and nilm_sfra_svm_house5.py,
    computed on train only and reused for val/test (both for the gate's BCE
    target and the reported precision/recall/F1) -- verified stable across
    all three House 5 splits (the p5 > CYCLING_P5_W "clean channel" branch
    fires consistently for fridge in train/val/test alike, so there is no
    moving-goalpost risk here). Measured values on the House 5 train split:
        dishwasher=77W, fridge=10W, microwave=68W, washing_machine=34W
    These differ from the hand-picked THRESHOLDS dict in
    test_gru/lnn/lstm_apr_new_house5_dataset.py (10/50/200/20 W, taken from
    building5.yaml's on_power_threshold values) -- the same known
    threshold inconsistency across baseline vs. gated scripts already
    documented for House 1/House 2 in this repo.

POS_WEIGHT_CLAMP raised for House 5's extreme microwave imbalance
    House 5 microwave is ON only ~0.06-0.2% of samples at its adaptive
    threshold (real usage is a handful of short zaps across the whole
    recording, see DATASETS.md), giving an off:on ratio of roughly 1587:1
    at the train-locked threshold -- far more extreme than House 1's 50:1
    or House 2's 249:1. The upper clamp is raised to 1600 here so the gate's
    BCE loss isn't starved for this appliance; other appliances (dishwasher
    ~53:1, fridge ~139:1, washing_machine ~34:1) sit well under this ceiling
    and are unaffected.

Everything else -- window_size=100 (seq2point, midpoint targeting),
num_channels=[32,64,128]/kernel_size=3 TCN stack, per-appliance training
loop, Adam/ReduceLROnPlateau, early stopping on val loss, checkpoint
restore before test -- matches the House 1/House 2 v2 scripts unchanged.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Source Code'))
from models import TCNBlock
from utils import calculate_nilm_metrics, save_model

DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'APR-new-House5-dataset')
APPLIANCES  = ['dishwasher', 'fridge', 'microwave', 'washing_machine']

# FIX 2: adaptive-threshold constants (identical to combined_pinn_lnn_apr_new_house5_dataset.py)
THRESHOLD_DELTA   = 20.0
THRESHOLD_LOW_PCT = 0.05
THRESHOLD_MIN     = 10.0
CYCLING_P5_W      = 80.0
POS_WEIGHT_CLAMP  = (1.0, 1600.0)  # House 5 microwave has a ~1587:1 off:on ratio
                                    # at its adaptive threshold -- raised well beyond
                                    # House 1's (1.0, 50.0) / House 2's (1.0, 250.0);
                                    # combined_pinn_lnn_apr_new_house5_dataset.py in
                                    # this folder keeps the original unraised 50.0
LAMBDA_EVENT      = 1.0   # BCE gate-loss weight (seq2point: one prediction per
                          # sample, not averaged over a whole window like the
                          # LNN scripts, so this starts at unit weight relative
                          # to MSE rather than the LNN's 0.05).


def compute_adaptive_thresholds(df: pd.DataFrame) -> dict:
    """Identical formula to combined_pinn_lnn_apr_new_house5_dataset.py."""
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


class UKDALEDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data(dataset_dir=DATASET_DIR):
    print(f"Loading data from: {os.path.abspath(dataset_dir)}")

    def _load(filename):
        path = os.path.join(dataset_dir, filename)
        df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
        return df.rename(columns={'aggregate': 'main'})

    train_data = _load('UKDALE_HF_train.csv')
    val_data   = _load('UKDALE_HF_validation.csv')
    test_data  = _load('UKDALE_HF_test.csv')

    for name, df in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
        print(f"  {name}: {df.index.min()} to {df.index.max()}  ({len(df):,} rows)")

    return {'train': train_data, 'val': val_data, 'test': test_data, 'appliances': APPLIANCES}


def create_sequences(data, appliance, window_size=100, stride=5):
    """Midpoint targeting: y[i] is the appliance value at the window centre. (Unchanged from v1.)"""
    mains    = data['main'].values
    app_vals = data[appliance].values
    X, y = [], []
    for i in range(0, len(mains) - window_size, stride):
        X.append(mains[i:i + window_size])
        mid = i + window_size // 2
        y.append(app_vals[mid])
    X = np.array(X, dtype=np.float32).reshape(-1, window_size, 1)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    return X, y


# ---------------------------------------------------------------------------
# FIX 1: gated TCN -- same dilated-TCN trunk as v1's TCNModel, but with a
# second (gate) head. gated_power = sigmoid(gate) * sigmoid(power).
# ---------------------------------------------------------------------------

class GatedTCNModel(nn.Module):
    """
    Same dilated-TCN trunk as models.TCNModel (a stack of TCNBlock layers
    with exponentially increasing dilation, then global-average-pool over
    the sequence dim). Output head is replaced with two heads:

        gate_logits  -- BCEWithLogits target, detection confidence
        power_logits -- sigmoid-bounded power estimate in [0,1]
        gated_power  = sigmoid(gate_logits) * sigmoid(power_logits)

    Identical design to test_tcn_apr_new_house1_dataset_v2.py's
    GatedTCNModel, adapted to House 5's pos_weight clamp.
    """

    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation     = 2 ** i
            in_channels  = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation, dropout))
        self.tcn_layers = nn.Sequential(*layers)

        self.gate_head  = nn.Linear(num_channels[-1], output_size)
        self.power_head = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = x.permute(0, 2, 1)       # -> (batch, input_size, seq_len)
        x = self.tcn_layers(x)
        x = torch.mean(x, dim=2)     # global average pool over the sequence

        gate_logits = self.gate_head(x)
        power       = torch.sigmoid(self.power_head(x))
        gated_power = torch.sigmoid(gate_logits) * power
        return gated_power, gate_logits


def train_tcn_on_appliance(data_dict, appliance_name, thresholds, window_size=100,
                            num_channels=None, kernel_size=3, dropout=0.2,
                            epochs=80, lr=0.001, patience=20, lambda_event=LAMBDA_EVENT,
                            save_dir='models/tcn_apr_new_house5_dataset_v2'):
    if num_channels is None:
        num_channels = [32, 64, 128]

    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_data = data_dict['train']
    val_data   = data_dict['val']
    test_data  = data_dict['test']
    threshold  = thresholds[appliance_name]  # FIX 2: train-locked adaptive threshold

    print(f"Creating sequences for {appliance_name}...  (threshold={threshold:.1f}W)")
    X_train, y_train = create_sequences(train_data, appliance_name, window_size)
    X_val,   y_val   = create_sequences(val_data,   appliance_name, window_size)
    X_test,  y_test  = create_sequences(test_data,  appliance_name, window_size)

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train = x_scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    X_val   = x_scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
    X_test  = x_scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

    y_train = y_scaler.fit_transform(y_train)
    y_val   = y_scaler.transform(y_val)
    y_test  = y_scaler.transform(y_test)

    # FIX 1: scaled threshold + pos_weight for the gate's BCE target, same
    # formula as GatedEventLoss in combined_pinn_lnn_apr_new_house5_dataset.py
    thr_scaled = (threshold - float(y_scaler.data_min_[0])) / float(y_scaler.data_range_[0])
    n_on  = float((y_train > thr_scaled).sum())
    n_off = float((y_train <= thr_scaled).sum())
    pos_weight = float(np.clip(n_off / max(n_on, 1.0), *POS_WEIGHT_CLAMP))
    print(f"  Gate target: on={100*n_on/(n_on+n_off):.1f}%  pos_weight={pos_weight:.1f}  "
          f"thr_scaled={thr_scaled:.4f}")

    print(f"Training sequences:   {X_train.shape} -> {y_train.shape}")
    print(f"Validation sequences: {X_val.shape} -> {y_val.shape}")
    print(f"Test sequences:       {X_test.shape} -> {y_test.shape}")

    train_loader = torch.utils.data.DataLoader(
        UKDALEDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        UKDALEDataset(X_val, y_val), batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        UKDALEDataset(X_test, y_test), batch_size=32, shuffle=False)

    input_size  = 1
    output_size = 1

    model = GatedTCNModel(
        input_size=input_size,
        output_size=output_size,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout
    ).to(device)

    mse_crit   = nn.MSELoss()
    thr_t      = torch.tensor(thr_scaled, dtype=torch.float32, device=device)
    pw_t       = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    history = {'train_loss': [], 'train_mse': [], 'train_bce': [],
               'val_loss': [], 'val_metrics': []}
    best_val_loss = float('inf')
    best_state    = None
    counter = 0

    print(f"Starting Gated TCN training for {appliance_name}...")

    for epoch in range(epochs):
        model.train()
        train_loss = train_mse = train_bce = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            gated_power, gate_logits = model(inputs)
            y_on   = (targets > thr_t).float()
            l_mse  = mse_crit(gated_power, targets)
            l_bce  = F.binary_cross_entropy_with_logits(gate_logits, y_on, pos_weight=pw_t)
            loss   = l_mse + lambda_event * l_bce
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item(); train_mse += l_mse.item(); train_bce += l_bce.item()
            progress_bar.set_postfix({'loss': loss.item(), 'mse': l_mse.item(), 'bce': l_bce.item()})

        nb = len(train_loader)
        avg_train_loss = train_loss / nb
        history['train_loss'].append(avg_train_loss)
        history['train_mse'].append(train_mse / nb)
        history['train_bce'].append(train_bce / nb)

        model.eval()
        val_loss = 0.0
        all_targets, all_outputs = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                gated_power, gate_logits = model(inputs)
                y_on  = (targets > thr_t).float()
                l_mse = mse_crit(gated_power, targets)
                l_bce = F.binary_cross_entropy_with_logits(gate_logits, y_on, pos_weight=pw_t)
                val_loss += (l_mse + lambda_event * l_bce).item()
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(gated_power.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        scheduler.step(avg_val_loss)

        all_targets = y_scaler.inverse_transform(
            np.concatenate(all_targets).reshape(-1, 1)).flatten()
        all_outputs = y_scaler.inverse_transform(
            np.concatenate(all_outputs).reshape(-1, 1)).flatten()
        metrics = calculate_nilm_metrics(all_targets, all_outputs, threshold=threshold)
        history['val_metrics'].append(metrics)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f} "
              f"(mse={history['train_mse'][-1]:.6f} bce={history['train_bce'][-1]:.6f}), "
              f"Val Loss: {avg_val_loss:.6f}, Val MAE: {metrics['mae']:.2f}, "
              f"Val SAE: {metrics['sae']:.2f}, Val F1: {metrics['f1']:.4f}, "
              f"Val Precision: {metrics['precision']:.4f}, Val Recall: {metrics['recall']:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            counter = 0
            best_model_path = os.path.join(
                save_dir, f"tcn_apr_new_house5_v2_{appliance_name}_best.pth")
            save_model(model,
                       {'input_size': input_size, 'output_size': output_size,
                        'num_channels': num_channels, 'kernel_size': kernel_size,
                        'dropout': dropout},
                       {'lr': lr, 'epochs': epochs, 'patience': patience,
                        'window_size': window_size, 'appliance': appliance_name,
                        'threshold': threshold, 'pos_weight': pos_weight},
                       metrics, best_model_path)
            print(f"Model saved to {best_model_path}")
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {patience}")
            if counter >= patience:
                print("Early stopping triggered")
                break

    print("Training completed!")

    if best_state is not None:
        model.load_state_dict(best_state)

    print("Evaluating on test set...")
    model.eval()
    test_loss = 0.0
    all_test_targets, all_test_outputs = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            gated_power, gate_logits = model(inputs)
            y_on  = (targets > thr_t).float()
            l_mse = mse_crit(gated_power, targets)
            l_bce = F.binary_cross_entropy_with_logits(gate_logits, y_on, pos_weight=pw_t)
            test_loss += (l_mse + lambda_event * l_bce).item()
            all_test_targets.append(targets.cpu().numpy())
            all_test_outputs.append(gated_power.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    all_test_targets = y_scaler.inverse_transform(
        np.concatenate(all_test_targets).reshape(-1, 1)).flatten()
    all_test_outputs = y_scaler.inverse_transform(
        np.concatenate(all_test_outputs).reshape(-1, 1)).flatten()
    test_metrics = calculate_nilm_metrics(all_test_targets, all_test_outputs, threshold=threshold)

    val_mae_series       = [m['mae']       for m in history['val_metrics']]
    val_sae_series       = [m['sae']       for m in history['val_metrics']]
    val_f1_series        = [m['f1']        for m in history['val_metrics']]
    val_precision_series = [m['precision'] for m in history['val_metrics']]
    val_recall_series    = [m['recall']    for m in history['val_metrics']]

    aggregates = {
        'train_loss_mean':    float(np.mean(history['train_loss'])),
        'train_loss_var':     float(np.var(history['train_loss'])),
        'val_loss_mean':      float(np.mean(history['val_loss'])),
        'val_loss_var':       float(np.var(history['val_loss'])),
        'val_mae_mean':       float(np.mean(val_mae_series)),
        'val_mae_var':        float(np.var(val_mae_series)),
        'val_sae_mean':       float(np.mean(val_sae_series)),
        'val_sae_var':        float(np.var(val_sae_series)),
        'val_f1_mean':        float(np.mean(val_f1_series)),
        'val_f1_var':         float(np.var(val_f1_series)),
        'val_precision_mean': float(np.mean(val_precision_series)),
        'val_precision_var':  float(np.var(val_precision_series)),
        'val_recall_mean':    float(np.mean(val_recall_series)),
        'val_recall_var':     float(np.var(val_recall_series)),
        'test_mae':           float(test_metrics['mae']),
        'test_sae':           float(test_metrics['sae']),
        'test_f1':            float(test_metrics['f1']),
        'test_precision':     float(test_metrics['precision']),
        'test_recall':        float(test_metrics['recall']),
        'test_loss':          float(avg_test_loss),
    }

    print(f"Test Loss: {avg_test_loss:.6f}")
    print(f"Test Metrics: {test_metrics}")
    print("Aggregates (mean/variance):")
    print(json.dumps(aggregates, indent=2))

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'],   label='Val Loss',   color='red')
    plt.title(f'Loss - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('MSE + BCE Loss')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(val_mae_series, label='Val MAE', color='red')
    plt.axhline(test_metrics['mae'], label='Test MAE', color='green', linestyle='--')
    plt.title(f'MAE - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('MAE (W)')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(val_sae_series, label='Val SAE', color='red')
    plt.axhline(test_metrics['sae'], label='Test SAE', color='green', linestyle='--')
    plt.title(f'SAE - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('SAE')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(val_f1_series, label='Val F1', color='red')
    plt.axhline(test_metrics['f1'], label='Test F1', color='green', linestyle='--')
    plt.title(f'F1 Score - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('F1')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"tcn_apr_new_house5_v2_{appliance_name}_metrics.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    config = {
        'appliance': appliance_name,
        'dataset': 'APR-new-House5-dataset',
        'model': 'GatedTCNModel',
        'version': 'v2',
        'fixes': {
            'fix1_gate_head': 'sigmoid(gate_logit) * sigmoid(power_logit), '
                              'BCEWithLogits(gate_logit, y_on, pos_weight) added to the loss',
            'fix2_adaptive_threshold': f'train-locked adaptive threshold ({threshold:.1f}W) '
                                       'replaces v1\'s hand-picked threshold for this appliance',
        },
        'window_size': window_size,
        'threshold_w': threshold,
        'pos_weight': pos_weight,
        'lambda_event': lambda_event,
        'model_params': {
            'input_size': input_size, 'output_size': output_size,
            'num_channels': num_channels, 'kernel_size': kernel_size, 'dropout': dropout
        },
        'train_params': {'lr': lr, 'epochs': epochs, 'patience': patience},
        'final_metrics': {
            'train_loss': history['train_loss'][-1] if history['train_loss'] else None,
            'val_loss':   history['val_loss'][-1]   if history['val_loss']   else None,
            'test_loss':  avg_test_loss,
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'aggregates': aggregates
        }
    }
    with open(os.path.join(save_dir, f'tcn_apr_new_house5_v2_{appliance_name}_history.json'),
              'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return model, history, test_metrics


def test_tcn_on_all_appliances(window_size=100, num_channels=None, kernel_size=3,
                                dropout=0.2, epochs=80, lr=0.001, patience=20,
                                lambda_event=LAMBDA_EVENT):
    if num_channels is None:
        num_channels = [32, 64, 128]

    data_dict = load_data()

    # FIX 2: adaptive thresholds computed on train only, reused for val/test
    thresholds = compute_adaptive_thresholds(data_dict['train'])
    print("\n  Adaptive thresholds (train-locked, replaces v1's hand-picked 10/20/30/10W):")
    for app in APPLIANCES:
        print(f"    {app:<16} {thresholds[app]:.1f}W")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = os.path.join(
        os.path.dirname(__file__), '..', 'models', f"tcn_apr_new_house5_dataset_v2_{timestamp}")

    all_results = {}

    for appliance_name in APPLIANCES:
        print(f"\n{'='*60}")
        print(f"Testing Gated TCN on {appliance_name}")
        print(f"{'='*60}\n")

        appliance_dir = os.path.join(base_save_dir, appliance_name)
        os.makedirs(appliance_dir, exist_ok=True)

        try:
            model, history, test_metrics = train_tcn_on_appliance(
                data_dict, appliance_name=appliance_name, thresholds=thresholds,
                window_size=window_size, num_channels=num_channels,
                kernel_size=kernel_size, dropout=dropout,
                epochs=epochs, lr=lr, patience=patience, lambda_event=lambda_event,
                save_dir=appliance_dir)
            if model is not None:
                all_results[appliance_name] = {
                    'model_path': os.path.join(
                        appliance_dir, f"tcn_apr_new_house5_v2_{appliance_name}_best.pth"),
                    'final_metrics': {k: float(v) for k, v in test_metrics.items()}
                }
                print(f"Successfully tested Gated TCN on {appliance_name}")
        except Exception as e:
            print(f"Error on {appliance_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    summary = {
        'timestamp': timestamp,
        'dataset': 'APR-new-House5-dataset',
        'model': 'GatedTCNModel',
        'version': 'v2',
        'dataset_splits': {
            'training':   {'house': 5, 'start': '2014-08-04', 'end': '2014-09-02'},
            'validation': {'house': 5, 'start': '2014-09-03', 'end': '2014-09-09'},
            'testing':    {'house': 5, 'start': '2014-09-10', 'end': '2014-09-16'}
        },
        'window_size': window_size,
        'thresholds_w': thresholds,
        'model_params': {'num_channels': num_channels, 'kernel_size': kernel_size,
                          'dropout': dropout},
        'train_params': {'epochs': epochs, 'lr': lr, 'patience': patience,
                         'lambda_event': lambda_event},
        'results': all_results
    }

    with open(os.path.join(base_save_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    print(f"\nGated TCN v2 APR-new-House5-dataset testing completed. Results saved to {base_save_dir}")
    print(f"{'Appliance':<18} {'F1':>8} {'Precision':>10} {'Recall':>8} {'MAE':>8} {'SAE':>8}")
    print("-" * 68)
    for app in APPLIANCES:
        if app in all_results:
            m = all_results[app]['final_metrics']
            print(f"{app:<18} {m['f1']:>8.4f} {m['precision']:>10.4f} "
                  f"{m['recall']:>8.4f} {m['mae']:>8.2f} {m['sae']:>8.4f}")

    return all_results


if __name__ == "__main__":
    print("Testing Gated TCN v2 on APR-new-House5-dataset...")

    for fname in ['UKDALE_HF_train.csv', 'UKDALE_HF_validation.csv', 'UKDALE_HF_test.csv']:
        path = os.path.join(DATASET_DIR, fname)
        if not os.path.exists(path):
            print(f"Error: {path} not found!")
            sys.exit(1)

    results = test_tcn_on_all_appliances(
        window_size=100, num_channels=[32, 64, 128], kernel_size=3,
        dropout=0.2, epochs=80, lr=0.001, patience=20, lambda_event=LAMBDA_EVENT)

    print(f"\nSummary of Gated TCN v2 testing on APR-new-House5-dataset:")
    print(f"Total appliances tested: {len(results)}")
    for appliance, result in results.items():
        print(f"  {appliance}:")
        print(f"    Test MAE:       {result['final_metrics']['mae']:.4f}")
        print(f"    Test SAE:       {result['final_metrics']['sae']:.4f}")
        print(f"    Test F1:        {result['final_metrics']['f1']:.4f}")
        print(f"    Test Precision: {result['final_metrics']['precision']:.4f}")
        print(f"    Test Recall:    {result['final_metrics']['recall']:.4f}")
