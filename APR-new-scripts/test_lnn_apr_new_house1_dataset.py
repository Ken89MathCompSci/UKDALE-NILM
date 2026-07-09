"""
Liquid Neural Network (LNN) baseline for NILM -- APR-new-House1-dataset/ splits.

Derived from Old-LNN_Algorithms/test_lnn_ukdale_specific_splits.py with the same
three changes described in test_gru_apr_new_house1_dataset.py:
  1. Reads APR-new-House1-dataset/ CSVs instead of pickled UKDALE splits
     (House 1, train 2014-09-08..10-07 / val 10-08..10-14 / test 10-15..10-21).
  2. Fixed the X/y midpoint-alignment bug from the old script.
  3. Uniform 10 W threshold for all four appliances.

Architecture/training loop otherwise unchanged: one LiquidNetworkModel
(fixed dt) per appliance, MSE loss, Adam, ReduceLROnPlateau, early stopping,
best-checkpoint restore before test evaluation (the old LNN script already
did this correctly -- kept as-is).
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Source Code'))
from models import LiquidNetworkModel
from utils import calculate_nilm_metrics, save_model

DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'APR-new-House1-dataset')
APPLIANCES  = ['dishwasher', 'fridge', 'microwave', 'washing_machine']
THRESHOLDS  = {app: 10.0 for app in APPLIANCES}


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
    """Midpoint targeting: y[i] is the appliance value at the window centre."""
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


def train_on_appliance(data_dict, appliance_name, window_size=100,
                        hidden_size=64, dt=0.1,
                        epochs=80, lr=0.001, patience=20,
                        save_dir='models/lnn_apr_new_house1_dataset'):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_data = data_dict['train']
    val_data   = data_dict['val']
    test_data  = data_dict['test']

    print(f"Creating sequences for {appliance_name}...")
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

    print(f"Training sequences:   {X_train.shape} -> {y_train.shape}")
    print(f"Validation sequences: {X_val.shape} -> {y_val.shape}")
    print(f"Test sequences:       {X_test.shape} -> {y_test.shape}")

    train_loader = torch.utils.data.DataLoader(
        UKDALEDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        UKDALEDataset(X_val, y_val), batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        UKDALEDataset(X_test, y_test), batch_size=32, shuffle=False)

    model = LiquidNetworkModel(
        input_size=1,
        hidden_size=hidden_size,
        output_size=1,
        dt=dt
    ).to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    history = {'train_loss': [], 'val_loss': [], 'val_metrics': []}
    best_val_loss = float('inf')
    best_state    = None
    counter = 0

    print(f"Starting LiquidNetworkModel training for {appliance_name}...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        all_targets, all_outputs = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        scheduler.step(avg_val_loss)

        threshold = THRESHOLDS[appliance_name]
        raw_tgts = y_scaler.inverse_transform(
            np.concatenate(all_targets).reshape(-1, 1)).flatten()
        raw_outs = y_scaler.inverse_transform(
            np.concatenate(all_outputs).reshape(-1, 1)).flatten()
        metrics = calculate_nilm_metrics(raw_tgts, raw_outs, threshold=threshold)
        history['val_metrics'].append(metrics)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, "
              f"Val Loss: {avg_val_loss:.6f}, Val MAE: {metrics['mae']:.2f}, "
              f"Val SAE: {metrics['sae']:.2f}, Val F1: {metrics['f1']:.4f}, "
              f"Val Precision: {metrics['precision']:.4f}, Val Recall: {metrics['recall']:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            counter = 0
            best_model_path = os.path.join(
                save_dir, f"lnn_apr_new_house1_{appliance_name}_best.pth")
            save_model(model,
                       {'input_size': 1, 'output_size': 1,
                        'hidden_size': hidden_size, 'dt': dt},
                       {'lr': lr, 'epochs': epochs, 'patience': patience,
                        'window_size': window_size, 'appliance': appliance_name},
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
    all_test_targets, all_test_outputs = [], []
    test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, targets).item()
            all_test_targets.append(targets.cpu().numpy())
            all_test_outputs.append(outputs.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    threshold = THRESHOLDS[appliance_name]
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
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
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
    plt.plot(val_f1_series,        label='Val F1',        color='red')
    plt.plot(val_precision_series, label='Val Precision', color='blue')
    plt.plot(val_recall_series,    label='Val Recall',    color='orange')
    plt.axhline(test_metrics['f1'],        color='red',    linestyle='--', alpha=0.5)
    plt.axhline(test_metrics['precision'], color='blue',   linestyle='--', alpha=0.5)
    plt.axhline(test_metrics['recall'],    color='orange', linestyle='--', alpha=0.5)
    plt.title(f'F1 / Precision / Recall - {appliance_name}')
    plt.xlabel('Epoch'); plt.ylabel('Score')
    plt.legend(); plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"lnn_apr_new_house1_{appliance_name}_metrics.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    config = {
        'appliance': appliance_name,
        'dataset': 'APR-new-House1-dataset',
        'model': 'LiquidNetworkModel',
        'norm': 'none',
        'loss': 'MSE',
        'window_size': window_size,
        'model_params': {
            'input_size': 1, 'output_size': 1,
            'hidden_size': hidden_size, 'dt': dt
        },
        'train_params': {'lr': lr, 'epochs': epochs, 'patience': patience},
        'final_metrics': {
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'aggregates': aggregates
        }
    }
    with open(os.path.join(save_dir, f'lnn_apr_new_house1_{appliance_name}_history.json'),
              'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)

    return model, history, test_metrics


def test_on_all_appliances(window_size=100, hidden_size=64, dt=0.1,
                            epochs=80, lr=0.001, patience=20):
    data_dict = load_data()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = os.path.join(
        os.path.dirname(__file__), '..', 'models', f"lnn_apr_new_house1_dataset_{timestamp}")

    all_results = {}

    for appliance_name in APPLIANCES:
        print(f"\n{'='*60}")
        print(f"Testing LiquidNetworkModel on {appliance_name}")
        print(f"{'='*60}\n")

        appliance_dir = os.path.join(base_save_dir, appliance_name)
        os.makedirs(appliance_dir, exist_ok=True)

        try:
            model, history, test_metrics = train_on_appliance(
                data_dict, appliance_name=appliance_name,
                window_size=window_size, hidden_size=hidden_size, dt=dt,
                epochs=epochs, lr=lr, patience=patience, save_dir=appliance_dir)
            if model is not None:
                all_results[appliance_name] = {
                    'model_path': os.path.join(
                        appliance_dir, f"lnn_apr_new_house1_{appliance_name}_best.pth"),
                    'final_metrics': {k: float(v) for k, v in test_metrics.items()}
                }
        except Exception as e:
            print(f"Error on {appliance_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    summary = {
        'timestamp': timestamp,
        'dataset': 'APR-new-House1-dataset',
        'model': 'LiquidNetworkModel',
        'dataset_splits': {
            'training':   {'house': 1, 'start': '2014-09-08', 'end': '2014-10-07'},
            'validation': {'house': 1, 'start': '2014-10-08', 'end': '2014-10-14'},
            'testing':    {'house': 1, 'start': '2014-10-15', 'end': '2014-10-21'}
        },
        'window_size': window_size,
        'model_params': {'hidden_size': hidden_size, 'dt': dt},
        'train_params': {'epochs': epochs, 'lr': lr, 'patience': patience},
        'results': all_results
    }

    with open(os.path.join(base_save_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    print(f"\nLiquidNetworkModel APR-new-House1-dataset testing completed. "
          f"Results saved to {base_save_dir}\n")
    print(f"{'Appliance':<18} {'F1':>8} {'Precision':>10} {'Recall':>8} {'MAE':>8} {'SAE':>8}")
    print("-" * 68)
    for app in APPLIANCES:
        if app in all_results:
            m = all_results[app]['final_metrics']
            print(f"{app:<18} {m['f1']:>8.4f} {m['precision']:>10.4f} "
                  f"{m['recall']:>8.4f} {m['mae']:>8.2f} {m['sae']:>8.4f}")

    return all_results


if __name__ == "__main__":
    print("Testing LiquidNetworkModel on APR-new-House1-dataset...")

    for fname in ['UKDALE_HF_train.csv', 'UKDALE_HF_validation.csv', 'UKDALE_HF_test.csv']:
        path = os.path.join(DATASET_DIR, fname)
        if not os.path.exists(path):
            print(f"Error: {path} not found!")
            sys.exit(1)

    results = test_on_all_appliances(
        window_size=100, hidden_size=64, dt=0.1,
        epochs=80, lr=0.001, patience=20)

    print(f"\nSummary of LiquidNetworkModel testing on APR-new-House1-dataset:")
    print(f"Total appliances tested: {len(results)}")
    for appliance, result in results.items():
        print(f"  {appliance}:")
        print(f"    Test MAE:       {result['final_metrics']['mae']:.4f}")
        print(f"    Test SAE:       {result['final_metrics']['sae']:.4f}")
        print(f"    Test F1:        {result['final_metrics']['f1']:.4f}")
        print(f"    Test Precision: {result['final_metrics']['precision']:.4f}")
        print(f"    Test Recall:    {result['final_metrics']['recall']:.4f}")
