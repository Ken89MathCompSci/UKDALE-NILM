"""
PINN-LNN UKDALE specific splits v3.

Changes from v2:
  1. Uses new_dataset/ CSV splits.
  2. Predicts the full appliance window: (batch, WIN, 4), not only midpoint.
  3. Applies physics consistency across every timestep in the window.
  4. Removes the BCE term that destabilised training after warmup.
  5. Uses more realistic ON/OFF thresholds for UK-DALE appliance metrics.

Expected CSV columns:
  timestamp, aggregate, dishwasher, fridge, microwave, washing_machine
"""

import argparse
import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Source Code"))
from utils import calculate_nilm_metrics


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EPOCHS = 80
PATIENCE = 20
LR = 1e-3
BATCH = 32
WIN = 100
STRIDE = 5

LAMBDA_PHYS = 0.01
EPSILON_W = 50.0

APPLIANCES = ["dish washer", "fridge", "microwave", "washer dryer"]

THRESHOLDS = {
    "dish washer": 50.0,
    "fridge": 20.0,
    "microwave": 100.0,
    "washer dryer": 50.0,
}

CSV_TO_MODEL_COLUMNS = {
    "aggregate": "main",
    "dishwasher": "dish washer",
    "fridge": "fridge",
    "microwave": "microwave",
    "washing_machine": "washer dryer",
}

SPLIT_FILES = {
    "train": "UKDALE_HF_train.csv",
    "val": "UKDALE_HF_validation.csv",
    "test": "UKDALE_HF_test.csv",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class MultiApplianceSeqDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def load_split(dataset_dir, split_name):
    path = os.path.join(dataset_dir, SPLIT_FILES[split_name])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {split_name} split: {path}")

    df = pd.read_csv(path, parse_dates=["timestamp"])
    missing = [col for col in CSV_TO_MODEL_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df.rename(columns=CSV_TO_MODEL_COLUMNS)
    df = df.set_index("timestamp")
    df = df[["main"] + APPLIANCES]
    df = df.astype("float32").clip(lower=0.0)
    return df


def load_new_dataset(dataset_dir):
    dataset_dir = os.path.abspath(dataset_dir)
    print(f"Loading UKDALE v3 data from: {dataset_dir}")

    data = {split: load_split(dataset_dir, split) for split in ["train", "val", "test"]}
    for split, df in data.items():
        print(
            f"{split:<5} rows={len(df):5d}  "
            f"range={df.index.min()} -> {df.index.max()}"
        )
    print(f"Available columns: {list(data['train'].columns)}")
    return data


def create_sequences(data, window_size=WIN):
    mains = data["main"].values
    targets = data[APPLIANCES].values

    X, Y = [], []
    for i in range(0, len(mains) - window_size + 1, STRIDE):
        X.append(mains[i : i + window_size])
        Y.append(targets[i : i + window_size])

    return (
        np.array(X, dtype=np.float32).reshape(-1, window_size, 1),
        np.array(Y, dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class SequencePhysicsConsistencyLoss(nn.Module):
    """
    One-sided physics penalty over every timestep:

        ReLU(sum_i p_hat_i_raw[t] - aggregate_raw[t] - epsilon)
    """

    def __init__(self, x_scaler, y_scalers, epsilon_w=EPSILON_W):
        super().__init__()
        self.epsilon = epsilon_w

        self.register_buffer(
            "x_min",
            torch.tensor(float(x_scaler.data_min_[0]), dtype=torch.float32),
        )
        self.register_buffer(
            "x_range",
            torch.tensor(float(x_scaler.data_range_[0]), dtype=torch.float32),
        )

        y_mins = [float(scaler.data_min_[0]) for scaler in y_scalers]
        y_ranges = [float(scaler.data_range_[0]) for scaler in y_scalers]
        self.register_buffer("y_mins", torch.tensor(y_mins, dtype=torch.float32))
        self.register_buffer("y_ranges", torch.tensor(y_ranges, dtype=torch.float32))

    def forward(self, x_scaled, pred_scaled):
        x_raw = x_scaled.squeeze(-1) * self.x_range + self.x_min
        p_raw = pred_scaled * self.y_ranges.view(1, 1, -1) + self.y_mins.view(1, 1, -1)
        violation = F.relu(p_raw.sum(dim=2) - x_raw - self.epsilon)
        return violation.mean()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Seq2SeqPhysicsInformedLiquidNetworkModel(nn.Module):
    """
    Shared liquid recurrent encoder with per-timestep appliance heads.

    Input:
        x: (batch, seq_len, 1)
    Output:
        out: (batch, seq_len, n_appliances)
    """

    def __init__(self, input_size, hidden_size, n_appliances, dt=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_appliances = n_appliances
        self.dt = dt

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.tau_base = nn.Parameter(torch.ones(hidden_size))
        self.tau_mod = nn.Linear(input_size, hidden_size)
        self.rec_weights = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.rec_weights)
        self.gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.heads = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(n_appliances)])

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            input_proj = self.input_proj(x_t)
            rec_proj = torch.matmul(h, self.rec_weights)

            tau_base = F.softplus(self.tau_base).unsqueeze(0)
            tau_mod = torch.sigmoid(self.tau_mod(x_t))
            tau = (tau_base * tau_mod).clamp(min=self.dt)

            gate = torch.sigmoid(self.gate(torch.cat([x_t, h], dim=1)))
            f_t = torch.tanh(input_proj + rec_proj)
            dh = ((-h / tau) + gate * f_t) * self.dt
            h = (h + dh).clamp(-10.0, 10.0)

            h_t = self.norm(h)
            outputs.append(torch.cat([head(h_t) for head in self.heads], dim=1))

        return torch.stack(outputs, dim=1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def inverse_scale_sequence(values, scaler):
    flat = values.reshape(-1, 1)
    return scaler.inverse_transform(flat).reshape(values.shape)


def compute_per_appliance_metrics(y_true, y_pred, y_scalers):
    metrics = {}
    for i, app in enumerate(APPLIANCES):
        raw_true = inverse_scale_sequence(y_true[:, :, i], y_scalers[i]).reshape(-1)
        raw_pred = inverse_scale_sequence(y_pred[:, :, i], y_scalers[i]).reshape(-1)
        metrics[app] = calculate_nilm_metrics(
            raw_true, raw_pred, threshold=THRESHOLDS[app]
        )
    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(data_dict, save_dir, hidden_size=64, dt=0.1,
                lambda_phys=LAMBDA_PHYS, epsilon_w=EPSILON_W):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"seq2seq=True  lambda_phys={lambda_phys}  epsilon={epsilon_w} W")
    print(f"hidden={hidden_size}  dt={dt}  BCE=disabled")

    X_tr, Y_tr = create_sequences(data_dict["train"], WIN)
    X_va, Y_va = create_sequences(data_dict["val"], WIN)
    X_te, Y_te = create_sequences(data_dict["test"], WIN)

    x_scaler = MinMaxScaler()
    X_tr = x_scaler.fit_transform(X_tr.reshape(-1, 1)).reshape(X_tr.shape)
    X_va = x_scaler.transform(X_va.reshape(-1, 1)).reshape(X_va.shape)
    X_te = x_scaler.transform(X_te.reshape(-1, 1)).reshape(X_te.shape)

    y_scalers = []
    for i in range(len(APPLIANCES)):
        scaler = MinMaxScaler()
        Y_tr[:, :, i] = scaler.fit_transform(Y_tr[:, :, i].reshape(-1, 1)).reshape(Y_tr[:, :, i].shape)
        Y_va[:, :, i] = scaler.transform(Y_va[:, :, i].reshape(-1, 1)).reshape(Y_va[:, :, i].shape)
        Y_te[:, :, i] = scaler.transform(Y_te[:, :, i].reshape(-1, 1)).reshape(Y_te[:, :, i].shape)
        y_scalers.append(scaler)

    print(f"Train: {X_tr.shape} -> {Y_tr.shape}")
    print(f"Val:   {X_va.shape} -> {Y_va.shape}")
    print(f"Test:  {X_te.shape} -> {Y_te.shape}")

    tr_loader = torch.utils.data.DataLoader(
        MultiApplianceSeqDataset(X_tr, Y_tr), batch_size=BATCH, shuffle=True)
    va_loader = torch.utils.data.DataLoader(
        MultiApplianceSeqDataset(X_va, Y_va), batch_size=BATCH, shuffle=False)
    te_loader = torch.utils.data.DataLoader(
        MultiApplianceSeqDataset(X_te, Y_te), batch_size=BATCH, shuffle=False)

    model = Seq2SeqPhysicsInformedLiquidNetworkModel(
        input_size=1,
        hidden_size=hidden_size,
        n_appliances=len(APPLIANCES),
        dt=dt,
    ).to(device)

    mse_criterion = nn.MSELoss()
    phys_criterion = SequencePhysicsConsistencyLoss(
        x_scaler, y_scalers, epsilon_w=epsilon_w
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5)

    history = {
        "train_loss": [], "train_mse": [], "train_phys": [],
        "val_loss": [], "val_mse": [], "val_phys": [],
        "val_metrics": [],
    }
    best_val_mse = float("inf")
    best_state = None
    counter = 0

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print("Starting seq2seq PINN-LNN training...")

    for epoch in range(EPOCHS):
        model.train()
        ep_mse = ep_phys = ep_total = 0.0
        progress_bar = tqdm(tr_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False)

        for xb, yb in progress_bar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()

            pred = model(xb)
            mse_loss = mse_criterion(pred, yb)
            phys_loss = phys_criterion(xb, pred)
            loss = mse_loss + lambda_phys * phys_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ep_mse += mse_loss.item()
            ep_phys += phys_loss.item()
            ep_total += loss.item()
            progress_bar.set_postfix({
                "mse": f"{mse_loss.item():.5f}",
                "phys": f"{phys_loss.item():.3f}",
            })

        avg_tr_mse = ep_mse / len(tr_loader)
        avg_tr_phys = ep_phys / len(tr_loader)
        avg_tr_total = ep_total / len(tr_loader)
        history["train_mse"].append(avg_tr_mse)
        history["train_phys"].append(avg_tr_phys)
        history["train_loss"].append(avg_tr_total)

        model.eval()
        vl_mse = vl_phys = vl_total = 0.0
        val_preds, val_trues = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                mse_loss = mse_criterion(pred, yb)
                phys_loss = phys_criterion(xb, pred)
                loss = mse_loss + lambda_phys * phys_loss

                vl_mse += mse_loss.item()
                vl_phys += phys_loss.item()
                vl_total += loss.item()
                val_preds.append(pred.cpu().numpy())
                val_trues.append(yb.cpu().numpy())

        avg_va_mse = vl_mse / len(va_loader)
        avg_va_phys = vl_phys / len(va_loader)
        avg_va_total = vl_total / len(va_loader)
        history["val_mse"].append(avg_va_mse)
        history["val_phys"].append(avg_va_phys)
        history["val_loss"].append(avg_va_total)
        scheduler.step(avg_va_mse)

        y_pred_all = np.concatenate(val_preds)
        y_true_all = np.concatenate(val_trues)
        per_app_metrics = compute_per_appliance_metrics(y_true_all, y_pred_all, y_scalers)
        history["val_metrics"].append(per_app_metrics)

        avg_f1 = np.mean([per_app_metrics[a]["f1"] for a in APPLIANCES])
        avg_mae = np.mean([per_app_metrics[a]["mae"] for a in APPLIANCES])
        print(
            f"  Epoch {epoch + 1:3d}/{EPOCHS}  "
            f"train={avg_tr_total:.5f} (mse={avg_tr_mse:.5f} phys={avg_tr_phys:.3f})  "
            f"val={avg_va_total:.5f} (mse={avg_va_mse:.5f} phys={avg_va_phys:.3f})  "
            f"avgF1={avg_f1:.4f}  avgMAE={avg_mae:.2f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        for app in APPLIANCES:
            m = per_app_metrics[app]
            print(
                f"    {app:<14}  F1={m['f1']:.4f}  "
                f"P={m['precision']:.4f}  R={m['recall']:.4f}  "
                f"MAE={m['mae']:.2f}  SAE={m['sae']:.4f}"
            )

        if avg_va_mse < best_val_mse:
            best_val_mse = avg_va_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    test_preds, test_trues = [], []
    with torch.no_grad():
        for xb, yb in te_loader:
            test_preds.append(model(xb.to(device)).cpu().numpy())
            test_trues.append(yb.cpu().numpy())

    y_pred_te = np.concatenate(test_preds)
    y_true_te = np.concatenate(test_trues)
    test_metrics = compute_per_appliance_metrics(y_true_te, y_pred_te, y_scalers)

    print(f"\n{'Appliance':<15} {'F1':>8} {'Precision':>10} {'Recall':>8} {'MAE':>8} {'SAE':>8}")
    print("-" * 65)
    for app in APPLIANCES:
        m = test_metrics[app]
        print(
            f"{app:<15} {m['f1']:>8.4f} {m['precision']:>10.4f} "
            f"{m['recall']:>8.4f} {m['mae']:>8.2f} {m['sae']:>8.4f}"
        )

    plot_training(history, test_metrics, save_dir)
    save_results(save_dir, test_metrics, hidden_size, dt, lambda_phys, epsilon_w)
    return test_metrics, history


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def plot_training(history, test_metrics, save_dir):
    epochs_x = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs_x, history["train_loss"], label="Train total")
    plt.plot(epochs_x, history["val_loss"], label="Val total")
    plt.title("Total Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(epochs_x, history["train_mse"], label="Train MSE")
    plt.plot(epochs_x, history["val_mse"], label="Val MSE")
    plt.title("MSE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(epochs_x, history["train_phys"], label="Train Phys")
    plt.plot(epochs_x, history["val_phys"], label="Val Phys")
    plt.title("Physics Consistency Loss")
    plt.xlabel("Epoch")
    plt.ylabel("L_phys")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pinn_lnn_ukdale_v3_loss.png"), dpi=150, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(len(APPLIANCES), 2, figsize=(12, 4 * len(APPLIANCES)))
    fig.suptitle("PINN-LNN UKDALE v3 seq2seq - Per-Appliance Val Metrics", fontsize=13)

    for row, app in enumerate(APPLIANCES):
        f1_series = [m[app]["f1"] for m in history["val_metrics"]]
        mae_series = [m[app]["mae"] for m in history["val_metrics"]]

        axes[row][0].plot(epochs_x, f1_series)
        axes[row][0].axhline(test_metrics[app]["f1"], color="green", linestyle="--", label="Test F1")
        axes[row][0].set_title(f"{app} - F1")
        axes[row][0].set_xlabel("Epoch")
        axes[row][0].set_ylabel("F1")
        axes[row][0].legend()
        axes[row][0].grid(alpha=0.3)

        axes[row][1].plot(epochs_x, mae_series, color="red")
        axes[row][1].axhline(test_metrics[app]["mae"], color="green", linestyle="--", label="Test MAE")
        axes[row][1].set_title(f"{app} - MAE (W)")
        axes[row][1].set_xlabel("Epoch")
        axes[row][1].set_ylabel("MAE (W)")
        axes[row][1].legend()
        axes[row][1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pinn_lnn_ukdale_v3_per_appliance.png"), dpi=150, bbox_inches="tight")
    plt.close()


def save_results(save_dir, test_metrics, hidden_size, dt, lambda_phys, epsilon_w):
    config = {
        "dataset": "UKDALE new_dataset",
        "model": "Seq2SeqPhysicsInformedLiquidNetworkModel",
        "description": "full-window seq2seq PINN-LNN, BCE disabled",
        "window_size": WIN,
        "stride": STRIDE,
        "appliances": APPLIANCES,
        "thresholds": THRESHOLDS,
        "loss": f"MSE over full window + {lambda_phys} * sequence physics consistency",
        "model_params": {
            "input_size": 1,
            "hidden_size": hidden_size,
            "n_appliances": len(APPLIANCES),
            "dt": dt,
        },
        "train_params": {
            "lr": LR,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "batch": BATCH,
            "lambda_phys": lambda_phys,
            "epsilon_w": epsilon_w,
            "bce_enabled": False,
        },
        "test_metrics": {
            app: {k: float(v) for k, v in metrics.items()}
            for app, metrics in test_metrics.items()
        },
    }
    with open(os.path.join(save_dir, "pinn_lnn_ukdale_v3_results.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train seq2seq PINN-LNN on new_dataset CSV splits."
    )
    parser.add_argument("--dataset-dir", default="new_dataset")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--lambda-phys", type=float, default=LAMBDA_PHYS)
    parser.add_argument("--epsilon-w", type=float, default=EPSILON_W)
    return parser.parse_args()


def main():
    args = parse_args()
    missing = [
        os.path.join(args.dataset_dir, name)
        for name in SPLIT_FILES.values()
        if not os.path.exists(os.path.join(args.dataset_dir, name))
    ]
    if missing:
        print("Missing required new_dataset files:")
        for path in missing:
            print(f"  {path}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"models/pinn_lnn_ukdale_v3_{timestamp}"

    data = load_new_dataset(args.dataset_dir)
    test_metrics, history = train_model(
        data,
        save_dir=save_dir,
        hidden_size=args.hidden_size,
        dt=args.dt,
        lambda_phys=args.lambda_phys,
        epsilon_w=args.epsilon_w,
    )

    print(f"\nResults saved to {save_dir}")
    return test_metrics, history


if __name__ == "__main__":
    main()
