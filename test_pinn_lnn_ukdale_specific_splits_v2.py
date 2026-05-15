"""
PINN-LNN UKDALE specific splits v2.

This version trains the existing multi-output PINN-LNN on the CSV files in
new_dataset/ instead of the old data/ukdale/*.pkl files.

Expected files:
    new_dataset/UKDALE_HF_train.csv
    new_dataset/UKDALE_HF_validation.csv
    new_dataset/UKDALE_HF_test.csv

Expected CSV columns:
    timestamp, aggregate, dishwasher, fridge, microwave, washing_machine

The base training script expects:
    main, dish washer, fridge, microwave, washer dryer

So this wrapper only handles loading/renaming. The model, loss, metrics, and
plots are reused from test_pinn_lnn_ukdale_specific_splits.py.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd

import test_pinn_lnn_ukdale_specific_splits as base


DEFAULT_DATASET_DIR = "new_dataset"

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
    df = df[["main"] + base.APPLIANCES]
    df = df.astype("float32")
    df = df.clip(lower=0.0)
    return df


def load_new_dataset(dataset_dir=DEFAULT_DATASET_DIR):
    dataset_dir = os.path.abspath(dataset_dir)
    print(f"Loading UKDALE v2 data from: {dataset_dir}")

    data = {
        split_name: load_split(dataset_dir, split_name)
        for split_name in ["train", "val", "test"]
    }

    for split_name, df in data.items():
        print(
            f"{split_name:<5} rows={len(df):5d}  "
            f"range={df.index.min()} -> {df.index.max()}"
        )

    print(f"Available columns: {list(data['train'].columns)}")
    return data


def save_dataset_manifest(save_dir, dataset_dir):
    manifest = {
        "dataset_dir": os.path.abspath(dataset_dir),
        "splits": SPLIT_FILES,
        "column_mapping": CSV_TO_MODEL_COLUMNS,
        "appliances": base.APPLIANCES,
        "thresholds": base.THRESHOLDS,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    path = os.path.join(save_dir, "dataset_manifest_v2.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PINN-LNN on the preprocessed new_dataset CSV splits."
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help="Directory containing UKDALE_HF_train/validation/test.csv files.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="Hidden size for the shared liquid encoder.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Liquid dynamics integration step.",
    )
    parser.add_argument(
        "--lambda-phys",
        type=float,
        default=base.LAMBDA_PHYS,
        help="Physics consistency loss weight.",
    )
    parser.add_argument(
        "--epsilon-w",
        type=float,
        default=base.EPSILON_W,
        help="Allowed unlabelled/background wattage before physics penalty.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    missing_files = [
        os.path.join(args.dataset_dir, filename)
        for filename in SPLIT_FILES.values()
        if not os.path.exists(os.path.join(args.dataset_dir, filename))
    ]
    if missing_files:
        print("Missing required new_dataset files:")
        for path in missing_files:
            print(f"  {path}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"models/pinn_lnn_ukdale_v2_{timestamp}"

    data_dict = load_new_dataset(args.dataset_dir)
    test_metrics, history = base.train_pinn_model(
        data_dict,
        save_dir=save_dir,
        hidden_size=args.hidden_size,
        dt=args.dt,
        lambda_phys=args.lambda_phys,
        epsilon_w=args.epsilon_w,
    )

    manifest_path = save_dataset_manifest(save_dir, args.dataset_dir)
    print(f"\nDataset manifest saved to {manifest_path}")
    print(f"Results saved to {save_dir}")

    return test_metrics, history


if __name__ == "__main__":
    main()
