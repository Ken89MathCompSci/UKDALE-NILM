"""
Generates APR-new-House1-dataset/ -- multi-day House 1 splits, same column
format as APR-dataset/ (timestamp, aggregate, dishwasher, fridge, microwave,
washing_machine at 6s resolution).

Replaces the original single-day version of this script: single-day splits
gave the model too few positive windows for rare appliances (microwave,
washing_machine were only ~2.5-6% ON per day -> ~70-200 positive training
sequences after windowing), which produced degenerate high-recall/low-
precision models. Multi-day splits fix this directly.

Day range was chosen by scanning all of House 1's channel_*.dat files for a
44-day contiguous stretch (30 train + 7 val + 7 test) that (a) has no data
gaps (daily aggregate sample count >= 13000 of ~14400 expected), (b)
maximizes the minimum per-appliance activation count across dishwasher,
microwave, and washing_machine (fridge cycles continuously regardless of
day so it wasn't a selection factor), and (c) does not overlap the three
single dates already used by APR-dataset/ (2014-11-09 / 2014-12-07 /
2014-12-19):

    train      : 2014-09-08 to 2014-10-07  (30 days)
                 DW=10695 hits (12 zero-days), MW=5422 hits (0 zero-days),
                 WM=24126 hits (9 zero-days)
    validation : 2014-10-08 to 2014-10-14  (7 days)
                 DW=2478 hits (3 zero-days), MW=1446 hits (0 zero-days),
                 WM=5835 hits (1 zero-day)
    test       : 2014-10-15 to 2014-10-21  (7 days)
                 DW=1659 hits (3 zero-days), MW=1122 hits (0 zero-days),
                 WM=6941 hits (0 zero-days)

Splits are chronologically contiguous (train -> val -> test, no overlap,
no gap) to keep the scenario realistic (no data leakage from shuffling
across time).

Gap-filling policy matches preprocess_hf.py: forward-fill gaps <= 5 min,
then zero-fill (appliance assumed off).
"""

import os
import numpy as np
import pandas as pd

BASE    = os.path.dirname(os.path.abspath(__file__))
UKDALE  = os.path.join(BASE, "..", "ukdale")
OUT_DIR = os.path.join(BASE, "..", "APR-new-House1-dataset")
os.makedirs(OUT_DIR, exist_ok=True)

H1_CHANNELS = {
    "aggregate":       1,
    "dishwasher":      6,
    "fridge":         12,
    "microwave":      13,
    "washing_machine": 5,
}

SPLITS = [
    {"name": "train",      "start": "2014-09-08 00:00:00", "end": "2014-10-07 23:59:54"},
    {"name": "validation", "start": "2014-10-08 00:00:00", "end": "2014-10-14 23:59:54"},
    {"name": "test",       "start": "2014-10-15 00:00:00", "end": "2014-10-21 23:59:54"},
]

FREQ     = "6s"
MAX_FILL = 50  # ~5 min of consecutive 6s steps


def load_channel(house_dir: str, channel: int) -> pd.Series:
    path = os.path.join(house_dir, f"channel_{channel}.dat")
    df = pd.read_csv(path, sep=" ", header=None, names=["ts", "power"],
                      dtype={"ts": np.int64, "power": np.float64})
    df["datetime"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df.drop_duplicates(subset="datetime").set_index("datetime")["power"]


def resample_to_grid(series: pd.Series, start: str, end: str) -> pd.Series:
    grid = pd.date_range(start=start, end=end, freq=FREQ, tz="UTC")
    reindexed = series.reindex(grid.union(series.index)).sort_index()
    reindexed = (
        reindexed
        .resample(FREQ)
        .mean()
        .reindex(grid)
        .ffill(limit=MAX_FILL)
        .fillna(0.0)
    )
    return reindexed.clip(lower=0.0)


def generate_split(split: dict) -> None:
    name  = split["name"]
    start = split["start"]
    end   = split["end"]

    house_dir = os.path.join(UKDALE, "house_1")
    print(f"\n{'='*60}")
    print(f"Processing split: {name}  (House 1, {start[:10]} -> {end[:10]})")
    print(f"{'='*60}")

    cols = {}
    for label, ch_num in H1_CHANNELS.items():
        print(f"  Loading channel_{ch_num}  ->  {label}")
        raw = load_channel(house_dir, ch_num)
        cols[label] = resample_to_grid(raw, start, end)

    df = pd.DataFrame(cols)
    df.index.name = "timestamp"

    n_days = (df.index[-1] - df.index[0]).total_seconds() / 86400
    print(f"\n  Shape       : {df.shape}  ({n_days:.1f} days)")
    print(f"  Time range  : {df.index[0]}  ->  {df.index[-1]}")
    print(f"\n  Power stats (W):")
    print(df.describe().round(2).to_string())

    threshold = {"dishwasher": 10, "fridge": 10, "microwave": 10, "washing_machine": 10}
    print("\n  Appliance activation rate (% time > threshold):")
    for app, thr in threshold.items():
        pct = (df[app] > thr).mean() * 100
        print(f"    {app:<20s}: {pct:6.2f}%")

    out_path = os.path.join(OUT_DIR, f"UKDALE_HF_{name}.csv")
    df.to_csv(out_path)
    print(f"\n  Saved -> {out_path}")


if __name__ == "__main__":
    for split in SPLITS:
        generate_split(split)
    print("\nAll splits complete.")
