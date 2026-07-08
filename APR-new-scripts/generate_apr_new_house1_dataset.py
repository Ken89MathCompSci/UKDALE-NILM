"""
Generates APR-new-House1-dataset/ -- single-day House 1 splits, same format
as APR-dataset/ (timestamp, aggregate, dishwasher, fridge, microwave,
washing_machine at 6s resolution), but on different calendar days so the
split is independent of APR-dataset's (2014-11-09 / 2014-12-07 / 2014-12-19).

Days were selected by scanning all of House 1's channel_*.dat files for the
day with the highest minimum activation count across dishwasher, microwave,
and washing_machine (the three appliances with intermittent/imbalanced
activity; fridge cycles continuously regardless of day):

    train      : 2015-12-22  (highest joint activity: DW=738, MW=466, WM=597)
    validation : 2014-10-11  (different season/year: DW=804, MW=316, WM=761)
    test       : 2016-01-22  (different season/year: DW=768, MW=331, WM=513)

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
    {"name": "train",      "day": "2015-12-22"},
    {"name": "validation", "day": "2014-10-11"},
    {"name": "test",       "day": "2016-01-22"},
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
    name = split["name"]
    day  = split["day"]
    start = f"{day} 00:00:00"
    end   = f"{day} 23:59:54"

    house_dir = os.path.join(UKDALE, "house_1")
    print(f"\n{'='*60}")
    print(f"Processing split: {name}  (House 1, {day})")
    print(f"{'='*60}")

    cols = {}
    for label, ch_num in H1_CHANNELS.items():
        print(f"  Loading channel_{ch_num}  ->  {label}")
        raw = load_channel(house_dir, ch_num)
        cols[label] = resample_to_grid(raw, start, end)

    df = pd.DataFrame(cols)
    df.index.name = "timestamp"

    print(f"\n  Shape       : {df.shape}")
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
