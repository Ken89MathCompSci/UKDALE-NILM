"""
Generates APR-new-House2-dataset/ -- multi-day House 2 splits, same column
format as APR-new-House1-dataset/ (timestamp, aggregate, dishwasher, fridge,
microwave, washing_machine at 6s resolution).

House 2 appliance channels:
    aggregate        -> channel_1
    washing_machine  -> channel_12
    dishwasher       -> channel_13
    fridge           -> channel_14
    microwave        -> channel_15

Available data range for all appliance channels: 2013-05-20 to 2013-10-09.

A 44-day contiguous window (30 train + 7 val + 7 test) was selected by
scanning daily appliance activation counts (>10 W) across the full range:

  - Avoided 2013-08-05 to 2013-08-22: household absence (all appliances read
    zero for all 18 days).
  - Avoided 2013-09-08 to 2013-09-10: secondary 3-day absence block.
  - Maximised minimum activation across dishwasher, microwave, washing_machine.

Selected window:
    train      : 2013-06-15 to 2013-07-14  (30 days)
                 DW=12360 hits, MW=13145 hits, WM=6280 hits
    validation : 2013-07-15 to 2013-07-21  (7 days)
                 DW=2154 hits, MW=751 hits, WM=673 hits
    test       : 2013-07-22 to 2013-07-28  (7 days)
                 DW=3102 hits, MW=830 hits, WM=2401 hits

Note: Jun 20-21 in the train window has anomalously high microwave readings
(~33% and ~9% ON respectively vs. a typical ~1-2%). This is real House 2
data; no samples are excluded.

House 2 sensor artefacts (documented in DATASETS.md):
  - fridge (ch14): never reads 0 W; idle floor ~10-11 W, compressor ~95 W.
    A >20 W threshold is recommended to distinguish cycling from idle.
  - washing_machine (ch12): idle standby ~3-4 W; a >10 W threshold correctly
    classifies this as OFF.
  - microwave (ch15): true 0 W when off -- cleanest channel across all houses.
  - dishwasher (ch13): off = 0-1 W; same pattern as House 1.

Gap-filling policy matches generate_apr_new_house1_dataset.py:
forward-fill gaps <= 5 min (50 x 6s steps), then zero-fill.
"""

import os
import numpy as np
import pandas as pd

BASE    = os.path.dirname(os.path.abspath(__file__))
UKDALE  = os.path.join(BASE, "..", "ukdale")
OUT_DIR = os.path.join(BASE, "..", "APR-new-House2-dataset")
os.makedirs(OUT_DIR, exist_ok=True)

H2_CHANNELS = {
    "aggregate":       1,
    "dishwasher":      13,
    "fridge":          14,
    "microwave":       15,
    "washing_machine": 12,
}

SPLITS = [
    {"name": "train",      "start": "2013-06-15 00:00:00", "end": "2013-07-14 23:59:54"},
    {"name": "validation", "start": "2013-07-15 00:00:00", "end": "2013-07-21 23:59:54"},
    {"name": "test",       "start": "2013-07-22 00:00:00", "end": "2013-07-28 23:59:54"},
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

    house_dir = os.path.join(UKDALE, "house_2")
    print(f"\n{'='*60}")
    print(f"Processing split: {name}  (House 2, {start[:10]} -> {end[:10]})")
    print(f"{'='*60}")

    cols = {}
    for label, ch_num in H2_CHANNELS.items():
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

    threshold = {"dishwasher": 10, "fridge": 20, "microwave": 10, "washing_machine": 10}
    print("\n  Appliance activation rate (% time > threshold):")
    for app, thr in threshold.items():
        pct = (df[app] > thr).mean() * 100
        print(f"    {app:<20s} (>{thr:2d}W): {pct:6.2f}%")

    out_path = os.path.join(OUT_DIR, f"UKDALE_HF_{name}.csv")
    df.to_csv(out_path)
    print(f"\n  Saved -> {out_path}")


if __name__ == "__main__":
    for split in SPLITS:
        generate_split(split)
    print("\nAll splits complete.")
