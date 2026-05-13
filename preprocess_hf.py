"""
Preprocessing script for UKDALE high-frequency (6s) NILM dataset.

Splits:
  Training  : House 1, 2014-11-09, 14400 points
  Validation: House 1, 2014-12-07, 14400 points
  Testing   : House 5, 2014-08-24, 14400 points

Appliances: Dishwasher, Fridge, Microwave, Washing Machine
"""

import os
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
UKDALE    = os.path.join(BASE, "ukdale")
OUT_DIR   = os.path.join(BASE, "dataset")
os.makedirs(OUT_DIR, exist_ok=True)

# ── channel maps ───────────────────────────────────────────────────────────────
# house_1 channel → appliance label
H1_CHANNELS = {
    "aggregate":       1,
    "dishwasher":      6,
    "fridge":         12,
    "microwave":      13,
    "washing_machine": 5,
}

# house_5 channel → appliance label
H5_CHANNELS = {
    "aggregate":       1,
    "dishwasher":     22,
    "fridge":         19,   # fridge_freezer
    "microwave":      23,
    "washing_machine":24,   # washer_dryer
}

# ── split definitions ──────────────────────────────────────────────────────────
SPLITS = [
    {
        "name":  "train",
        "house": 1,
        "channels": H1_CHANNELS,
        "start": "2014-11-09 00:00:00",
        "end":   "2014-11-09 23:59:54",
    },
    {
        "name":  "validation",
        "house": 1,
        "channels": H1_CHANNELS,
        "start": "2014-12-07 00:00:00",
        "end":   "2014-12-07 23:59:54",
    },
    {
        "name":  "test",
        "house": 5,
        "channels": H5_CHANNELS,
        "start": "2014-08-24 00:00:00",
        "end":   "2014-08-24 23:59:54",
    },
]

N_POINTS  = 14400          # expected data points per split
FREQ      = "6s"           # 6-second grid
MAX_FILL  = 50             # max consecutive 6s steps to forward-fill (~5 min)


# ── helpers ────────────────────────────────────────────────────────────────────

def load_channel(house_dir: str, channel: int) -> pd.Series:
    """Load a single channel .dat file → Series indexed by UTC datetime."""
    path = os.path.join(house_dir, f"channel_{channel}.dat")
    df   = pd.read_csv(path, sep=" ", header=None, names=["ts", "power"],
                       dtype={"ts": np.int64, "power": np.float64})
    df["datetime"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.drop_duplicates(subset="datetime").set_index("datetime")["power"]
    return df


def resample_to_grid(series: pd.Series, start: str, end: str) -> pd.Series:
    """
    Reindex series onto a regular 6-second grid, forward-fill short gaps,
    and zero-fill any remaining NaNs (appliances simply off during missing data).
    """
    grid = pd.date_range(start=start, end=end, freq=FREQ, tz="UTC")
    reindexed = series.reindex(grid.union(series.index)).sort_index()
    # interpolate tiny sub-6s jitter then resample to exact grid
    reindexed = (
        reindexed
        .resample(FREQ)
        .mean()                  # average if multiple readings in one 6s window
        .reindex(grid)
        .ffill(limit=MAX_FILL)   # forward-fill gaps up to ~5 min
        .fillna(0.0)             # zero-fill remaining gaps (appliance off)
    )
    reindexed = reindexed.clip(lower=0.0)   # no negative power
    return reindexed


# ── main ───────────────────────────────────────────────────────────────────────

def preprocess_split(split: dict) -> None:
    name       = split["name"]
    house_num  = split["house"]
    channels   = split["channels"]
    start      = split["start"]
    end        = split["end"]

    house_dir  = os.path.join(UKDALE, f"house_{house_num}")
    print(f"\n{'='*60}")
    print(f"Processing split: {name}  (House {house_num}, {start[:10]})")
    print(f"{'='*60}")

    cols = {}
    for label, ch_num in channels.items():
        print(f"  Loading channel_{ch_num}  ->  {label}")
        raw = load_channel(house_dir, ch_num)
        resampled = resample_to_grid(raw, start, end)
        cols[label] = resampled

    df = pd.DataFrame(cols)
    df.index.name = "timestamp"

    # ── diagnostics ────────────────────────────────────────────────────────────
    assert len(df) == N_POINTS, (
        f"Expected {N_POINTS} rows but got {len(df)}"
    )
    print(f"\n  Shape       : {df.shape}")
    print(f"  Time range  : {df.index[0]}  ->  {df.index[-1]}")
    print(f"\n  Power stats (W):")
    print(df.describe().round(2).to_string())

    # ── on/off activity ────────────────────────────────────────────────────────
    threshold = {"dishwasher": 10, "fridge": 10, "microwave": 10,
                 "washing_machine": 10}
    print("\n  Appliance activation rate (% time > threshold):")
    for app, thr in threshold.items():
        pct = (df[app] > thr).mean() * 100
        print(f"    {app:<20s}: {pct:6.2f}%")

    # ── save ───────────────────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, f"UKDALE_HF_{name}.csv")
    df.to_csv(out_path)
    print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    for split in SPLITS:
        preprocess_split(split)
    print("\nAll splits complete.")
