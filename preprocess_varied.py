"""
Preprocessing script: varied_dataset — multi-day, multi-season splits.

Selects non-contiguous days spread across different months and seasons to
maximise temporal variety.  Each split is a concatenation of full 24-hour
days; days with data gaps > 5 min are still kept (gap-filled) but logged.

Splits
------
Training   : House 1 — 7 days spanning Feb–Nov 2014 (winter, spring, summer,
             autumn) + 3 days from House 2 (Feb–Aug 2013) for cross-house
             training diversity.   Total: up to 10 days × 14,400 = 144,000 rows
Validation : House 1 — 4 days from Dec 2014 – Apr 2015
Testing    : House 5 — 5 days spanning Jul–Nov 2014

Appliances: dishwasher, fridge, microwave, washing_machine
"""

import os
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
UKDALE  = os.path.join(BASE, "ukdale")
OUT_DIR = os.path.join(BASE, "varied_dataset")
os.makedirs(OUT_DIR, exist_ok=True)

# ── channel maps ───────────────────────────────────────────────────────────────
H1_CHANNELS = {
    "aggregate":       1,
    "dishwasher":      6,
    "fridge":         12,
    "microwave":      13,
    "washing_machine": 5,
}

H2_CHANNELS = {
    "aggregate":        1,
    "dishwasher":      13,
    "fridge":          14,
    "microwave":       15,
    "washing_machine": 12,
}

H5_CHANNELS = {
    "aggregate":        1,
    "dishwasher":      22,
    "fridge":          19,   # fridge_freezer
    "microwave":       23,
    "washing_machine": 24,   # washer_dryer
}

# ── split definitions ──────────────────────────────────────────────────────────
# Each entry is one 24-hour day (14,400 rows at 6 s).
# Days are chosen to cover different months/seasons and avoid exact overlap
# with existing dataset/, new_dataset/, medium_dataset/ splits where possible.
TRAIN_DAYS = [
    # House 1 — spread across Feb–Nov 2014
    {"house": 1, "channels": H1_CHANNELS, "date": "2014-02-16"},  # winter
    {"house": 1, "channels": H1_CHANNELS, "date": "2014-04-20"},  # spring
    {"house": 1, "channels": H1_CHANNELS, "date": "2014-06-08"},  # early summer
    {"house": 1, "channels": H1_CHANNELS, "date": "2014-08-03"},  # late summer
    {"house": 1, "channels": H1_CHANNELS, "date": "2014-10-05"},  # autumn
    {"house": 1, "channels": H1_CHANNELS, "date": "2014-11-23"},  # late autumn
    {"house": 1, "channels": H1_CHANNELS, "date": "2015-01-18"},  # winter
    # House 2 — adds cross-house variety (Feb–Aug 2013)
    {"house": 2, "channels": H2_CHANNELS, "date": "2013-03-10"},  # spring H2
    {"house": 2, "channels": H2_CHANNELS, "date": "2013-06-15"},  # summer H2
    {"house": 2, "channels": H2_CHANNELS, "date": "2013-09-01"},  # autumn H2
]

VALID_DAYS = [
    {"house": 1, "channels": H1_CHANNELS, "date": "2014-12-14"},  # Dec H1
    {"house": 1, "channels": H1_CHANNELS, "date": "2015-02-08"},  # Feb H1
    {"house": 1, "channels": H1_CHANNELS, "date": "2015-03-22"},  # Mar H1
    {"house": 1, "channels": H1_CHANNELS, "date": "2015-05-10"},  # May H1
]

TEST_DAYS = [
    {"house": 5, "channels": H5_CHANNELS, "date": "2014-07-06"},  # early Jul H5
    {"house": 5, "channels": H5_CHANNELS, "date": "2014-08-03"},  # Aug H5
    {"house": 5, "channels": H5_CHANNELS, "date": "2014-09-07"},  # Sep H5
    {"house": 5, "channels": H5_CHANNELS, "date": "2014-10-12"},  # Oct H5
    {"house": 5, "channels": H5_CHANNELS, "date": "2014-11-08"},  # Nov H5
]

FREQ     = "6s"
MAX_FILL = 50   # forward-fill limit (~5 min)

# ── helpers ────────────────────────────────────────────────────────────────────

def load_channel(house_dir: str, channel: int) -> pd.Series:
    path = os.path.join(house_dir, f"channel_{channel}.dat")
    df   = pd.read_csv(path, sep=" ", header=None, names=["ts", "power"],
                       dtype={"ts": np.int64, "power": np.float64})
    df["datetime"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.drop_duplicates(subset="datetime").set_index("datetime")["power"]
    return df


def extract_day(series: pd.Series, date: str) -> pd.Series:
    """Reindex series onto the 14,400-point 6-second grid for a given date."""
    start = f"{date} 00:00:00"
    end   = f"{date} 23:59:54"
    grid  = pd.date_range(start=start, end=end, freq=FREQ, tz="UTC")

    reindexed = (
        series
        .reindex(grid.union(series.index))
        .sort_index()
        .resample(FREQ).mean()
        .reindex(grid)
        .ffill(limit=MAX_FILL)
        .fillna(0.0)
        .clip(lower=0.0)
    )

    missing_frac = reindexed.eq(0.0).mean()
    if missing_frac > 0.5:
        print(f"    WARNING: {date} has >{missing_frac:.0%} zeros — likely sparse/missing data")

    return reindexed


def build_split(day_list: list, split_name: str) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"Building split: {split_name}  ({len(day_list)} days)")
    print(f"{'='*60}")

    frames = []
    for entry in day_list:
        house    = entry["house"]
        channels = entry["channels"]
        date     = entry["date"]
        print(f"\n  House {house}  {date}")

        house_dir = os.path.join(UKDALE, f"house_{house}")
        cols = {}
        for label, ch_num in channels.items():
            raw = load_channel(house_dir, ch_num)
            cols[label] = extract_day(raw, date)
            print(f"    ch{ch_num:>2} ({label:<16}) min={cols[label].min():.1f}  max={cols[label].max():.1f}  "
                  f"mean={cols[label].mean():.1f}")

        day_df = pd.DataFrame(cols)
        day_df.index.name = "timestamp"
        frames.append(day_df)

    df = pd.concat(frames)
    print(f"\n  Total rows : {len(df):,}")
    print(f"  Date range : {df.index[0].date()} to {df.index[-1].date()}")

    # ── activation rates ────────────────────────────────────────────────────
    print("\n  Activation rates (% rows > 10 W):")
    for app in ["dishwasher", "fridge", "microwave", "washing_machine"]:
        pct = (df[app] > 10).mean() * 100
        print(f"    {app:<20}: {pct:5.1f}%")

    return df


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    splits = [
        ("train",      TRAIN_DAYS),
        ("validation", VALID_DAYS),
        ("test",       TEST_DAYS),
    ]

    for name, day_list in splits:
        df       = build_split(day_list, name)
        out_path = os.path.join(OUT_DIR, f"UKDALE_HF_{name}.csv")
        df.to_csv(out_path)
        print(f"\n  Saved: {out_path}")

    print("\nAll splits complete.")
