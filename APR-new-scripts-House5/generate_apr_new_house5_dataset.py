"""
Generates APR-new-House5-dataset/ -- multi-day House 5 splits, same column
format as APR-new-House1-dataset/ and APR-new-House2-dataset/ (timestamp,
aggregate, dishwasher, fridge, microwave, washing_machine at 6s resolution).

House 5 appliance channels (ukdale/metadata/building5.yaml):
    aggregate        -> channel_1
    dishwasher       -> channel_22
    fridge           -> channel_19  (fridge_freezer)
    microwave        -> channel_23
    washing_machine  -> channel_24  (washer_dryer)

Available data range common to all five channels: 2014-06-29 to 2014-11-13
(~137 days). Two gap regions were found by checking daily aggregate sample
counts against the ~13300-13480 samples/day baseline for this house:
    - 2014-09-07: partial-day gap (8422 samples, ~63% of a normal day)
    - 2014-10-15 to 2014-10-24: multi-day gap, including four fully-missing
      days (2014-10-16 to 2014-10-19) and several partial days either side

House 5's raw microwave (ch23) and washing_machine/washer_dryer (ch24)
channels never read 0 W -- they sit on a persistent standby floor (~48-52 W
for microwave, ~14-17 W for washer_dryer) that keeps them permanently "ON"
under the generic 10 W threshold used for Houses 1/2 (see DATASETS.md "House
5 standby artefact"). building5.yaml's own per-appliance on_power_threshold
values sit above these standby floors and were used instead:
    dishwasher       > 10 W  (on_power_threshold: 10)
    fridge            > 50 W  (on_power_threshold: 50)
    microwave        > 200 W  (on_power_threshold: 200)
    washing_machine  >  20 W  (on_power_threshold: 20)
These thresholds are for informational activation-rate printing only; the
saved CSVs contain raw power values so downstream scripts can apply
whichever threshold they need.

A 44-day contiguous window (30 train + 7 val + 7 test) was selected by
scanning daily appliance activation counts (at the thresholds above) across
every gap-free 44-day stretch in the 2014-06-30 to 2014-09-06 block (the
longest gap-free run, ending just before the 2014-09-07 partial-day gap),
maximising the minimum activation count across dishwasher, microwave, and
washing_machine (fridge cycles continuously regardless of day so it wasn't a
selection factor). House 5's real microwave usage is legitimately sparse
(short, infrequent zaps -- a few minutes/day at most) throughout the entire
recording, so no window fully escapes this; the chosen window is the best
available:

    train      : 2014-07-25 to 2014-08-23  (30 days)
                 DW=8009 hits, MW=225 hits, WM=12087 hits
    validation : 2014-08-24 to 2014-08-30  (7 days)
                 DW=3619 hits, MW=147 hits, WM=2802 hits
    test       : 2014-08-31 to 2014-09-06  (7 days)
                 DW=2763 hits, MW=171 hits, WM=2601 hits

Splits are chronologically contiguous (train -> val -> test, no overlap,
no gap) to keep the scenario realistic (no data leakage from shuffling
across time).

Gap-filling policy matches generate_apr_new_house1_dataset.py /
generate_apr_new_house2_dataset.py: forward-fill gaps <= 5 min
(50 x 6s steps), then zero-fill.
"""

import os
import numpy as np
import pandas as pd

BASE    = os.path.dirname(os.path.abspath(__file__))
UKDALE  = os.path.join(BASE, "..", "ukdale")
OUT_DIR = os.path.join(BASE, "..", "APR-new-House5-dataset")
os.makedirs(OUT_DIR, exist_ok=True)

H5_CHANNELS = {
    "aggregate":       1,
    "dishwasher":     22,
    "fridge":         19,
    "microwave":      23,
    "washing_machine": 24,
}

SPLITS = [
    {"name": "train",      "start": "2014-07-25 00:00:00", "end": "2014-08-23 23:59:54"},
    {"name": "validation", "start": "2014-08-24 00:00:00", "end": "2014-08-30 23:59:54"},
    {"name": "test",       "start": "2014-08-31 00:00:00", "end": "2014-09-06 23:59:54"},
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

    house_dir = os.path.join(UKDALE, "house_5")
    print(f"\n{'='*60}")
    print(f"Processing split: {name}  (House 5, {start[:10]} -> {end[:10]})")
    print(f"{'='*60}")

    cols = {}
    for label, ch_num in H5_CHANNELS.items():
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

    threshold = {"dishwasher": 10, "fridge": 50, "microwave": 200, "washing_machine": 20}
    print("\n  Appliance activation rate (% time > threshold):")
    for app, thr in threshold.items():
        pct = (df[app] > thr).mean() * 100
        print(f"    {app:<20s} (>{thr:3d}W): {pct:6.2f}%")

    out_path = os.path.join(OUT_DIR, f"UKDALE_HF_{name}.csv")
    df.to_csv(out_path)
    print(f"\n  Saved -> {out_path}")


if __name__ == "__main__":
    for split in SPLITS:
        generate_split(split)
    print("\nAll splits complete.")
