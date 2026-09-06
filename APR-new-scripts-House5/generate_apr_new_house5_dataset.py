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

A 44-day window (30 train + 7 val + 7 test) was selected by scanning every
30-day window across the full 2014-06-30 to 2014-11-13 range, maximising
microwave event count in the training split.

v1 selection (2014-07-25 to 2014-09-06) was the best fully gap-free 44-day
stretch but yielded only 16 MW events in training.

v2 selection (2014-08-04 to 2014-09-16) shifts training 10 days later to
capture the peak microwave-usage period, nearly doubling MW training events
to ~31. The validation split (2014-09-03 to 2014-09-09) spans the known
2014-09-07 partial-day gap (8422 samples vs ~13400 normal), which is handled
by the existing forward-fill policy (MAX_FILL=50 steps, ~5 min). One partial
day of patched data in validation is an acceptable trade-off for the
substantial improvement in training microwave coverage.

    train      : 2014-08-04 to 2014-09-02  (30 days, ~31 MW events)
    validation : 2014-09-03 to 2014-09-09  (7 days,  Sep 07 forward-filled)
    test       : 2014-09-10 to 2014-09-16  (7 days)

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
    {"name": "train",      "start": "2014-08-04 00:00:00", "end": "2014-09-02 23:59:54"},
    {"name": "validation", "start": "2014-09-03 00:00:00", "end": "2014-09-09 23:59:54"},
    {"name": "test",       "start": "2014-09-10 00:00:00", "end": "2014-09-16 23:59:54"},
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
