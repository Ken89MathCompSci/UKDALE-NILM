"""
On-duration and time-of-day distribution analysis for APR-new-House1-dataset/,
broken down per split (train / validation / test).

NILM appliance models (HMM-based and neural) rely on two behavioural priors
per appliance:
    - ON-duration distribution -- how long a device stays on once it switches
      on (e.g. a microwave run is a couple of minutes; a dishwasher cycle is
      1-2 hours; a fridge compressor cycle is short and very regular).
    - Time-of-day distribution -- when during the day the device tends to be
      used/active (e.g. dishwasher/washing_machine cluster around morning and
      evening; fridge is roughly uniform across all 24 hours).

Splits (all House 1, chronologically contiguous, no gap/overlap):
    train      2014-09-08 to 2014-10-07 (30 days)
    validation 2014-10-08 to 2014-10-14 (7 days)
    test       2014-10-15 to 2014-10-21 (7 days)

Per split, per appliance:
    1. Threshold at 10 W (this repo's established House-1 convention) to get
       a binary ON/OFF state per 6s sample.
    2. Run-length-encode contiguous ON segments (vectorized, not a per-row
       Python loop -- fridge alone has ~2,000 cycles across just the 30-day
       training split).
    3. ON-duration distribution: summary stats (count, mean, median,
       p25/p75/p95, max) in minutes.
    4. Time-of-day distribution: ON-time fraction per hour-of-day bucket.

This is a direct follow-up to the multi-day dataset regeneration: we found
40% of *training* days have zero dishwasher activity, which turned out to
matter a lot for the shared-encoder PINN-LNN's dishwasher performance.
Splitting the analysis by split (rather than pooling all 44 days) lets you
see directly whether validation/test look like what the model actually
trains on, or whether a split's appliance behaviour has drifted.

Outputs (written into APR-new-House1-dataset/):
    - on_time_and_time_of_day_train.png       -- 4x2 grid, train split only
    - on_time_and_time_of_day_validation.png  -- 4x2 grid, validation split only
    - on_time_and_time_of_day_test.png        -- 4x2 grid, test split only
    - on_time_and_time_of_day_comparison.png  -- train/val/test overlaid per appliance
    - distribution_summary_by_split.json      -- per-split, per-appliance summary stats
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'APR-new-House1-dataset')
APPLIANCES  = ['dishwasher', 'fridge', 'microwave', 'washing_machine']
THRESHOLD_W = 10.0
SPLITS      = ['train', 'validation', 'test']
SPLIT_COLORS = {'train': 'steelblue', 'validation': 'darkorange', 'test': 'seagreen'}


def load_splits(dataset_dir=DATASET_DIR):
    """Load train/validation/test as separate DataFrames (no concatenation)."""
    dfs = {}
    for name in SPLITS:
        path = os.path.join(dataset_dir, f'UKDALE_HF_{name}.csv')
        df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
        dfs[name] = df
        n_days = (df.index[-1] - df.index[0]).days + 1
        print(f"{name:<12} {df.index.min()} to {df.index.max()}  "
              f"({len(df):,} rows, {n_days} days)")
    return dfs


def extract_on_segments(series, threshold=THRESHOLD_W):
    """
    Vectorized run-length encoding of contiguous ON (> threshold) segments.

    Returns:
        durations_min:  array of segment durations in minutes
        start_hours:    array of hour-of-day (0-23, fractional) at segment start
    """
    state = (series.values > threshold).astype(np.int8)
    padded = np.concatenate(([0], state, [0]))
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends   = np.where(diffs == -1)[0]   # exclusive end index

    step_seconds = 6.0
    durations_min = (ends - starts) * step_seconds / 60.0

    start_timestamps = series.index[starts]
    start_hours = start_timestamps.hour + start_timestamps.minute / 60.0

    return durations_min, np.asarray(start_hours)


def on_time_fraction_by_hour(series, threshold=THRESHOLD_W):
    """ON-time fraction per hour-of-day bucket (0-23), robust to appliances
    without discrete user-initiated events (e.g. fridge)."""
    state = series > threshold
    by_hour = state.groupby(series.index.hour)
    on_counts    = by_hour.sum()
    total_counts = by_hour.count()
    frac = (on_counts / total_counts).reindex(range(24), fill_value=0.0)
    return frac


def summarize(durations_min, span_hours):
    if len(durations_min) == 0:
        return {'count': 0, 'total_on_hours': 0.0, 'pct_of_span': 0.0}
    total_on_hours = float(np.sum(durations_min) / 60.0)
    return {
        'count':      int(len(durations_min)),
        'mean_min':   float(np.mean(durations_min)),
        'median_min': float(np.median(durations_min)),
        'p25_min':    float(np.percentile(durations_min, 25)),
        'p75_min':    float(np.percentile(durations_min, 75)),
        'p95_min':    float(np.percentile(durations_min, 95)),
        'max_min':    float(np.max(durations_min)),
        'total_on_hours': total_on_hours,
        'pct_of_span':    float(total_on_hours / span_hours * 100),
    }


def analyze_split(df, split_name):
    span_hours = len(df) * 6 / 3600
    results, summary = {}, {}

    for app in APPLIANCES:
        durations_min, start_hours = extract_on_segments(df[app])
        hour_frac = on_time_fraction_by_hour(df[app])
        results[app] = {'durations_min': durations_min, 'hour_frac': hour_frac}
        summary[app] = summarize(durations_min, span_hours)

    print(f"\n{'='*78}\n{split_name.upper()} -- ON-DURATION SUMMARY\n{'='*78}")
    for app in APPLIANCES:
        s = summary[app]
        print(f"\n{app}:")
        if s['count'] == 0:
            print("  No ON events found.")
            continue
        print(f"  events         : {s['count']:,}")
        print(f"  mean duration  : {s['mean_min']:7.2f} min")
        print(f"  median duration: {s['median_min']:7.2f} min")
        print(f"  p25 / p75      : {s['p25_min']:7.2f} / {s['p75_min']:7.2f} min")
        print(f"  total ON time  : {s['total_on_hours']:7.1f} hours ({s['pct_of_span']:.2f}% of split span)")

    print(f"\n{'='*78}\n{split_name.upper()} -- TIME-OF-DAY SUMMARY\n{'='*78}")
    for app in APPLIANCES:
        hour_frac = results[app]['hour_frac']
        if hour_frac.max() == 0:
            print(f"\n{app}: no ON time in this split.")
            continue
        peak_hour = int(hour_frac.idxmax())
        print(f"\n{app}: peak hour = {peak_hour:02d}:00-{peak_hour+1:02d}:00  "
              f"(ON {hour_frac.max()*100:.1f}% of the time in that hour)")

    return results, summary


def plot_split(results, split_name, out_dir=DATASET_DIR):
    fig, axes = plt.subplots(len(APPLIANCES), 2, figsize=(14, 4 * len(APPLIANCES)))
    fig.suptitle(f'APR-new-House1-dataset -- ON-Duration & Time-of-Day '
                 f'({split_name} split only)', fontsize=13)

    for row, app in enumerate(APPLIANCES):
        durations_min = results[app]['durations_min']
        hour_frac     = results[app]['hour_frac']

        ax_dur = axes[row][0]
        if len(durations_min) > 0:
            cap = np.percentile(durations_min, 99) if len(durations_min) > 1 else durations_min[0]
            cap = max(cap, 1.0)
            ax_dur.hist(np.clip(durations_min, 0, cap), bins=30,
                        color=SPLIT_COLORS[split_name], edgecolor='none')
            ax_dur.axvline(np.median(durations_min), color='red', linestyle='--',
                           linewidth=1, label=f"median={np.median(durations_min):.1f} min")
            ax_dur.legend(fontsize=8)
        else:
            ax_dur.text(0.5, 0.5, 'no ON events', ha='center', va='center',
                        transform=ax_dur.transAxes, color='gray')
        ax_dur.set_title(f'{app} -- ON-duration distribution')
        ax_dur.set_xlabel('Duration (min, capped at p99)')
        ax_dur.set_ylabel('Event count')
        ax_dur.grid(True, alpha=0.3)

        ax_tod = axes[row][1]
        ax_tod.bar(range(24), hour_frac.values * 100, color=SPLIT_COLORS[split_name], width=0.85)
        ax_tod.set_title(f'{app} -- Time-of-day (% ON per hour)')
        ax_tod.set_xlabel('Hour of day')
        ax_tod.set_ylabel('% time ON')
        ax_tod.set_xticks(range(0, 24, 2))
        ax_tod.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'on_time_and_time_of_day_{split_name}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved -> {out_path}")


def plot_comparison(all_results, out_dir=DATASET_DIR):
    """Overlay train/validation/test per appliance -- duration boxplot + time-of-day lines."""
    fig, axes = plt.subplots(len(APPLIANCES), 2, figsize=(14, 4 * len(APPLIANCES)))
    fig.suptitle('APR-new-House1-dataset -- Train vs Validation vs Test comparison', fontsize=13)

    for row, app in enumerate(APPLIANCES):
        ax_dur = axes[row][0]
        box_data, box_labels, box_colors = [], [], []
        for split in SPLITS:
            durations_min = all_results[split][app]['durations_min']
            if len(durations_min) > 0:
                cap = np.percentile(durations_min, 99) if len(durations_min) > 1 else durations_min[0]
                box_data.append(np.clip(durations_min, 0, max(cap, 1.0)))
            else:
                box_data.append(np.array([]))
            box_labels.append(f"{split}\n(n={len(durations_min)})")
            box_colors.append(SPLIT_COLORS[split])

        bp = ax_dur.boxplot(box_data, labels=box_labels, patch_artist=True, showfliers=False)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax_dur.set_title(f'{app} -- ON-duration by split')
        ax_dur.set_ylabel('Duration (min, capped at p99)')
        ax_dur.grid(True, alpha=0.3)

        ax_tod = axes[row][1]
        for split in SPLITS:
            hour_frac = all_results[split][app]['hour_frac']
            ax_tod.plot(range(24), hour_frac.values * 100, marker='o', markersize=3,
                       color=SPLIT_COLORS[split], label=split, linewidth=1.5)
        ax_tod.set_title(f'{app} -- Time-of-day by split')
        ax_tod.set_xlabel('Hour of day')
        ax_tod.set_ylabel('% time ON')
        ax_tod.set_xticks(range(0, 24, 2))
        ax_tod.legend(fontsize=8)
        ax_tod.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'on_time_and_time_of_day_comparison.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved -> {out_path}")


def main():
    dfs = load_splits()

    all_results, all_summaries = {}, {}
    for split in SPLITS:
        results, summary = analyze_split(dfs[split], split)
        all_results[split] = results
        all_summaries[split] = summary
        plot_split(results, split)

    plot_comparison(all_results)

    json_summary = {
        split: {
            app: {
                **all_summaries[split][app],
                'hour_frac': {str(h): float(all_results[split][app]['hour_frac'][h]) for h in range(24)},
            }
            for app in APPLIANCES
        }
        for split in SPLITS
    }
    json_path = os.path.join(DATASET_DIR, 'distribution_summary_by_split.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_summary, f, indent=2)
    print(f"\nSummary saved -> {json_path}")

    # -- Quick cross-split sanity flag: appliances with zero events in a split --
    print(f"\n{'='*78}\nCROSS-SPLIT COVERAGE CHECK\n{'='*78}")
    for app in APPLIANCES:
        counts = {split: all_summaries[split][app]['count'] for split in SPLITS}
        flag = "  <-- zero events in at least one split!" if 0 in counts.values() else ""
        print(f"  {app:<18} train={counts['train']:4d}  "
              f"validation={counts['validation']:4d}  test={counts['test']:4d}{flag}")


if __name__ == "__main__":
    main()
