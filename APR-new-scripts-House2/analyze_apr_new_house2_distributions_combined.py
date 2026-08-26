"""
On-duration and time-of-day distribution analysis for APR-new-House2-dataset/,
with all splits (train + validation + test) concatenated into a single
combined view -- the House 2 counterpart to House 1's
on_time_and_time_of_day_analysis.png / distribution_summary.json.

Reuses the threshold definitions, segment-extraction, and summary logic from
analyze_apr_new_house2_distributions.py (the per-split analysis) so the two
scripts stay consistent; this script only adds the combined-dataset view.

Outputs (written into APR-new-House2-dataset/):
    - on_time_and_time_of_day_analysis.png  -- 4x2 grid, all days combined
    - distribution_summary.json             -- combined, per-appliance summary stats
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyze_apr_new_house2_distributions import (
    DATASET_DIR, APPLIANCES, THRESHOLDS, SPLITS,
    load_splits, extract_on_segments, on_time_fraction_by_hour, summarize,
)

COMBINED_COLOR = 'steelblue'


def combine_splits(dfs):
    """Concatenate train/validation/test into one chronologically-sorted DataFrame."""
    combined = pd.concat([dfs[s] for s in SPLITS]).sort_index()
    return combined


def analyze_combined(df):
    span_hours = len(df) * 6 / 3600
    n_days = (df.index[-1] - df.index[0]).days + 1
    results, summary = {}, {}

    for app in APPLIANCES:
        thr = THRESHOLDS[app]
        durations_min, start_hours = extract_on_segments(df[app], thr)
        hour_frac = on_time_fraction_by_hour(df[app], thr)
        results[app] = {'durations_min': durations_min, 'hour_frac': hour_frac}
        summary[app] = summarize(durations_min, span_hours)

    print(f"\n{'='*78}\nCOMBINED ({n_days} days) -- ON-DURATION SUMMARY\n{'='*78}")
    for app in APPLIANCES:
        s = summary[app]
        thr = THRESHOLDS[app]
        print(f"\n{app}  (threshold={thr:.0f} W):")
        if s['count'] == 0:
            print("  No ON events found.")
            continue
        print(f"  events         : {s['count']:,}")
        print(f"  mean duration  : {s['mean_min']:7.2f} min")
        print(f"  median duration: {s['median_min']:7.2f} min")
        print(f"  p25 / p75      : {s['p25_min']:7.2f} / {s['p75_min']:7.2f} min")
        print(f"  total ON time  : {s['total_on_hours']:7.1f} hours  "
              f"({s['pct_of_span']:.2f}% of combined span)")

    print(f"\n{'='*78}\nCOMBINED -- TIME-OF-DAY SUMMARY\n{'='*78}")
    for app in APPLIANCES:
        hour_frac = results[app]['hour_frac']
        if hour_frac.max() == 0:
            print(f"\n{app}: no ON time found.")
            continue
        peak_hour = int(hour_frac.idxmax())
        print(f"\n{app}: peak hour = {peak_hour:02d}:00-{peak_hour+1:02d}:00  "
              f"(ON {hour_frac.max()*100:.1f}% of the time in that hour)")

    return results, summary, n_days


def plot_combined(results, n_days, out_dir=DATASET_DIR):
    fig, axes = plt.subplots(len(APPLIANCES), 2, figsize=(14, 4 * len(APPLIANCES)))
    fig.suptitle(f'APR-new-House2-dataset -- ON-Duration & Time-of-Day '
                 f'Distributions (House 2, {n_days} days combined)', fontsize=13)

    for row, app in enumerate(APPLIANCES):
        thr           = THRESHOLDS[app]
        durations_min = results[app]['durations_min']
        hour_frac     = results[app]['hour_frac']

        ax_dur = axes[row][0]
        if len(durations_min) > 0:
            cap = np.percentile(durations_min, 99) if len(durations_min) > 1 else durations_min[0]
            cap = max(cap, 1.0)
            ax_dur.hist(np.clip(durations_min, 0, cap), bins=30,
                        color=COMBINED_COLOR, edgecolor='none')
            ax_dur.axvline(np.median(durations_min), color='red', linestyle='--',
                           linewidth=1, label=f"median={np.median(durations_min):.1f} min")
            ax_dur.legend(fontsize=8)
        else:
            ax_dur.text(0.5, 0.5, 'no ON events', ha='center', va='center',
                        transform=ax_dur.transAxes, color='gray')
        ax_dur.set_title(f'{app}  (>{thr:.0f} W) -- ON-duration distribution')
        ax_dur.set_xlabel('Duration (min, capped at p99)')
        ax_dur.set_ylabel('Event count')
        ax_dur.grid(True, alpha=0.3)

        ax_tod = axes[row][1]
        ax_tod.bar(range(24), hour_frac.values * 100,
                   color='coral', width=0.85)
        ax_tod.set_title(f'{app}  (>{thr:.0f} W) -- Time-of-day (% ON per hour)')
        ax_tod.set_xlabel('Hour of day')
        ax_tod.set_ylabel('% time ON')
        ax_tod.set_xticks(range(0, 24, 2))
        ax_tod.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, 'on_time_and_time_of_day_analysis.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved -> {out_path}")


def main():
    dfs = load_splits()
    combined = combine_splits(dfs)
    print(f"\nCombined: {combined.index.min()} to {combined.index.max()}  "
          f"({len(combined):,} rows)")

    results, summary, n_days = analyze_combined(combined)
    plot_combined(results, n_days)

    json_summary = {
        app: {
            **summary[app],
            'threshold_w': THRESHOLDS[app],
            'hour_frac': {
                str(h): float(results[app]['hour_frac'][h])
                for h in range(24)
            },
        }
        for app in APPLIANCES
    }
    json_path = os.path.join(DATASET_DIR, 'distribution_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_summary, f, indent=2)
    print(f"Summary saved -> {json_path}")


if __name__ == "__main__":
    main()
