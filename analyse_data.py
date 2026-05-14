import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
APPS = ['dishwasher', 'fridge', 'microwave', 'washing_machine']
SPLITS = [('train', 1, '2014-11-09'), ('validation', 1, '2014-12-07'), ('test', 5, '2014-08-24')]

dfs = {}
for name, house, date in SPLITS:
    dfs[name] = pd.read_csv(
        os.path.join(DATA_DIR, f'UKDALE_HF_{name}.csv'),
        index_col='timestamp', parse_dates=True)

# ── 1. Basic stats ────────────────────────────────────────────────────────────
print("=" * 70)
print("1. BASIC STATISTICS  (Watts)")
print("=" * 70)
for name, house, date in SPLITS:
    df = dfs[name]
    print(f"\n--- {name.upper()}  House {house}  {date} ---")
    print(df.describe().round(1).to_string())

# ── 2. Activation rates ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("2. ACTIVATION RATES  (% time signal exceeds threshold)")
print("=" * 70)
for name, house, date in SPLITS:
    df = dfs[name]
    print(f"\n{name.upper()} House {house}:")
    for app in APPS:
        col = df[app]
        rates = "   ".join([f"{(col > t).mean()*100:5.1f}% (>{t}W)"
                            for t in [5, 10, 20, 50, 100]])
        print(f"  {app:<22} {rates}")

# ── 3. Power profile per appliance ────────────────────────────────────────────
print("\n" + "=" * 70)
print("3. POWER PROFILE  (nonzero readings)")
print("=" * 70)
for name, house, date in SPLITS:
    df = dfs[name]
    print(f"\n{name.upper()} House {house}:")
    for app in APPS:
        col = df[app]
        nz  = col[col > 0]
        if len(nz) == 0:
            print(f"  {app:<22} all zero")
            continue
        print(f"  {app:<22} "
              f"zero={100*(col==0).mean():5.1f}%  "
              f"min={col.min():7.1f}  "
              f"p5_nz={nz.quantile(0.05):7.1f}  "
              f"median_nz={nz.median():7.1f}  "
              f"p95_nz={nz.quantile(0.95):8.1f}  "
              f"max={col.max():8.1f}")

# ── 4. Aggregate mains stats ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("4. AGGREGATE MAINS SIGNAL")
print("=" * 70)
for name, house, date in SPLITS:
    df = dfs[name]
    agg = df['aggregate']
    diff1 = agg.diff().abs().dropna()
    print(f"\n{name.upper()} House {house}:")
    print(f"  Mean       : {agg.mean():.1f} W")
    print(f"  Std        : {agg.std():.1f} W")
    print(f"  Min / Max  : {agg.min():.1f} / {agg.max():.1f} W")
    print(f"  |dP/dt| mean : {diff1.mean():.2f} W/step  (6s)")
    print(f"  |dP/dt| p95  : {diff1.quantile(0.95):.2f} W/step")
    print(f"  |dP/dt| max  : {diff1.max():.2f} W/step")
    # Fraction of steps with large step changes
    for thr in [50, 100, 200, 500]:
        pct = (diff1 > thr).mean() * 100
        print(f"  Steps with |dP| > {thr:3d}W : {pct:.2f}%")

# ── 5. Appliance event counts (ON transitions) ────────────────────────────────
print("\n" + "=" * 70)
print("5. ON-EVENT COUNTS  (rising edges above threshold)")
print("=" * 70)
THR = 20.0
for name, house, date in SPLITS:
    df = dfs[name]
    print(f"\n{name.upper()} House {house}:")
    for app in APPS:
        col   = df[app]
        state = (col > THR).astype(int)
        rises = (state.diff() == 1).sum()
        # Mean ON duration
        blocks, on_dur = [], 0
        in_on = False
        for v in state:
            if v == 1:
                on_dur += 1
                in_on   = True
            elif in_on:
                blocks.append(on_dur)
                on_dur = 0
                in_on  = False
        avg_dur = np.mean(blocks) * 6 if blocks else 0  # seconds
        total_on_min = state.sum() * 6 / 60
        print(f"  {app:<22} ON-events={rises:4d}  "
              f"avg_duration={avg_dur/60:6.1f} min  "
              f"total_ON={total_on_min:6.1f} min")

# ── 6. Energy consumption ─────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("6. ENERGY CONSUMPTION  (Wh over 24-hour split)")
print("=" * 70)
for name, house, date in SPLITS:
    df = dfs[name]
    print(f"\n{name.upper()} House {house}:")
    for col_name in ['aggregate'] + APPS:
        wh = df[col_name].sum() * 6 / 3600  # 6s samples -> Wh
        print(f"  {col_name:<22} {wh:8.1f} Wh")

# ── 7. Cross-split energy fraction (how much of aggregate each app explains) ──
print("\n" + "=" * 70)
print("7. APPLIANCE ENERGY FRACTION  (% of aggregate Wh)")
print("=" * 70)
for name, house, date in SPLITS:
    df = dfs[name]
    agg_wh = df['aggregate'].sum() * 6 / 3600
    print(f"\n{name.upper()} House {house}  (aggregate={agg_wh:.1f} Wh):")
    total_labelled = 0
    for app in APPS:
        wh  = df[app].sum() * 6 / 3600
        pct = wh / agg_wh * 100
        total_labelled += pct
        print(f"  {app:<22} {wh:7.1f} Wh  ({pct:5.1f}%)")
    print(f"  {'Total labelled':<22} {total_labelled:5.1f}% of aggregate")
    print(f"  {'Unlabelled/other':<22} {100-total_labelled:5.1f}%")

# ── 8. Spectral summary: dominant periods ──────────────────────────────────────
print("\n" + "=" * 70)
print("8. SPECTRAL SUMMARY  (FFT dominant period in aggregate)")
print("=" * 70)
for name, house, date in SPLITS:
    df   = dfs[name]
    agg  = df['aggregate'].values - df['aggregate'].mean()
    fft  = np.abs(np.fft.rfft(agg))
    freq = np.fft.rfftfreq(len(agg), d=6.0)   # Hz
    # Skip DC
    fft[0] = 0
    top5   = np.argsort(fft)[::-1][:5]
    print(f"\n{name.upper()} House {house}:")
    for idx in top5:
        if freq[idx] > 0:
            period_min = 1 / freq[idx] / 60
            print(f"  Freq={freq[idx]:.5f} Hz  Period={period_min:.1f} min  Power={fft[idx]:.0f}")

# ── 9. House 5 standby analysis (key issue) ───────────────────────────────────
print("\n" + "=" * 70)
print("9. HOUSE 5 STANDBY POWER ANALYSIS  (test split)")
print("=" * 70)
df5 = dfs['test']
for app in APPS:
    col = df5[app]
    nz  = col[col > 0]
    print(f"\n{app}:")
    print(f"  Overall min            : {col.min():.2f} W")
    print(f"  % time == 0            : {(col == 0).mean()*100:.2f}%")
    print(f"  % time < 5 W           : {(col < 5).mean()*100:.2f}%")
    print(f"  % time < 10 W          : {(col < 10).mean()*100:.2f}%")
    print(f"  % time < 20 W          : {(col < 20).mean()*100:.2f}%")
    print(f"  % time < 50 W          : {(col < 50).mean()*100:.2f}%")
    print(f"  p5 nonzero             : {nz.quantile(0.05):.2f} W  (standby estimate)")
    print(f"  Adaptive threshold     : {nz.quantile(0.05) + 20:.2f} W")
    print(f"  ON% at fixed 10W thr   : {(col > 10).mean()*100:.1f}%")
    print(f"  ON% at adaptive thr    : {(col > nz.quantile(0.05)+20).mean()*100:.1f}%")

# ── 10. Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 5, figsize=(22, 12))
fig.suptitle('UKDALE HF Dataset Analysis  (6-second intervals)', fontsize=13)

for row, (name, house, date) in enumerate(SPLITS):
    df = dfs[name]
    t  = np.arange(len(df)) * 6 / 3600   # hours

    # Aggregate
    axes[row][0].plot(t, df['aggregate'], color='steelblue', linewidth=0.4)
    axes[row][0].set_title(f'{name.upper()} H{house} — Aggregate')
    axes[row][0].set_ylabel('Power (W)')

    for col_idx, app in enumerate(APPS):
        ax = axes[row][col_idx + 1]
        ax.plot(t, df[app], linewidth=0.4, color='tomato')
        ax.set_title(f'{app}')
        if row == 2:
            ax.set_xlabel('Time (hours)')

plt.tight_layout()
out = 'dataset/ukdale_analysis.png'
plt.savefig(out, dpi=120, bbox_inches='tight')
print(f"\n\nPlot saved -> {out}")
