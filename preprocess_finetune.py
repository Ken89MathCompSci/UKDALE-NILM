"""
Preprocessing script for UKDALE fine-tuning split set.

Strategy: Cross-House with Fine-Tuning (Approach 3)
    Pretrain : House 1 data (two single days)
    Fine-tune: House 5 — first FINETUNE_HOURS of the Aug-24 test day
    Test     : House 5 — remaining hours of Aug-24 (held-out evaluation)

Builds directly from the existing `dataset/` CSVs so no raw .dat files are
re-read.  Column names and sampling rate (6 s) are preserved as-is.

Output files (saved to fine_tuning_dataset/):
    UKDALE_HF_pretrain.csv    — House 1, Nov-09 + Dec-07 combined (28 800 rows)
    UKDALE_HF_validation.csv  — House 1, Dec-07 only (14 400 rows)  [for val during pretrain]
    UKDALE_HF_finetune.csv    — House 5, Aug-24 first FINETUNE_HOURS
    UKDALE_HF_test.csv        — House 5, Aug-24 remaining hours

Configurable:
    FINETUNE_HOURS  — how many hours of House 5 to use for adaptation (default 2)
    SOURCE_DIR      — directory containing the original single-day CSVs
    OUT_DIR         — output directory
"""

import os
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────────
FINETUNE_HOURS = 2                          # 1–3 hours is realistic for fast adaptation
ROWS_PER_HOUR  = 3600 // 6                 # 600 rows per hour at 6-second resolution

SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
OUT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fine_tuning_dataset")
os.makedirs(OUT_DIR, exist_ok=True)

APPLIANCES = ['dishwasher', 'fridge', 'microwave', 'washing_machine']


# ── Load source CSVs ─────────────────────────────────────────────────────────

def load(name: str) -> pd.DataFrame:
    path = os.path.join(SOURCE_DIR, f"UKDALE_HF_{name}.csv")
    df = pd.read_csv(path, index_col='timestamp', parse_dates=True)
    print(f"  Loaded {name:12s}: {len(df):6,} rows  "
          f"[{df.index[0]}  to  {df.index[-1]}]")
    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("UKDALE Fine-Tuning Dataset Preprocessing")
    print(f"  Source : {SOURCE_DIR}")
    print(f"  Output : {OUT_DIR}")
    print(f"  Fine-tune window: {FINETUNE_HOURS} hour(s)  "
          f"({FINETUNE_HOURS * ROWS_PER_HOUR:,} rows)")
    print("=" * 60)

    # ── Source splits ────────────────────────────────────────────────────────
    print("\nLoading source splits...")
    h1_train = load("train")       # House 1, 2014-11-09
    h1_val   = load("validation")  # House 1, 2014-12-07
    h5_test  = load("test")        # House 5, 2014-08-24

    # ── Pretrain: House 1 train + val days combined ──────────────────────────
    pretrain = pd.concat([h1_train, h1_val])
    pretrain = pretrain.sort_index()

    # ── Fine-tune / Test split of House 5 day ────────────────────────────────
    n_ft = FINETUNE_HOURS * ROWS_PER_HOUR
    if n_ft >= len(h5_test):
        raise ValueError(
            f"FINETUNE_HOURS={FINETUNE_HOURS} would consume the entire test day "
            f"({len(h5_test)} rows).  Use a smaller value."
        )
    finetune = h5_test.iloc[:n_ft]
    test     = h5_test.iloc[n_ft:]

    # ── Print diagnostics ────────────────────────────────────────────────────
    print("\nOutput splits:")
    for label, df in [
        ("pretrain",   pretrain),
        ("validation", h1_val),
        ("finetune",   finetune),
        ("test",       test),
    ]:
        ts0 = df.index[0]
        ts1 = df.index[-1]
        print(f"\n  {label.upper()} ({len(df):,} rows)  "
              f"{str(ts0)[:19]}  to  {str(ts1)[:19]}")
        agg = df['aggregate']
        print(f"    aggregate  mean={agg.mean():.1f}W  std={agg.std():.1f}W  "
              f"min={agg.min():.0f}W  max={agg.max():.0f}W")
        print(f"    {'Appliance':<20}  {'mean':>8}  {'mean_ON':>9}  {'frac_ON':>8}  {'max':>7}")
        for app in APPLIANCES:
            col   = df[app]
            on    = col > 10
            m_on  = col[on].mean() if on.any() else 0.0
            print(f"    {app:<20}  {col.mean():>8.1f}W  {m_on:>8.1f}W  "
                  f"{on.mean():>8.3f}  {col.max():>7.0f}W")

    # ── Save ─────────────────────────────────────────────────────────────────
    print("\nSaving...")
    for label, df in [
        ("pretrain",   pretrain),
        ("validation", h1_val),
        ("finetune",   finetune),
        ("test",       test),
    ]:
        out_path = os.path.join(OUT_DIR, f"UKDALE_HF_{label}.csv")
        df.to_csv(out_path)
        print(f"  Saved {out_path}  ({len(df):,} rows)")

    print(f"\nDone.  Fine-tuning split: {FINETUNE_HOURS}h adapt / "
          f"{(len(h5_test) - n_ft) / ROWS_PER_HOUR:.1f}h test  "
          f"(House 5, 2014-08-24)")
    print(
        "\nDataset structure:\n"
        "  pretrain.csv    — House 1 Nov-09 + Dec-07 (pretrain model from scratch)\n"
        "  validation.csv  — House 1 Dec-07           (monitor val loss during pretrain)\n"
        f"  finetune.csv    — House 5 Aug-24 first {FINETUNE_HOURS}h (brief cross-house adaptation)\n"
        "  test.csv        — House 5 Aug-24 remainder (held-out evaluation)\n"
    )


if __name__ == "__main__":
    main()
