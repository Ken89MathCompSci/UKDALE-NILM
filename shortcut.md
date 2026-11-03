# Ablation Shortcut: Single Run per Variant

Run each command in its own Colab cell from `/content/UKDALE-NILM` on branch
`enhancement-Oct-26`. Full instructions are in
[31-October-2026-Ablation-Run-Instructions.md](31-October-2026-Ablation-Run-Instructions.md).

## Setup

```
!git clone -b enhancement-Oct-26 https://github.com/Ken89MathCompSci/UKDALE-NILM.git
%cd /content/UKDALE-NILM
```

## House 1

`full` (seed 0) is already done: MAE 12.85, SAE 9.92, F1 0.696, P 0.616, R 0.825.

```
!python APR-new-scripts/ablations/run_ablation.py --house 1 --variant no_phys   --seeds 0
!python APR-new-scripts/ablations/run_ablation.py --house 1 --variant no_event  --seeds 0
!python APR-new-scripts/ablations/run_ablation.py --house 1 --variant no_aux    --seeds 0
!python APR-new-scripts/ablations/run_ablation.py --house 1 --variant fixed_tau --seeds 0
```

## House 2

```
!python APR-new-scripts/ablations/run_ablation.py --house 2 --variant full      --seeds 0
!python APR-new-scripts/ablations/run_ablation.py --house 2 --variant no_phys   --seeds 0
!python APR-new-scripts/ablations/run_ablation.py --house 2 --variant no_event  --seeds 0
!python APR-new-scripts/ablations/run_ablation.py --house 2 --variant no_aux    --seeds 0
!python APR-new-scripts/ablations/run_ablation.py --house 2 --variant fixed_tau --seeds 0
```

## House 5

```
!python APR-new-scripts/ablations/run_ablation.py --house 5 --variant full      --seeds 0
!python APR-new-scripts/ablations/run_ablation.py --house 5 --variant no_phys   --seeds 0
!python APR-new-scripts/ablations/run_ablation.py --house 5 --variant no_event  --seeds 0
!python APR-new-scripts/ablations/run_ablation.py --house 5 --variant no_aux    --seeds 0
!python APR-new-scripts/ablations/run_ablation.py --house 5 --variant fixed_tau --seeds 0
```

## Notes

- **Compare against the seed-0 `full` run of the same house,** not the paper's unseeded runs.
- **Expect `no_phys`, `no_event` and `no_aux` to match `full` closely.** The best checkpoint
  is selected during the MSE-only warmup, before those loss terms switch on.
- **Read results from the `MACRO h{house} {variant} seed=0 ...` line** at the end of each
  cell. Each run also saves `models/ablation_h{house}_{variant}_seed0_{timestamp}/ablation_summary.json`.
- **Save outputs before the Colab session ends,** because `/content` is wiped.
