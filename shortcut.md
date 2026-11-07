# Ablation Shortcut: Single Run per Variant

Run each command in its own Colab cell from `/content/UKDALE-NILM` on branch
`enhancement-Oct-26`. Full instructions are in
[31-October-2026-Ablation-Run-Instructions.md](31-October-2026-Ablation-Run-Instructions.md).

## Setup

```
!git clone -b enhancement-Oct-26 https://github.com/Ken89MathCompSci/UKDALE-NILM.git
%cd /content/UKDALE-NILM
```

## Checkpoint rule (changed 7 November 2026)

`run_ablation.py` now only keeps checkpoints from epochs where the physics and event losses
are at full weight: epoch 21 for House 1, 30 for House 2, 20 for House 5. It also makes GPU
runs repeatable (`cudnn.deterministic`). Earlier runs (1-November, 4-November) always kept
an MSE-only warmup epoch, so all of them, including House 1 `full`, need rerunning.
To reproduce the old behaviour, add `--select-from-epoch 0`.

## House 1

```
!python APR-new-scripts/ablations/run_ablation.py --house 1 --variant full      --seeds 0
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
  The new `full` result replaces the AB-PILNN numbers in the paper.
- **Check the header line of each run.** It should say `checkpoint from epoch 21` (House 1),
  `30` (House 2) or `20` (House 5).
- **Read results from the `MACRO h{house} {variant} seed=0 ...` line** at the end of each
  cell. Each run also saves `models/ablation_h{house}_{variant}_seed0_{timestamp}/ablation_summary.json`.
- **Save outputs before the Colab session ends,** because `/content` is wiped.
