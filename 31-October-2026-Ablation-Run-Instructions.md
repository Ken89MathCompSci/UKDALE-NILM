# Ablation Run Instructions (31 October 2026)

How to run the AB-PILNN loss-term and time-constant ablations with
[`APR-new-scripts/ablations/run_ablation.py`](APR-new-scripts/ablations/run_ablation.py).

## What the script does

The script imports the exact script that produced each house's paper results and changes
one thing per variant. Everything else (features, CNN, WIN=299, thresholds, pos_weight
clamp, warmup/ramp, checkpoint-on-val-MSE) stays identical to the reported runs.

| House | Baseline script |
|---|---|
| 1 | `APR-new-scripts/combined_pinn_lnn_apr_new_house1_dataset.py` |
| 2 | `APR-new-scripts-House2/combined_pinn_lnn_apr_new_house2_dataset_v2.py` |
| 5 | `APR-new-scripts-House5/combined_pinn_lnn_apr_new_house5_dataset.py` |

| Variant | Change |
|---|---|
| `full` | None. The baseline rerun with the same seeds as the ablations. |
| `no_phys` | λ_phys = 0 (removes L_phys) |
| `no_event` | λ_event = 0 (the gate stays, but only MSE trains it) |
| `no_aux` | λ_phys = λ_event = 0 (MSE only) |
| `fixed_tau` | τ = max(Δt, softplus(θ_τ)), dropping the σ(W_τ x̃) input modulation |

## 1. Open Colab

Use a GPU runtime, then clone this branch:

```
!git clone -b enhancement-Oct-26 https://github.com/Ken89MathCompSci/UKDALE-NILM.git
%cd /content/UKDALE-NILM
```

If a clone already exists in the session, run this instead:

```
%cd /content/UKDALE-NILM
!git fetch && git checkout enhancement-Oct-26 && git pull
```

## 2. Run one house per cell

```
!python APR-new-scripts/ablations/run_ablation.py --house 1 --variant all --seeds 0 1 2
!python APR-new-scripts/ablations/run_ablation.py --house 2 --variant all --seeds 0 1 2
!python APR-new-scripts/ablations/run_ablation.py --house 5 --variant all --seeds 0 1 2
```

Each command does 15 training runs (5 variants × 3 seeds), which can outlast a single
Colab session. To split the work, run one variant per cell (shown here for House 1):

```
!python APR-new-scripts/ablations/run_ablation.py --house 1 --variant full      --seeds 0 1 2
!python APR-new-scripts/ablations/run_ablation.py --house 1 --variant no_phys   --seeds 0 1 2
!python APR-new-scripts/ablations/run_ablation.py --house 1 --variant fixed_tau --seeds 0 1 2
!python APR-new-scripts/ablations/run_ablation.py --house 1 --variant no_event  --seeds 0 1 2
!python APR-new-scripts/ablations/run_ablation.py --house 1 --variant no_aux    --seeds 0 1 2
```

Priorities:

- **Always run `full` with the same seeds.** The ablations are compared against it, not
  against the single unseeded number in the paper tables.
- **If time is short, run `no_phys` and `fixed_tau` first.** They test contribution (ii)
  and the adaptive-τ claim.

## 3. Outputs

- **Notebook output.** Each run prints a line like
  `MACRO h1 no_phys seed=0  MAE=...  SAE=...  F1=...  P=...  R=...`.
  The end of each cell prints a mean ± std table over the seeds.
- **Saved files.** Each run writes
  `models/ablation_h{house}_{variant}_seed{seed}_{timestamp}/ablation_summary.json`,
  which holds macro and per-appliance test metrics, alongside the checkpoint and plots.
- **Colab deletes `/content` when the session ends.** Keep the outputs: save the notebook
  to GitHub (as with the other result notebooks) or copy `models/ablation_*` to Drive.

## Notes

- **Existing single-seed ablations need checking across seeds.** These are
  `30-August-2026-House-One-Ablation-tests.ipynb` and
  `1-September_2026- House-2-Ablation-Results.ipynb`. On House 2 they show
  "no gate" (MAE 6.62, F1 0.84) and "GRU instead of liquid cell" (MAE 6.39) matching or
  beating the full model (MAE 6.61, F1 0.82). Use the seeded `full` runs to check
  whether these gaps hold.
- **House 5 has no ablations yet.** This script covers it.
- **Model size.** AB-PILNN has 92,168 trainable parameters (83,848 with fixed τ), for
  the "lightweight" claim in the paper.
