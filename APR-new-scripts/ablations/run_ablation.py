"""
run_ablation.py
===============
Loss-term and time-constant ablations of AB-PILNN for Houses 1, 2 and 5.

Unlike ablation1-4 (standalone copies), this wrapper imports the exact script
that produced each house's paper results and changes ONE thing, so every other
setting (features, CNN, WIN=299, thresholds, pos_weight clamp, warmup/ramp,
checkpoint-on-val-MSE) is guaranteed identical to the baseline.

CHECKPOINT SELECTION: by default only epochs where the auxiliary losses are at
full weight are eligible for checkpointing/early stopping (0-based epoch 20 for
H1, 29 for H2, 19 for H5). The original scripts always picked a warmup (MSE-only)
epoch, so lambda_phys/lambda_event never affected the saved model. Pass
--select-from-epoch 0 to reproduce the original behaviour.

Baselines:

    House 1: APR-new-scripts/combined_pinn_lnn_apr_new_house1_dataset.py
    House 2: APR-new-scripts-House2/combined_pinn_lnn_apr_new_house2_dataset_v2.py
    House 5: APR-new-scripts-House5/combined_pinn_lnn_apr_new_house5_dataset.py

VARIANTS
--------
    full       -- baseline, unchanged (rerun with the same seeds as the ablations)
    no_phys    -- lambda_phys = 0         (removes L_phys)
    no_event   -- lambda_event = 0        (gate kept, but trained by MSE only)
    no_aux     -- lambda_phys = lambda_event = 0   (MSE only)
    fixed_tau  -- tau = max(dt, softplus(tau_base)): learnable per unit but
                  input-independent (drops the sigmoid(W_tau x) modulation)

Usage (from repo root):
    python APR-new-scripts/ablations/run_ablation.py --house 1 --variant fixed_tau --seed 0
    python APR-new-scripts/ablations/run_ablation.py --house 5 --variant all --seeds 0 1 2
"""

import sys
import os
import json
import random
import importlib.util
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

BASELINES = {
    1: os.path.join(ROOT, 'APR-new-scripts',        'combined_pinn_lnn_apr_new_house1_dataset.py'),
    2: os.path.join(ROOT, 'APR-new-scripts-House2', 'combined_pinn_lnn_apr_new_house2_dataset_v2.py'),
    5: os.path.join(ROOT, 'APR-new-scripts-House5', 'combined_pinn_lnn_apr_new_house5_dataset.py'),
}
VARIANTS = ['full', 'no_phys', 'no_event', 'no_aux', 'fixed_tau']


# ---------------------------------------------------------------------------
# Fixed-tau LiquidCell  (identical to baseline LiquidCell except tau)
# ---------------------------------------------------------------------------

class FixedTauLiquidCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dt: float = 0.1):
        super().__init__()
        self.dt          = dt
        self.hidden_size = hidden_size
        self.input_proj  = nn.Linear(input_size, hidden_size)
        self.tau_base    = nn.Parameter(torch.ones(hidden_size))
        # No tau_mod in this ablation.
        self.rec_weights = nn.Parameter(torch.empty(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.rec_weights)
        self.gate        = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x_t, h):
        ip  = self.input_proj(x_t)
        rp  = h @ self.rec_weights
        tau = F.softplus(self.tau_base).unsqueeze(0).clamp(min=self.dt)
        g   = torch.sigmoid(self.gate(torch.cat([x_t, h], dim=1)))
        dh  = ((-h / tau) + g * torch.tanh(ip + rp)) * self.dt
        return (h + dh).clamp(-10.0, 10.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_baseline(house: int):
    path = BASELINES[house]
    spec = importlib.util.spec_from_file_location(f'baseline_h{house}', path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def aux_fully_on_epoch(mod) -> int:
    """First 0-based epoch with the auxiliary losses at full weight (H1 20, H2 29, H5 19)."""
    return mod.WARMUP_EPOCHS + getattr(mod, 'RAMP_EPOCHS', 1) - 1


def macro(test_metrics: dict) -> dict:
    keys = ['mae', 'sae', 'f1', 'precision', 'recall']
    return {k: float(np.mean([m[k] for m in test_metrics.values()])) for k in keys}


def run_one(house: int, variant: str, seed: int, dataset_dir: str = None,
            select_from_epoch: int = None,
            lambda_phys: float = None, lambda_event: float = None) -> dict:
    mod = load_baseline(house)   # fresh module each run, so patches never leak
    set_seed(seed)
    if select_from_epoch is None:
        select_from_epoch = aux_fully_on_epoch(mod)

    # Overrides replace the house default; the variant may still zero them below.
    tag = ''
    if lambda_phys is not None or lambda_event is not None:
        tag = f'_lp{lambda_phys if lambda_phys is not None else mod.LAMBDA_PHYS:g}' \
              f'_le{lambda_event if lambda_event is not None else mod.LAMBDA_EVENT:g}'
    lambda_phys  = mod.LAMBDA_PHYS  if lambda_phys  is None else lambda_phys
    lambda_event = mod.LAMBDA_EVENT if lambda_event is None else lambda_event
    if variant in ('no_phys', 'no_aux'):
        lambda_phys = 0.0
    if variant in ('no_event', 'no_aux'):
        lambda_event = 0.0
    if variant == 'fixed_tau':
        mod.LiquidCell = FixedTauLiquidCell   # model class looks it up at construction

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir  = os.path.join(ROOT, 'models', f'ablation_h{house}_{variant}{tag}_seed{seed}_{timestamp}')
    print(f"\n{'=' * 85}\nHouse {house} | variant={variant} | seed={seed} | "
          f"lambda_phys={lambda_phys} lambda_event={lambda_event} | "
          f"checkpoint from epoch {select_from_epoch + 1}\n{'=' * 85}")

    data_dict = mod.load_data(dataset_dir or mod.DEFAULT_DATASET_DIR)
    test_metrics, _ = mod.train(data_dict, save_dir,
                                hidden       = 64,
                                dt           = 0.1,
                                lambda_phys  = lambda_phys,
                                lambda_event = lambda_event,
                                epsilon_w    = mod.EPSILON_W,
                                select_from_epoch = select_from_epoch)

    summary = {'house': house, 'variant': variant, 'seed': seed,
               'lambda_phys': lambda_phys, 'lambda_event': lambda_event,
               'select_from_epoch': select_from_epoch,
               'baseline_script': os.path.relpath(BASELINES[house], ROOT),
               'macro': macro(test_metrics),
               'test_metrics': {a: {k: float(v) for k, v in m.items()} for a, m in test_metrics.items()}}
    with open(os.path.join(save_dir, 'ablation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    m = summary['macro']
    print(f"\nMACRO h{house} {variant:<9} seed={seed}  lp={lambda_phys:g} le={lambda_event:g}  MAE={m['mae']:.2f}  SAE={m['sae']:.2f}  "
          f"F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--house',       type=int, required=True, choices=sorted(BASELINES))
    parser.add_argument('--variant',     default='all', choices=VARIANTS + ['all'])
    parser.add_argument('--seeds',       type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--dataset-dir', default=None)
    parser.add_argument('--select-from-epoch', type=int, default=None,
                        help='0-based first epoch eligible for checkpointing; default = epoch the '
                             'aux losses reach full weight; 0 = original (warmup-inclusive) behaviour')
    parser.add_argument('--lambda-phys',  type=float, default=None,
                        help='override the house default lambda_phys (no_phys/no_aux still set it to 0)')
    parser.add_argument('--lambda-event', type=float, default=None,
                        help='override the house default lambda_event (no_event/no_aux still set it to 0)')
    args = parser.parse_args()

    variants = VARIANTS if args.variant == 'all' else [args.variant]
    results  = [run_one(args.house, v, s, args.dataset_dir, args.select_from_epoch,
                        args.lambda_phys, args.lambda_event)
                for v in variants for s in args.seeds]

    print(f"\n{'=' * 85}\nSUMMARY  House {args.house}  (mean +/- std over seeds {args.seeds})\n{'=' * 85}")
    print(f"{'Variant':<10} {'MAE':>14} {'SAE':>14} {'F1':>14} {'Prec':>14} {'Rec':>14}")
    for v in variants:
        rows = [r['macro'] for r in results if r['variant'] == v]
        cells = []
        for k in ['mae', 'sae', 'f1', 'precision', 'recall']:
            vals = np.array([r[k] for r in rows])
            fmt  = '.2f' if k in ('mae', 'sae') else '.3f'
            cells.append(f"{vals.mean():{fmt}} +/- {vals.std(ddof=1) if len(vals) > 1 else 0.0:{fmt}}")
        print(f"{v:<10} " + ' '.join(f'{c:>14}' for c in cells))
