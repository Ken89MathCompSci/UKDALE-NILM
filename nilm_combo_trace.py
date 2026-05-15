from itertools import product
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "dataset"
TRAIN_PATH = DATA_DIR / "UKDALE_HF_train.csv"
TARGET_PATH = DATA_DIR / "UKDALE_HF_validation.csv"
PLOT_PATH = DATA_DIR / "nilm_combo_validation_trace_py.png"
METRICS_PATH = DATA_DIR / "nilm_combo_validation_metrics_py.csv"

APPLIANCES = ["dishwasher", "fridge", "microwave", "washing_machine"]
THRESHOLDS = {
    "dishwasher": 50.0,
    "fridge": 20.0,
    "microwave": 100.0,
    "washing_machine": 50.0,
}
COLORS = {
    "aggregate": "#1f77b4",
    "reconstructed": "#202020",
    "dishwasher": "#d62728",
    "fridge": "#2ca02c",
    "microwave": "#ff7f0e",
    "washing_machine": "#9467bd",
}


def load_split(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"])


def learn_signatures(train: pd.DataFrame) -> dict[str, float]:
    signatures = {}
    for app in APPLIANCES:
        active = train.loc[train[app] > THRESHOLDS[app], app]
        signatures[app] = float(active.median()) if not active.empty else THRESHOLDS[app]
        signatures[app] = max(signatures[app], THRESHOLDS[app])

    inactive_mask = np.ones(len(train), dtype=bool)
    for app in APPLIANCES:
        inactive_mask &= train[app].to_numpy() <= THRESHOLDS[app]

    inactive_aggregate = train.loc[inactive_mask, "aggregate"]
    signatures["_base"] = float(inactive_aggregate.median()) if not inactive_aggregate.empty else 0.0
    return signatures


def make_state_space(signatures: dict[str, float]) -> pd.DataFrame:
    rows = []
    for bits in product([0, 1], repeat=len(APPLIANCES)):
        row = {"mask": 0, "labelled_power": 0.0}
        for idx, (app, bit) in enumerate(zip(APPLIANCES, bits)):
            row["mask"] |= bit << idx
            row[app] = signatures[app] if bit else 0.0
            row["labelled_power"] += row[app]
        rows.append(row)
    return pd.DataFrame(rows)


def hamming_distance(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def run_combinatorial_nilm(
    target: pd.DataFrame,
    signatures: dict[str, float],
    switch_penalty: float = 120.0,
) -> pd.DataFrame:
    states = make_state_space(signatures)
    n_steps = len(target)
    n_states = len(states)

    expected_power = signatures["_base"] + states["labelled_power"].to_numpy()
    aggregate = target["aggregate"].to_numpy(dtype=float)

    masks = states["mask"].to_numpy(dtype=int)
    transition_cost = np.zeros((n_states, n_states), dtype=float)
    for prev_idx, prev_mask in enumerate(masks):
        for next_idx, next_mask in enumerate(masks):
            transition_cost[prev_idx, next_idx] = switch_penalty * hamming_distance(
                int(prev_mask), int(next_mask)
            )

    previous_costs = np.abs(aggregate[0] - expected_power)
    back_pointers = np.full((n_steps, n_states), -1, dtype=np.int16)

    for t in range(1, n_steps):
        emission = np.abs(aggregate[t] - expected_power)
        candidate_costs = previous_costs[:, None] + transition_cost + emission[None, :]
        back_pointers[t] = np.argmin(candidate_costs, axis=0)
        previous_costs = np.min(candidate_costs, axis=0)

    path = np.zeros(n_steps, dtype=np.int16)
    path[-1] = int(np.argmin(previous_costs))
    for t in range(n_steps - 1, 0, -1):
        path[t - 1] = back_pointers[t, path[t]]

    chosen_states = states.iloc[path].reset_index(drop=True)
    trace = pd.DataFrame(
        {
            "timestamp": target["timestamp"],
            "aggregate": target["aggregate"].astype(float),
            "reconstructed": signatures["_base"] + chosen_states["labelled_power"].to_numpy(),
            "actual_labelled": target[APPLIANCES].sum(axis=1).astype(float),
            "estimated_labelled": chosen_states["labelled_power"].to_numpy(),
        }
    )

    for app in APPLIANCES:
        trace[f"actual_{app}"] = target[app].astype(float)
        trace[f"estimated_{app}"] = chosen_states[app].to_numpy()

    return trace


def compute_metrics(trace: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for app in APPLIANCES:
        actual = trace[f"actual_{app}"]
        estimated = trace[f"estimated_{app}"]
        actual_wh = actual.sum() * 6.0 / 3600.0
        estimated_wh = estimated.sum() * 6.0 / 3600.0
        rows.append(
            {
                "appliance": app,
                "mae_watts": round(float((actual - estimated).abs().mean()), 2),
                "actual_wh": round(float(actual_wh), 2),
                "estimated_wh": round(float(estimated_wh), 2),
                "energy_error_wh": round(float(estimated_wh - actual_wh), 2),
            }
        )
    return pd.DataFrame(rows)


def plot_trace(trace: pd.DataFrame, signatures: dict[str, float]) -> None:
    hours = np.arange(len(trace)) * 6.0 / 3600.0
    fig, axes = plt.subplots(
        6,
        1,
        figsize=(18, 12.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.7, 1, 1, 1, 1]},
        constrained_layout=True,
    )
    fig.suptitle("Combinatorial NILM on UK-DALE Validation Split", fontsize=16)

    axes[0].plot(hours, trace["aggregate"], color=COLORS["aggregate"], linewidth=1.0, label="aggregate")
    axes[0].plot(
        hours,
        trace["reconstructed"],
        color=COLORS["reconstructed"],
        linewidth=1.0,
        linestyle="--",
        label="reconstructed aggregate",
    )
    axes[0].set_title("Aggregate and Reconstructed Aggregate")

    axes[1].plot(
        hours,
        trace["actual_labelled"],
        color=COLORS["aggregate"],
        linewidth=1.0,
        label="actual labelled total",
    )
    axes[1].plot(
        hours,
        trace["estimated_labelled"],
        color=COLORS["reconstructed"],
        linewidth=1.0,
        linestyle="--",
        label="estimated labelled total",
    )
    axes[1].set_title("Actual vs Estimated Labelled Appliance Total")

    for ax, app in zip(axes[2:], APPLIANCES):
        ax.plot(hours, trace[f"actual_{app}"], color=COLORS[app], linewidth=1.0, label=f"actual {app}")
        ax.plot(
            hours,
            trace[f"estimated_{app}"],
            color=COLORS[app],
            linewidth=1.0,
            linestyle="--",
            label=f"estimated {app}",
        )
        ax.set_title(f"{app} actual vs estimated, learned signature {signatures[app]:.1f} W")

    for ax in axes:
        ax.set_ylabel("W")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)

    axes[-1].set_xlabel("Hours")
    axes[-1].set_xlim(0, 24)
    fig.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    train = load_split(TRAIN_PATH)
    target = load_split(TARGET_PATH)
    signatures = learn_signatures(train)
    trace = run_combinatorial_nilm(target, signatures)
    metrics = compute_metrics(trace)

    plot_trace(trace, signatures)
    metrics.to_csv(METRICS_PATH, index=False)

    print("Learned signatures:")
    print(f"  {'base':<16} {signatures['_base']:8.1f} W")
    for app in APPLIANCES:
        print(f"  {app:<16} {signatures[app]:8.1f} W")

    print("\nValidation metrics:")
    print(metrics.to_string(index=False))
    print(f"\nPlot saved -> {PLOT_PATH}")
    print(f"Metrics saved -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
