import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
SPLITS = [
    ("train", "House 1"),
    ("validation", "House 1"),
    ("test", "House 5"),
]
CHANNELS = ["aggregate", "dishwasher", "fridge", "microwave", "washing_machine"]
APPLIANCES = CHANNELS[1:]
COLORS = {
    "aggregate": "#1f77b4",
    "dishwasher": "#d62728",
    "fridge": "#2ca02c",
    "microwave": "#ff7f0e",
    "washing_machine": "#9467bd",
}


def load_split(name):
    path = os.path.join(DATA_DIR, f"UKDALE_HF_{name}.csv")
    return pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")


def plot_overview(dfs):
    fig, axes = plt.subplots(
        len(SPLITS),
        len(CHANNELS),
        figsize=(22, 11),
        sharex=False,
        constrained_layout=True,
    )
    fig.suptitle("UK-DALE HF Power Signals by Split", fontsize=16)

    for row, (split_name, house) in enumerate(SPLITS):
        df = dfs[split_name]
        hours = (df.index - df.index[0]).total_seconds() / 3600

        for col_idx, channel in enumerate(CHANNELS):
            ax = axes[row, col_idx]
            ax.plot(hours, df[channel], linewidth=0.55, color=COLORS[channel])
            ax.set_title(channel.replace("_", " ").title(), fontsize=10)
            ax.grid(True, alpha=0.22)
            ax.set_xlim(hours.min(), hours.max())
            if col_idx == 0:
                ax.set_ylabel(f"{split_name.title()}\n{house}\nPower (W)")
            if row == len(SPLITS) - 1:
                ax.set_xlabel("Hours from split start")

    return fig


def plot_first_hour_detail(dfs):
    fig, axes = plt.subplots(
        len(SPLITS),
        1,
        figsize=(16, 10),
        sharex=False,
        constrained_layout=True,
    )
    fig.suptitle("UK-DALE First Hour Detail: Aggregate and Appliances", fontsize=16)

    for ax, (split_name, house) in zip(axes, SPLITS):
        df = dfs[split_name].iloc[:600]
        ax.plot(df.index, df["aggregate"], color=COLORS["aggregate"], linewidth=1.1, label="aggregate")
        for app in APPLIANCES:
            ax.plot(df.index, df[app], color=COLORS[app], linewidth=0.9, alpha=0.9, label=app)

        ax.set_title(f"{split_name.title()} - {house}", loc="left", fontsize=11)
        ax.set_ylabel("Power (W)")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=5, fontsize=9, frameon=False, loc="upper right")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_xlabel("Time")

    return fig


def plot_energy_summary(dfs):
    energy = pd.DataFrame(index=[name for name, _ in SPLITS], columns=CHANNELS, dtype=float)
    for split_name, _house in SPLITS:
        df = dfs[split_name]
        # 6-second samples: watt-seconds -> Wh.
        energy.loc[split_name] = df[CHANNELS].sum() * 6 / 3600

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    energy[APPLIANCES].plot(kind="bar", ax=axes[0], color=[COLORS[c] for c in APPLIANCES])
    axes[0].set_title("Labelled Appliance Energy")
    axes[0].set_ylabel("Wh")
    axes[0].set_xlabel("Split")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].tick_params(axis="x", rotation=0)

    fractions = energy[APPLIANCES].div(energy["aggregate"], axis=0) * 100
    fractions.plot(kind="bar", stacked=True, ax=axes[1], color=[COLORS[c] for c in APPLIANCES])
    axes[1].set_title("Labelled Appliance Share of Aggregate")
    axes[1].set_ylabel("% of aggregate Wh")
    axes[1].set_xlabel("Split")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(frameon=False, fontsize=9)

    return fig


def main():
    dfs = {split_name: load_split(split_name) for split_name, _house in SPLITS}

    outputs = [
        ("ukdale_power_overview.png", plot_overview(dfs)),
        ("ukdale_first_hour_detail.png", plot_first_hour_detail(dfs)),
        ("ukdale_energy_summary.png", plot_energy_summary(dfs)),
    ]

    for filename, fig in outputs:
        path = os.path.join(DATA_DIR, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()
