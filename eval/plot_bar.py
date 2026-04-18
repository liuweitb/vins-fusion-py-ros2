from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Tune these directly if you want a different look.
FONT_FAMILY = "DejaVu Sans"
COLORS = {
    ("OPEN_VINS", "mono"): "#1f77b4",
    ("OPEN_VINS", "stereo"): "#6baed6",
    ("ORB_SLAM3", "mono"): "#d62728",
    ("ORB_SLAM3", "stereo"): "#ff9896",
    ("ORB_SLAM3_VIO", "mono"): "#2ca02c",
    ("ORB_SLAM3_VIO", "stereo"): "#98df8a",
}

TRAJECTORIES = [
    "V1_01_easy",
    "V1_02_medium",
    "V1_03_difficult",
    "V2_01_easy",
    "V2_02_medium",
    "V2_03_difficult",
]

BAR_ORDER = [
    ("OPEN_VINS", "mono", "OPEN_VINS mono"),
    ("OPEN_VINS", "stereo", "OPEN_VINS stereo"),
    ("ORB_SLAM3", "mono", "ORB_SLAM3 mono"),
    ("ORB_SLAM3", "stereo", "ORB_SLAM3 stereo"),
    ("ORB_SLAM3_VIO", "mono", "ORB_SLAM3_VIO mono"),
    ("ORB_SLAM3_VIO", "stereo", "ORB_SLAM3_VIO stereo"),
]

MONO_CSV = Path("results/trajectory_metrics_mono.csv")
STEREO_CSV = Path("results/trajectory_metrics_stereo.csv")


def load_metrics():
    mono = pd.read_csv(MONO_CSV)
    stereo = pd.read_csv(STEREO_CSV)
    return pd.concat([mono, stereo], ignore_index=True)


def plot_metric(data, mean_col, ylabel, title, output_path):
    plt.rcParams["font.family"] = FONT_FAMILY

    x = np.arange(len(TRAJECTORIES))
    n_bars = len(BAR_ORDER)
    width = 0.13
    offset = (n_bars - 1) / 2.0

    fig, ax = plt.subplots(figsize=(14, 5))

    for i, (algorithm, mode, label) in enumerate(BAR_ORDER):
        subset = (
            data[(data["algorithm"] == algorithm) & (data["mode"] == mode)]
            .set_index("trajectory")
            .reindex(TRAJECTORIES)
        )
        means = subset[mean_col].to_numpy(dtype=float)
        means = np.nan_to_num(means, nan=0.0)

        ax.bar(
            x + (i - offset) * width,
            means,
            width=width,
            color=COLORS[(algorithm, mode)],
            label=label,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(TRAJECTORIES, rotation=20)
    ax.set_xlabel("Trajectory")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncol=3, fontsize=9)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    return fig


def main():
    data = load_metrics()

    plot_metric(
        data,
        mean_col="ate_mean",
        ylabel="ATE (m)",
        title="Average ATE Across Trajectories",
        output_path="ate_bar.png",
    )

    plot_metric(
        data,
        mean_col="rpe_mean",
        ylabel="RPE (m)",
        title="Average RPE Across Trajectories",
        output_path="rpe_bar.png",
    )

    plt.show()


if __name__ == "__main__":
    main()
