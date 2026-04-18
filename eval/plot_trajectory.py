"""Plot per-frame trajectory heatmaps (ATE/RPE) for one EuRoC sequence.

Usage:
    python plot_trajectory.py v101

The short name is mapped to the full EuRoC sequence name (e.g. v101 -> V1_01_easy).
Two PNGs are saved to ``results/figures/``:
    <sequence>_ate_heatmap.png
    <sequence>_rpe_heatmap.png

Each figure is a 2x3 grid of 3D subplots:
    open_vins_mono   orb_slam_vio_mono   vins_fusion_mono
    open_vins_stereo orb_slam_vio_stereo vins_fusion_stereo

Ground truth is plotted in grey; the estimated trajectory is colored by the
per-frame ATE or RPE with a dedicated colorbar for each subplot.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: F401 (registers 3D)


SHORT_TO_SEQUENCE = {
    "v101": "V1_01_easy",
    "v102": "V1_02_medium",
    "v103": "V1_03_difficult",
    "v201": "V2_01_easy",
    "v202": "V2_02_medium",
    "v203": "V2_03_difficult",
}


SUBPLOT_GRID = [
    [
        ("OPEN_VINS", "mono", "OPEN_VINS mono"),
        ("ORB_SLAM3_VIO", "mono", "ORB_SLAM3_VIO mono"),
        ("VINS_FUSION", "mono", "VINS_FUSION mono"),
    ],
    [
        ("OPEN_VINS", "stereo", "OPEN_VINS stereo"),
        ("ORB_SLAM3_VIO", "stereo", "ORB_SLAM3_VIO stereo"),
        ("VINS_FUSION", "stereo", "VINS_FUSION stereo"),
    ],
]

DATA_DIR = Path("results/per-frame-pos")
GT_DIR = Path("GT")
OUT_DIR = Path("results/figures")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sequence",
        help="Short sequence name, e.g. v101 (V1_01_easy), v203 (V2_03_difficult).",
    )
    return parser.parse_args()


def resolve_sequence(short_name: str) -> str:
    key = short_name.lower()
    if key not in SHORT_TO_SEQUENCE:
        valid = ", ".join(sorted(SHORT_TO_SEQUENCE))
        sys.exit(f"Unknown sequence '{short_name}'. Valid options: {valid}")
    return SHORT_TO_SEQUENCE[key]


def load_frame_data(algorithm: str, mode: str, sequence: str):
    path = DATA_DIR / f"{algorithm}_{mode}_{sequence}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_full_gt(sequence: str):
    """Load the full EuRoC ground truth trajectory for a sequence.

    The per-frame CSVs hold a GT that has been time-synced (and thus
    subsampled) to each algorithm's estimate timestamps, so the GT plot
    differs between subplots whenever algorithms have different matched
    frame counts. Loading the raw EuRoC GT here guarantees a single,
    full-resolution reference curve shared across all subplots.
    """
    path = GT_DIR / sequence / "data.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, comment="#", header=None)
    if df.shape[1] < 4:
        return None
    return df.iloc[:, 1:4].to_numpy(dtype=float)


def draw_colored_trajectory(ax, xyz, values, cmap, vmin, vmax):
    """Draw a 3D polyline colored per-segment by `values` (length N)."""
    if len(xyz) < 2:
        return None

    segments = np.stack([xyz[:-1], xyz[1:]], axis=1)
    seg_values = 0.5 * (values[:-1] + values[1:])

    lc = Line3DCollection(segments, cmap=cmap, linewidth=2)
    lc.set_array(seg_values)
    lc.set_clim(vmin, vmax)
    ax.add_collection3d(lc)
    return lc


def set_equal_3d_limits(ax, xyz):
    mins = np.nanmin(xyz, axis=0)
    maxs = np.nanmax(xyz, axis=0)
    centers = (mins + maxs) / 2.0
    half = np.nanmax(maxs - mins) / 2.0
    if not np.isfinite(half) or half <= 0:
        half = 1.0
    pad = half * 1.05
    ax.set_xlim(centers[0] - pad, centers[0] + pad)
    ax.set_ylim(centers[1] - pad, centers[1] + pad)
    ax.set_zlim(centers[2] - pad, centers[2] + pad)


def plot_metric_figure(sequence: str, metric: str, out_path: Path):
    metric_label = metric.upper()
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        f"{sequence} — {metric_label} heatmap (ground truth in grey)",
        fontsize=14,
    )

    cmap = plt.get_cmap("viridis")
    full_gt_xyz = load_full_gt(sequence)

    for row_idx, row in enumerate(SUBPLOT_GRID):
        for col_idx, (algorithm, mode, label) in enumerate(row):
            ax_index = row_idx * 3 + col_idx + 1
            ax = fig.add_subplot(2, 3, ax_index, projection="3d")
            ax.set_title(label, fontsize=11)
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_zlabel("Z [m]")

            df = load_frame_data(algorithm, mode, sequence)
            if df is None or df.empty:
                ax.text2D(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12,
                    color="gray",
                )
                continue

            est_xyz = df[["est_x", "est_y", "est_z"]].to_numpy(dtype=float)
            values = df[metric].to_numpy(dtype=float)

            mask = np.isfinite(values)
            est_valid = est_xyz[mask]
            values_valid = values[mask]

            gt_xyz = (
                full_gt_xyz
                if full_gt_xyz is not None
                else df[["gt_x", "gt_y", "gt_z"]].to_numpy(dtype=float)
            )

            ax.plot(
                gt_xyz[:, 0],
                gt_xyz[:, 1],
                gt_xyz[:, 2],
                color="0.55",
                linewidth=1.5,
                label="Ground truth",
            )

            if len(est_valid) >= 2 and np.any(np.isfinite(values_valid)):
                vmin, vmax = np.nanquantile(values_valid, [0.05, 0.95])
                vmin = float(vmin)
                vmax = float(vmax)
                if vmax <= vmin:
                    vmax = vmin + 1e-9
                lc = draw_colored_trajectory(
                    ax, est_valid, values_valid, cmap, vmin, vmax
                )
                if lc is not None:
                    cbar = fig.colorbar(
                        lc, ax=ax, shrink=0.7, pad=0.1, fraction=0.04,
                        extend="both",
                    )
                    cbar.set_label(f"{metric_label} [m]", fontsize=9)

            all_xyz = np.vstack([gt_xyz, est_xyz])
            set_equal_3d_limits(ax, all_xyz)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


def main():
    args = parse_args()
    sequence = resolve_sequence(args.sequence)

    plot_metric_figure(
        sequence, "ate", OUT_DIR / f"{sequence}_ate_heatmap.png"
    )
    plot_metric_figure(
        sequence, "rpe", OUT_DIR / f"{sequence}_rpe_heatmap.png"
    )


if __name__ == "__main__":
    main()
