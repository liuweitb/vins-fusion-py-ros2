"""Find frames with extreme ATE per trajectory and copy the image immediately
prior to each extreme frame into results/extreme_cases/<trajectory>/.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
PER_FRAME_DIR = REPO_ROOT / "results" / "per-frame"
OUTPUT_ROOT = REPO_ROOT / "results" / "extreme_cases"
AGGREGATE_DIR = REPO_ROOT / "results" / "extreme_cases_tables"
EUROC_ROOT = Path("/home/xuguanyu/Downloads")

BASELINES = ["OPEN_VINS", "ORB_SLAM3", "ORB_SLAM3_VIO", "VINS_FUSION", "VINS_FUSION_L"]
SEQUENCES = [
    "V1_01_easy",
    "V1_02_medium",
    "V1_03_difficult",
    "V2_01_easy",
    "V2_02_medium",
    "V2_03_difficult",
]
ROW_SPECS = [
    ("mono", "max", 1),
    ("stereo", "max", 1),
    ("mono", "min", 1),
    ("stereo", "min", 1),
]

TOP_K = 5
SEQ_RE = re.compile(r"V(?P<room>[12])_0\d_(?:easy|medium|difficult)")


def euroc_image_dir(sequence: str) -> Path:
    m = SEQ_RE.search(sequence)
    if not m:
        raise ValueError(f"Cannot parse EuRoC sequence from '{sequence}'")
    room = f"vicon_room{m.group('room')}"
    return EUROC_ROOT / room / sequence / sequence / "mav0" / "cam0" / "data"


def load_image_index(image_dir: Path) -> np.ndarray:
    stems = [int(p.stem) for p in image_dir.glob("*.png")]
    if not stems:
        raise FileNotFoundError(f"No images found in {image_dir}")
    arr = np.asarray(sorted(stems), dtype=np.int64)
    return arr


def nearest_image(image_index: np.ndarray, target_ns: int) -> int:
    pos = np.searchsorted(image_index, target_ns)
    candidates = []
    if pos < len(image_index):
        candidates.append(image_index[pos])
    if pos > 0:
        candidates.append(image_index[pos - 1])
    return int(min(candidates, key=lambda v: abs(v - target_ns)))


def trajectory_name(csv_path: Path) -> str:
    return csv_path.stem


def sequence_from_trajectory(name: str) -> str:
    m = SEQ_RE.search(name)
    if not m:
        raise ValueError(f"Cannot find EuRoC sequence in trajectory '{name}'")
    return m.group(0)


def select_extremes(df: pd.DataFrame, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = df.dropna(subset=["ate"]).copy()
    valid = valid[valid["frame_index"] > 0]  # need a prior frame
    top_max = valid.nlargest(k, "ate")
    top_min = valid.nsmallest(k, "ate")
    return top_max, top_min


def copy_extremes_for_trajectory(csv_path: Path) -> None:
    name = trajectory_name(csv_path)
    sequence = sequence_from_trajectory(name)
    image_dir = euroc_image_dir(sequence)
    if not image_dir.is_dir():
        print(f"[skip] missing image dir for {name}: {image_dir}")
        return

    df = pd.read_csv(csv_path)
    if "ate" not in df.columns or "timestamp" not in df.columns:
        print(f"[skip] {csv_path} lacks required columns")
        return

    top_max, top_min = select_extremes(df, TOP_K)
    if top_max.empty or top_min.empty:
        print(f"[skip] {name}: no usable ATE rows")
        return

    image_index = load_image_index(image_dir)
    out_dir = OUTPUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    def copy_group(group: pd.DataFrame, label: str) -> None:
        ranked = group.sort_values("ate", ascending=(label == "min")).reset_index(drop=True)
        # max: largest first; min: smallest first
        if label == "max":
            ranked = group.sort_values("ate", ascending=False).reset_index(drop=True)
        for rank, row in enumerate(ranked.itertuples(index=False), start=1):
            extreme_idx = int(row.frame_index)
            prior_rows = df[df["frame_index"] == extreme_idx - 1]
            if prior_rows.empty:
                print(f"[warn] {name} {label}_{rank}: no prior frame for index {extreme_idx}")
                continue
            prior_ts_s = float(prior_rows.iloc[0]["timestamp"])
            target_ns = int(round(prior_ts_s * 1e9))
            ns = nearest_image(image_index, target_ns)
            src = image_dir / f"{ns}.png"
            dst = out_dir / f"{label}_{rank}.png"
            shutil.copyfile(src, dst)
            print(
                f"[ok] {name} {label}_{rank}: ate={row.ate:.6f} "
                f"frame={extreme_idx} prior_ts={prior_ts_s:.6f} -> {src.name}"
            )

    copy_group(top_max, "max")
    copy_group(top_min, "min")


def build_aggregate_table(sequence: str) -> None:
    n_rows = len(ROW_SPECS)
    n_cols = len(BASELINES)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.2 * n_cols, 2.4 * n_rows),
        squeeze=False,
    )

    for r, (mode, kind, rank) in enumerate(ROW_SPECS):
        for c, baseline in enumerate(BASELINES):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            img_path = OUTPUT_ROOT / f"{baseline}_{mode}_{sequence}" / f"{kind}_{rank}.png"
            if img_path.is_file():
                ax.imshow(mpimg.imread(img_path), cmap="gray")
            else:
                ax.text(
                    0.5,
                    0.5,
                    "missing",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="red",
                )
            if r == 0:
                ax.set_title(baseline, fontsize=11)
            if c == 0:
                ax.set_ylabel(f"{mode}_{kind}_ate", fontsize=11)

    fig.suptitle(sequence, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    AGGREGATE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = AGGREGATE_DIR / f"{sequence}_extremes_table.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote aggregate table for {sequence} -> {out_path}")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_paths = sorted(PER_FRAME_DIR.glob("*.csv"))
    if not csv_paths:
        raise SystemExit(f"No per-frame CSVs found in {PER_FRAME_DIR}")
    for csv_path in csv_paths:
        copy_extremes_for_trajectory(csv_path)

    for sequence in SEQUENCES:
        build_aggregate_table(sequence)


if __name__ == "__main__":
    main()
