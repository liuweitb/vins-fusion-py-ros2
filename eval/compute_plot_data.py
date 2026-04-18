"""Compute per-frame trajectory plot data (aligned positions + ATE/RPE).

Outputs one CSV per (algorithm, mode, sequence) under
``results/per-frame-pos/<algorithm>_<mode>_<sequence>.csv``
with columns:
    frame_index, timestamp, gt_x, gt_y, gt_z, est_x, est_y, est_z, ate, rpe
"""

from pathlib import Path

import numpy as np
import pandas as pd

from evaluate import (
    BASELINES,
    RPE_ALL_PAIRS,
    RPE_DELTA,
    RPE_DELTA_UNIT,
    SEQUENCES,
    compute_metrics,
    load_and_align_pair,
)
from evo.core import metrics


def build_plot_dataframe(traj_ref, traj_est, ape_metric, rpe_metric):
    timestamps = traj_ref.timestamps
    gt_xyz = traj_ref.positions_xyz
    est_xyz = traj_est.positions_xyz
    n = len(timestamps)

    ate = np.asarray(ape_metric.error, dtype=float)
    rpe = np.full(n, np.nan, dtype=float)
    if RPE_DELTA_UNIT == metrics.Unit.frames and not RPE_ALL_PAIRS:
        for end_index, rpe_value in zip(rpe_metric.delta_ids, rpe_metric.error):
            rpe[end_index] = float(rpe_value)

    df = pd.DataFrame(
        {
            "frame_index": np.arange(n, dtype=int),
            "timestamp": timestamps.astype(float),
            "gt_x": gt_xyz[:, 0],
            "gt_y": gt_xyz[:, 1],
            "gt_z": gt_xyz[:, 2],
            "est_x": est_xyz[:, 0],
            "est_y": est_xyz[:, 1],
            "est_z": est_xyz[:, 2],
            "ate": ate,
            "rpe": rpe,
        }
    )
    return df


def main():
    root = Path(".")
    out_dir = root / "results" / "per-frame-pos"
    out_dir.mkdir(parents=True, exist_ok=True)

    for mode in ("mono", "stereo"):
        for seq in SEQUENCES:
            gt_path = root / "GT" / seq / "data.csv"
            if not gt_path.exists():
                print(f"[WARN] Missing GT: {gt_path}")
                continue

            for baseline in BASELINES:
                algo = baseline["name"]
                est_path = (
                    root
                    / baseline["directory"]
                    / seq
                    / baseline["filenames"][mode]
                )
                if not est_path.exists():
                    print(f"[WARN] Missing estimate: {est_path}")
                    continue

                try:
                    gt_sync, est_aligned = load_and_align_pair(gt_path, est_path)
                    _, ape_metric, rpe_metric = compute_metrics(gt_sync, est_aligned)
                    df = build_plot_dataframe(
                        gt_sync, est_aligned, ape_metric, rpe_metric
                    )
                    out_path = out_dir / f"{algo}_{mode}_{seq}.csv"
                    df.to_csv(out_path, index=False, float_format="%.6f")
                    print(f"[OK] {out_path}")
                except Exception as exc:
                    print(f"[WARN] {algo}/{mode}/{seq}: {exc}")


if __name__ == "__main__":
    main()
