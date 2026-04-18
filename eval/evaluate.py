import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

SEQUENCES = [
    "V1_01_easy",
    "V1_02_medium",
    "V1_03_difficult",
    "V2_01_easy",
    "V2_02_medium",
    "V2_03_difficult",
]

BASELINES = [
    {
        "name": "OPEN_VINS",
        "directory": "OPEN_VINS",
        "filenames": {
            "mono": "mono_inertial.csv",
            "stereo": "stereo_inertial.csv",
        },
    },
    {
        "name": "ORB_SLAM3",
        "directory": "ORB_SLAM",
        "filenames": {
            "mono": "mono_inertial.csv",
            "stereo": "stereo_inertial.csv",
        },
    },
    {
        "name": "ORB_SLAM3_VIO",
        "directory": "ORB_SLAM",
        "filenames": {
            "mono": "mono_inertial_vio.csv",
            "stereo": "stereo_inertial_vio.csv",
        },
    },
    {
        "name": "VINS_FUSION",
        "directory": "VINS_FUSION",
        "filenames": {
            "mono": "mono_inertial.csv",
            "stereo": "stereo_inertial.csv",
        },
    },
    {
        "name": "VINS_FUSION_L",
        "directory": "VINS_FUSION",
        "filenames": {
            "mono": "mono_inertial_loop.csv",
            "stereo": "stereo_inertial_loop.csv",
        },
    },
]

TIME_SYNC_MAX_DIFF = 0.01  # seconds
RPE_DELTA = 10             # 10 frame
RPE_DELTA_UNIT = metrics.Unit.frames
RPE_ALL_PAIRS = False
RPE_REL_DELTA_TOL = 0.1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["mono", "stereo"],
        help="Trajectory mode to evaluate.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root directory containing GT/, OPEN_VINS/, and ORB_SLAM/.",
    )
    return parser.parse_args()


def read_estimate_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        header=None,
        comment="#",
        sep=r"[,\s]+",
        engine="python",
    )

    if df.empty:
        raise ValueError(f"{path} is empty.")

    first_row_probe = pd.to_numeric(
        df.iloc[0, : min(8, df.shape[1])], errors="coerce"
    )
    if first_row_probe.isna().sum() >= 4:
        df = df.iloc[1:].reset_index(drop=True)

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all").reset_index(drop=True)

    if df.shape[1] < 8:
        raise ValueError(f"{path} has {df.shape[1]} columns; expected at least 8.")

    df = df.iloc[:, :8].copy()
    df.columns = ["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"]
    return df


def read_estimate_trajectory(path: Path) -> PoseTrajectory3D:
    df = read_estimate_table(path)

    stamps = df["timestamp"].to_numpy(dtype=float)
    if np.nanmedian(np.abs(stamps)) > 1e12:
        stamps = stamps / 1e9

    xyz = df[["x", "y", "z"]].to_numpy(dtype=float)
    quat_wxyz = df[["qw", "qx", "qy", "qz"]].to_numpy(dtype=float)

    return PoseTrajectory3D(xyz, quat_wxyz, stamps)


def load_and_align_pair(gt_path: Path, est_path: Path):
    gt_traj = file_interface.read_euroc_csv_trajectory(str(gt_path))
    est_traj = read_estimate_trajectory(est_path)

    gt_sync, est_sync = sync.associate_trajectories(
        gt_traj,
        est_traj,
        max_diff=TIME_SYNC_MAX_DIFF,
        first_name=str(gt_path),
        snd_name=str(est_path),
    )

    est_aligned = copy.deepcopy(est_sync)
    est_aligned.align(gt_sync, correct_scale=False)

    return gt_sync, est_aligned


def compute_metrics(traj_ref, traj_est):
    ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
    ape_metric.process_data((traj_ref, traj_est))
    ape_stats = ape_metric.get_all_statistics()

    rpe_metric = metrics.RPE(
        metrics.PoseRelation.translation_part,
        delta=RPE_DELTA,
        delta_unit=RPE_DELTA_UNIT,
        rel_delta_tol=RPE_REL_DELTA_TOL,
        all_pairs=RPE_ALL_PAIRS,
    )
    rpe_metric.process_data((traj_ref, traj_est))
    rpe_stats = rpe_metric.get_all_statistics()

    summary = {
        "ate_mean": ape_stats["rmse"],
        "ate_std": ape_stats["std"],
        "rpe_mean": rpe_stats["rmse"],
        "rpe_std": rpe_stats["std"],
    }
    return summary, ape_metric, rpe_metric


def build_frame_dataframe(traj_ref, ape_metric, rpe_metric):
    timestamps = traj_ref.timestamps
    rows = []

    for frame_index, (timestamp, ate_value) in enumerate(
        zip(timestamps, ape_metric.error)
    ):
        rows.append(
            {
                "frame_index": frame_index,
                "timestamp": float(timestamp),
                "ate": float(ate_value),
                "rpe": np.nan,
                "rpe_from_frame_index": np.nan,
                "rpe_from_timestamp": np.nan,
            }
        )

    if RPE_DELTA_UNIT == metrics.Unit.frames and not RPE_ALL_PAIRS:
        frame_delta = int(RPE_DELTA)
        for end_index, rpe_value in zip(rpe_metric.delta_ids, rpe_metric.error):
            start_index = end_index - frame_delta
            rows[end_index]["rpe"] = float(rpe_value)
            rows[end_index]["rpe_from_frame_index"] = start_index
            rows[end_index]["rpe_from_timestamp"] = float(timestamps[start_index])

    return pd.DataFrame(
        rows,
        columns=[
            "frame_index",
            "timestamp",
            "ate",
            "rpe",
            "rpe_from_frame_index",
            "rpe_from_timestamp",
        ],
    )


def evaluate_mode(root: Path, mode: str, results_dir: Path):
    summary_rows = []
    results_dir.mkdir(parents=True, exist_ok=True)

    for seq in SEQUENCES:
        gt_path = root / "GT" / seq / "data.csv"

        if not gt_path.exists():
            print(f"[WARN] Missing GT file: {gt_path}")
            for baseline in BASELINES:
                summary_rows.append(
                    {
                        "mode": mode,
                        "trajectory": seq,
                        "algorithm": baseline["name"],
                        "ate_mean": np.nan,
                        "ate_std": np.nan,
                        "rpe_mean": np.nan,
                        "rpe_std": np.nan,
                        "status": f"missing GT: {gt_path}",
                    }
                )
            continue

        for baseline in BASELINES:
            algo_name = baseline["name"]
            est_path = (
                root
                / baseline["directory"]
                / seq
                / baseline["filenames"][mode]
            )

            if not est_path.exists():
                print(f"[WARN] Missing estimate file: {est_path}")
                summary_rows.append(
                    {
                        "mode": mode,
                        "trajectory": seq,
                        "algorithm": algo_name,
                        "ate_mean": np.nan,
                        "ate_std": np.nan,
                        "rpe_mean": np.nan,
                        "rpe_std": np.nan,
                        "status": f"missing estimate: {est_path}",
                    }
                )
                continue

            try:
                gt_sync, est_aligned = load_and_align_pair(gt_path, est_path)
                summary, ape_metric, rpe_metric = compute_metrics(gt_sync, est_aligned)

                summary_rows.append(
                    {
                        "mode": mode,
                        "trajectory": seq,
                        "algorithm": algo_name,
                        "ate_mean": summary["ate_mean"],
                        "ate_std": summary["ate_std"],
                        "rpe_mean": summary["rpe_mean"],
                        "rpe_std": summary["rpe_std"],
                        "status": "ok",
                    }
                )

                frame_df = build_frame_dataframe(gt_sync, ape_metric, rpe_metric)
                frame_path = results_dir / f"per-frame/{algo_name}_{mode}_{seq}.csv"
                frame_df.to_csv(frame_path, index=False, float_format="%.6f")
            except Exception as exc:
                print(f"[WARN] Failed on {algo_name}/{seq}/{est_path.name}: {exc}")
                summary_rows.append(
                    {
                        "mode": mode,
                        "trajectory": seq,
                        "algorithm": algo_name,
                        "ate_mean": np.nan,
                        "ate_std": np.nan,
                        "rpe_mean": np.nan,
                        "rpe_std": np.nan,
                        "status": str(exc),
                    }
                )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df[
        [
            "mode",
            "trajectory",
            "algorithm",
            "ate_mean",
            "ate_std",
            "rpe_mean",
            "rpe_std",
            "status",
        ]
    ]

    return summary_df


def load_sequence_plot_data(root: Path, mode: str, selected_baselines=None):
    sequence_plot_data = {}
    if selected_baselines is None:
        baseline_specs = BASELINES
    else:
        selected_baselines_set = set(selected_baselines)
        baseline_specs = [
            baseline
            for baseline in BASELINES
            if baseline["name"] in selected_baselines_set
        ]

    for seq in SEQUENCES:
        gt_path = root / "GT" / seq / "data.csv"
        sequence_plot_data[seq] = {}

        if not gt_path.exists():
            print(f"[WARN] Missing GT file: {gt_path}")
            continue

        for baseline in baseline_specs:
            algo_name = baseline["name"]
            est_path = (
                root
                / baseline["directory"]
                / seq
                / baseline["filenames"][mode]
            )

            if not est_path.exists():
                print(f"[WARN] Missing estimate file: {est_path}")
                continue

            try:
                gt_sync, est_aligned = load_and_align_pair(gt_path, est_path)
                sequence_plot_data[seq].setdefault("gt", gt_sync)
                sequence_plot_data[seq][algo_name] = est_aligned
            except Exception as exc:
                print(f"[WARN] Failed on {algo_name}/{seq}/{est_path.name}: {exc}")

    return sequence_plot_data


def main():
    args = parse_args()
    root = Path(args.root)
    results_dir = root / "results"
    summary_df = evaluate_mode(root, args.mode, results_dir)

    summary_path = results_dir / f"trajectory_metrics_{args.mode}.csv"
    summary_df.to_csv(summary_path, index=False, float_format="%.6f")

    print(f"\n=== {args.mode.upper()} trajectory evaluation ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved summary metrics to: {summary_path}")
    print(f"Saved per-frame metrics to: {results_dir}/per-frame/<Baseline>_{args.mode}_<Trajectory>.csv")


if __name__ == "__main__":
    main()
