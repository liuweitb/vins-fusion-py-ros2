import sys
import glob
import os

def convert_files(base_dir, skip_header, filename_patterns):
    """Convert space-separated txt files to comma-separated csv files."""
    txt_files = []
    for pattern in filename_patterns:
        txt_files.extend(glob.glob(os.path.join(base_dir, f"*/{pattern}")))

    for txt_path in sorted(txt_files):
        csv_path = txt_path.replace(".txt", ".csv")
        with open(txt_path) as f:
            lines = f.readlines()
        if skip_header and lines and lines[0].startswith("#"):
            lines = lines[1:]
        with open(csv_path, "w") as f:
            for line in lines:
                f.write(",".join(line.split()) + "\n")
        print(f"Converted: {csv_path}")


def reorder_fusion_csvs(base_dir, filename_patterns):
    """Reorder VINS-Fusion csv columns in place and drop velocity columns:
    'timestamp x y z qw qx qy qz vx vy vz' -> 'timestamp x y z qx qy qz qw'."""
    csv_files = []
    for pattern in filename_patterns:
        csv_files.extend(glob.glob(os.path.join(base_dir, f"*/{pattern}")))

    for csv_path in sorted(csv_files):
        with open(csv_path) as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            stripped = line.rstrip("\r\n")
            if not stripped.strip():
                new_lines.append(line)
                continue
            parts = [p for p in stripped.split(",") if p != ""]
            if len(parts) >= 8:
                ts, x, y, z, qw, qx, qy, qz = parts[:8]
                reordered = [ts, x, y, z, qx, qy, qz, qw]
                new_lines.append(",".join(reordered) + "\n")
            else:
                new_lines.append(line)

        with open(csv_path, "w") as f:
            f.writelines(new_lines)
        print(f"Reordered: {csv_path}")


def reorder_fusion_loop_csvs(base_dir, filename_patterns):
    """Reorder VINS-Fusion loop csv columns and convert timestamp ns->ms in place:
    'timestamp_ns x y z qw qx qy qz' -> 'timestamp_ms x y z qx qy qz qw'."""
    csv_files = []
    for pattern in filename_patterns:
        csv_files.extend(glob.glob(os.path.join(base_dir, f"*/{pattern}")))

    for csv_path in sorted(csv_files):
        with open(csv_path) as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            stripped = line.rstrip("\r\n")
            if not stripped.strip():
                new_lines.append(line)
                continue
            parts = [p for p in stripped.split(",") if p != ""]
            if len(parts) >= 8:
                ts_ns, x, y, z, qw, qx, qy, qz = parts[:8]
                ts_ms = f"{int(ts_ns) / 1e6:.6f}"
                reordered = [ts_ms, x, y, z, qx, qy, qz, qw]
                new_lines.append(",".join(reordered) + "\n")
            else:
                new_lines.append(line)

        with open(csv_path, "w") as f:
            f.writelines(new_lines)
        print(f"Reordered: {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("orb", "vins", "fusion"):
        print("Usage: python correct_format.py [orb|vins|fusion]")
        sys.exit(1)

    mode = sys.argv[1]
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if mode == "orb":
        convert_files(
            os.path.join(script_dir, "ORB_SLAM"),
            skip_header=False,
            filename_patterns=[
                "mono_inertial.txt",
                "stereo_inertial.txt",
                "mono_inertial_vio.txt",
                "stereo_inertial_vio.txt",
            ],
        )
    elif mode == "vins":
        convert_files(
            os.path.join(script_dir, "OPEN_VINS"),
            skip_header=True,
            filename_patterns=["mono_inertial.txt", "stereo_inertial.txt"],
        )
    else:
        fusion_dir = os.path.join(script_dir, "VINS_FUSION")
        reorder_fusion_csvs(
            fusion_dir,
            filename_patterns=[
                "mono_inertial.csv",
                "stereo_inertial.csv",
            ],
        )
        reorder_fusion_loop_csvs(
            fusion_dir,
            filename_patterns=[
                "mono_inertial_loop.csv",
                "stereo_inertial_loop.csv",
            ],
        )
