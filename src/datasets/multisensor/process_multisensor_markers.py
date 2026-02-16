import argparse
import os
from pathlib import Path
import site
import sys

import pandas as pd


def _ensure_ezc3d_dylib_or_reexec():
    if "DYLD_LIBRARY_PATH" in os.environ and "ezc3d" in os.environ["DYLD_LIBRARY_PATH"]:
        return
    if os.environ.get("EZC3D_DYLD_REEXEC") == "1":
        return
    candidates = [site.getusersitepackages()] + site.getsitepackages()
    for base in candidates:
        ez_dir = Path(base) / "ezc3d"
        if (ez_dir / "libezc3d.dylib").exists():
            current = os.environ.get("DYLD_LIBRARY_PATH", "")
            os.environ["DYLD_LIBRARY_PATH"] = f"{ez_dir}{os.pathsep}{current}" if current else str(ez_dir)
            os.environ["EZC3D_DYLD_REEXEC"] = "1"
            os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_ezc3d_dylib_or_reexec()
sys.path.append(str(Path(__file__).resolve().parents[2]))
sys.path.append(str(Path(__file__).resolve().parent))

from convert_multisensor_c3d import convert_c3d_to_marker_csv
from detect_gait_events_markers import detect_gait_events_markers
from segment_gait_cycle import segment_and_normalize, save_normalized_outputs
from segment_gait_cycle_marker_angles import (
    segment_and_normalize_marker_angles,
    save_marker_angles,
)


def _stem_base(c3d_path: Path) -> str:
    return c3d_path.stem.replace("_qtm", "")


def process_trial(
    c3d_path: Path,
    eurobench_dir: Path,
    processed_dir: Path,
    overwrite: bool,
    ankle_zero: bool,
    ankle_start_zero: bool,
    hip_zero: bool,
    hip_start_zero: bool,
    hip_relative: bool,
    hip_sagittal: bool,
    angle_mode: str,
):
    eurobench_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    base = _stem_base(c3d_path)
    traj_csv = eurobench_dir / f"{base}_Trajectories.csv"
    events_yaml = eurobench_dir / f"{base}_gaitEvents.yaml"

    if overwrite or not traj_csv.exists():
        convert_c3d_to_marker_csv(c3d_path, traj_csv)

    events_ok = False
    try:
        detect_gait_events_markers(
            str(traj_csv),
            pelvis_marker="PELV",
            heel_marker="HEE",
            toe_marker="TOE",
            axis_mode="vertical",
            axis_override="z",
            out_yaml=str(events_yaml),
        )
        events_ok = True
    except Exception as exc:
        return {
            "trial": c3d_path.name,
            "status": "fail_events",
            "error": str(exc),
        }

    if events_ok:
        try:
            pct, data = segment_and_normalize(
                str(traj_csv),
                str(events_yaml),
                landmarks=("hip", "knee", "ankle"),
                n_points=101,
            )
            save_normalized_outputs(processed_dir, base, pct, data)
        except Exception as exc:
            return {
                "trial": c3d_path.name,
                "status": "fail_traj_norm",
                "error": str(exc),
            }

    try:
        pct, hip, knee, ankle = segment_and_normalize_marker_angles(
            str(traj_csv),
            str(events_yaml),
            n_points=101,
            angle_mode=angle_mode,
            cycle_mode="knee_min",
            hip_relative=hip_relative,
            hip_sagittal=hip_sagittal,
        )
        if ankle_zero:
            ankle = ankle - 90.0
        if ankle_start_zero and ankle.size:
            ankle = ankle - ankle[0]
        if hip_zero:
            hip = hip - 90.0
        if hip_start_zero and hip.size:
            hip = hip - hip[0]
        save_marker_angles(processed_dir, base, pct, hip, knee, ankle)
    except Exception as exc:
        return {
            "trial": c3d_path.name,
            "status": "fail_angles",
            "error": str(exc),
        }

    return {
        "trial": c3d_path.name,
        "status": "ok",
        "traj_csv": str(traj_csv),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        default="data/multisensor_gait/raw/A multi-sensor human gait dataset/raw_data",
        help="Root directory with user folders containing C3D files",
    )
    parser.add_argument(
        "--eurobench-root",
        default="data/multisensor_gait/eurobench",
        help="Output directory for trajectories/events",
    )
    parser.add_argument(
        "--processed-root",
        default="data/multisensor_gait/processed",
        help="Output directory for normalized outputs",
    )
    parser.add_argument("--user", help="Single user folder name, e.g. user01")
    parser.add_argument("--all", action="store_true", help="Process all users")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs")
    parser.add_argument(
        "--ankle-zero",
        action="store_true",
        help="Subtract 90 degrees to set ankle anatomical zero",
    )
    parser.add_argument(
        "--ankle-start-zero",
        action="store_true",
        help="Shift ankle so the first normalized point is 0 degrees",
    )
    parser.add_argument(
        "--hip-zero",
        action="store_true",
        help="Subtract 90 degrees to set hip anatomical zero",
    )
    parser.add_argument(
        "--hip-start-zero",
        action="store_true",
        help="Shift hip so the first normalized point is 0 degrees",
    )
    parser.add_argument(
        "--hip-relative",
        action="store_true",
        help="Compute hip as thigh vs pelvis instead of vertical reference",
    )
    parser.add_argument(
        "--angle-mode",
        choices=["2d", "3d"],
        default="3d",
        help="Angle computation mode for flexion curves",
    )
    parser.add_argument(
        "--hip-sagittal",
        action="store_true",
        help="Project thigh to sagittal plane for hip flexion (stable ROM)",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    eurobench_root = Path(args.eurobench_root)
    processed_root = Path(args.processed_root)

    if args.all:
        user_dirs = sorted(p for p in raw_root.glob("user*") if p.is_dir())
    elif args.user:
        user_dirs = [raw_root / args.user]
    else:
        user_dirs = [raw_root / "user01"]

    log_rows = []
    for user_dir in user_dirs:
        c3d_files = sorted(user_dir.glob("*_qtm.c3d"))
        if not c3d_files:
            log_rows.append(
                {"user": user_dir.name, "trial": "", "status": "no_c3d"}
            )
            continue

        out_euro = eurobench_root / user_dir.name
        out_proc = processed_root / user_dir.name

        for c3d_path in c3d_files:
            result = process_trial(
                c3d_path,
                out_euro,
                out_proc,
                overwrite=args.overwrite,
                ankle_zero=args.ankle_zero,
                ankle_start_zero=args.ankle_start_zero,
                hip_zero=args.hip_zero,
                hip_start_zero=args.hip_start_zero,
                hip_relative=args.hip_relative,
                hip_sagittal=args.hip_sagittal,
                angle_mode=args.angle_mode,
            )
            result["user"] = user_dir.name
            log_rows.append(result)

    log_df = pd.DataFrame(log_rows)
    log_path = processed_root / "multisensor_processing_log.csv"
    log_df.to_csv(log_path, index=False)
    print(log_path)


if __name__ == "__main__":
    main()
