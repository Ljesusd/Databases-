import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "src"))

from segment_gait_cycle_marker_angles import (
    segment_and_normalize_marker_angles,
    save_marker_angles,
)


TASK_REGEX = re.compile(r"P\d+_S\d+_(?P<task>[^_]+)_(?P<run>\d+)_Trajectories\.csv$")


def fix_angle_continuity(angles: np.ndarray) -> np.ndarray:
    """Unwrap and recenter angles to avoid 180/360 jumps."""
    rad = np.deg2rad(angles)
    unwrapped = np.unwrap(rad)
    deg = np.rad2deg(unwrapped)
    mean_val = float(np.mean(deg))
    shift = round(mean_val / 360.0) * 360.0
    deg = deg - shift
    if np.mean(deg) > 90:
        deg = deg - 180.0
    if np.mean(deg) < -90:
        deg = deg + 180.0
    return deg


def _mean_markers(df: pd.DataFrame, names: list[str], axis: str) -> pd.Series:
    cols = [f"{name}_{axis}" for name in names if f"{name}_{axis}" in df.columns]
    if not cols:
        raise KeyError(f"Missing columns in trajectories: {', '.join(f'{n}_{axis}' for n in names)}")
    if len(cols) == 1:
        return df[cols[0]]
    return df[cols].mean(axis=1)


def _pick_marker(df: pd.DataFrame, names: list[str], axis: str) -> pd.Series:
    for name in names:
        col = f"{name}_{axis}"
        if col in df.columns:
            return df[col]
    raise KeyError(f"Missing columns in trajectories: {', '.join(f'{n}_{axis}' for n in names)}")


def _build_reduced_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    if "time" in df.columns:
        reduced = pd.DataFrame({"time": df["time"]})
    else:
        reduced = pd.DataFrame({"time": np.arange(len(df), dtype=float)})

    for axis in ["x", "y", "z"]:
        reduced[f"PELV_{axis}"] = _mean_markers(df, ["LASI", "RASI", "LPSI", "RPSI"], axis)
        reduced[f"RTHI_{axis}"] = _pick_marker(df, ["RTHI", "RTHI1", "RTHI2", "RTRO"], axis)
        reduced[f"RKNE_{axis}"] = _mean_markers(df, ["RKNE", "RKNE1", "RKNE2"], axis)
        reduced[f"RANK_{axis}"] = _mean_markers(df, ["RANK", "RANK1", "RANK2"], axis)
        reduced[f"RHEE_{axis}"] = _pick_marker(df, ["RHEE", "RHEEL", "RHEE1"], axis)
        reduced[f"RTOE_{axis}"] = _pick_marker(df, ["RTOE", "RTOE1", "RTOE2", "RTOE3", "RTOE4"], axis)
    return reduced


def _events_path(traj_csv: Path) -> Path | None:
    base = traj_csv.stem.replace("_Trajectories", "")
    cand1 = traj_csv.with_name(f"{base}_point_gaitEvents.yaml")
    cand2 = traj_csv.with_name(f"{base}_gaitEvents.yaml")
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2
    return None


def process_trial(
    traj_csv: Path,
    events_yaml: Path,
    out_dir: Path,
    angle_mode: str,
    cycle_mode: str,
    hip_relative: bool,
    hip_absolute: bool,
    hip_sagittal: bool,
    ankle_zero_90: bool,
    hip_flip: bool,
    hip_wrap180: bool,
    hip_unwrap: bool,
    hip_center: bool,
    hip_fix_continuity: bool,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(traj_csv)
    reduced = _build_reduced_trajectories(df)

    base = traj_csv.stem.replace("_Trajectories", "")
    reduced_csv = out_dir / f"{base}_Trajectories_reduced.csv"
    reduced.to_csv(reduced_csv, index=False)

    pct, hip, knee, ankle = segment_and_normalize_marker_angles(
        str(reduced_csv),
        str(events_yaml),
        n_points=101,
        angle_mode=angle_mode,
        cycle_mode=cycle_mode,
        hip_relative=hip_relative,
        hip_absolute=hip_absolute,
        hip_sagittal=hip_sagittal,
        ankle_zero_90=ankle_zero_90,
    )
    if hip_fix_continuity:
        if hip_flip:
            hip = -hip
        hip = fix_angle_continuity(hip)
    else:
        if hip_flip:
            hip = -hip
        if hip_wrap180:
            hip = (hip + 180.0) % 360.0 - 180.0
        if hip_unwrap:
            hip_rad = np.deg2rad(hip)
            hip_rad = np.unwrap(hip_rad)
            hip = np.rad2deg(hip_rad)
        if hip_center:
            med = float(np.median(hip))
            if med > 100:
                hip = hip - 180.0
            elif med < -100:
                hip = hip + 180.0
    save_marker_angles(out_dir, base, pct, hip, knee, ankle)

    return {"trial": traj_csv.name, "status": "ok"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eurobench-root",
        default="data/human_gait/eurobench",
        help="Root directory with Eurobench trajectories",
    )
    parser.add_argument(
        "--processed-root",
        default="data/human_gait/processed",
        help="Output directory for processed angles",
    )
    parser.add_argument(
        "--tasks",
        default="Gait,FastGait,2minWalk",
        help="Comma-separated tasks to process",
    )
    parser.add_argument(
        "--angle-mode",
        choices=["2d", "3d"],
        default="3d",
        help="Angle computation mode",
    )
    parser.add_argument(
        "--cycle-mode",
        choices=["events", "knee_min"],
        default="events",
        help="Cycle selection for angles",
    )
    parser.add_argument("--hip-relative", action="store_true")
    parser.add_argument("--hip-absolute", action="store_true")
    parser.add_argument("--hip-sagittal", action="store_true", help="Project hip to sagittal plane")
    parser.add_argument(
        "--hip-flip",
        action="store_true",
        help="Invert hip sign after computation (flexion +, extension -).",
    )
    parser.add_argument(
        "--ankle-zero-90",
        action="store_true",
        help="Restar 90° al tobillo para que 0 sea pie neutro",
    )
    parser.add_argument(
        "--hip-wrap180",
        action="store_true",
        help="Wrap hip angles to [-180, 180] before optional unwrap.",
    )
    parser.add_argument(
        "--hip-unwrap",
        action="store_true",
        help="Unwrap hip angle discontinuities (radians) after wrap/flip.",
    )
    parser.add_argument(
        "--hip-fix-continuity",
        action="store_true",
        help="Apply unwrap + recenter in one step (recommended for gimbal/phase jumps).",
    )
    parser.add_argument(
        "--hip-center",
        action="store_true",
        help="Recenter hip by +/-180 if median is outside [-100, 100].",
    )
    args = parser.parse_args()

    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}
    eurobench_root = Path(args.eurobench_root)
    processed_root = Path(args.processed_root)

    log_rows = []
    for subject_dir in sorted(p for p in eurobench_root.glob("P*_S*") if p.is_dir()):
        for traj_csv in sorted(subject_dir.glob("*_Trajectories.csv")):
            match = TASK_REGEX.search(traj_csv.name)
            if not match:
                log_rows.append({"trial": traj_csv.name, "status": "skip_no_match"})
                continue
            task = match.group("task")
            if task not in tasks:
                log_rows.append({"trial": traj_csv.name, "status": "skip_task", "task": task})
                continue

            events_yaml = _events_path(traj_csv)
            if events_yaml is None:
                log_rows.append({"trial": traj_csv.name, "status": "skip_no_events", "task": task})
                continue

            out_dir = processed_root / task / subject_dir.name
            try:
                result = process_trial(
                    traj_csv,
                    events_yaml,
                    out_dir,
                    angle_mode=args.angle_mode,
                    cycle_mode=args.cycle_mode,
                    hip_relative=args.hip_relative,
                    hip_absolute=args.hip_absolute,
                    hip_sagittal=args.hip_sagittal,
                    ankle_zero_90=args.ankle_zero_90,
                    hip_flip=args.hip_flip,
                    hip_wrap180=args.hip_wrap180,
                    hip_unwrap=args.hip_unwrap,
                    hip_center=args.hip_center,
                    hip_fix_continuity=args.hip_fix_continuity,
                )
                result.update({"subject": subject_dir.name, "task": task})
                log_rows.append(result)
            except Exception as exc:
                log_rows.append({
                    "trial": traj_csv.name,
                    "status": "fail",
                    "task": task,
                    "subject": subject_dir.name,
                    "error": str(exc),
                })

    log_df = pd.DataFrame(log_rows)
    log_path = processed_root / "human_gait_processing_log.csv"
    log_df.to_csv(log_path, index=False)
    print(log_path)


if __name__ == "__main__":
    main()
