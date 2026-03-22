import argparse
import os
from pathlib import Path
import site
import sys


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
try:
    from src.datasets.marker_standardization import TRAJECTORY_MARKER_STANDARDIZATION
except ModuleNotFoundError:
    from datasets.marker_standardization import TRAJECTORY_MARKER_STANDARDIZATION


def _sanitize_basename(stem: str) -> str:
    return stem.replace(" (", "_").replace(")", "")


def process_subject_trial0(
    subject_dir: Path,
    eurobench_root: Path,
    processed_root: Path,
    trial_strategy: str = "trial0",
    angle_cycle_mode: str = "events",
):
    from detect_gait_events_markers import detect_gait_events_markers
    from segment_gait_cycle import segment_and_normalize, save_normalized_outputs
    from segment_gait_cycle_angles import (
        segment_and_normalize_angles,
        save_normalized_angles,
        save_flexion_outputs,
    )
    from mh_toolbox.conversion.c3d.c3d_eurobench import convert_dir_c3d_to_eurobench
    from mh_toolbox.conversion.c3d.c3d_eurobench import (
        convert_dir_c3d_to_eurobench_using_predefined_types,
    )
    import ezc3d

    if trial_strategy == "best_range":
        c3d_file = _select_best_range_trial(subject_dir)
    else:
        c3d_files = sorted(subject_dir.glob("* (0).c3d"))
        if not c3d_files:
            raise FileNotFoundError(f"No trial (0) C3D found in {subject_dir}")
        c3d_file = c3d_files[0]
    out_eurobench = eurobench_root / subject_dir.name
    out_processed = processed_root / subject_dir.name

    convert_dir_c3d_to_eurobench(
        dir_in=str(subject_dir),
        dir_out=str(out_eurobench),
        dict_filter_cols={"Trajectories": []},
        pattern_c3d=c3d_file.name,
        pattern_subject_condition_run=r"(?P<condition>SUBJ)(?P<subject>\d+)\s*\((?P<run>\d+)\)",
        group_names=["subject", "condition", "run"],
        b_save_data=True,
        b_save_analogs=False,
        b_save_events=False,
        b_save_info=False,
        writing_mode="w",
        **TRAJECTORY_MARKER_STANDARDIZATION,
    )

    csv_path = out_eurobench / f"{c3d_file.stem}_Trajectories.csv"
    events_yaml = out_eurobench / f"{c3d_file.stem}_gaitEvents.yaml"
    detect_gait_events_markers(
        str(csv_path),
        out_yaml=str(events_yaml),
        axis_mode="vertical",
        axis_override="z",
    )

    time_norm, data_norm = segment_and_normalize(str(csv_path), str(events_yaml))

    trial_basename = _sanitize_basename(c3d_file.stem)
    save_normalized_outputs(out_processed, trial_basename, time_norm, data_norm)

    save_normalized_outputs(out_processed, subject_dir.name, time_norm, data_norm)

    convert_dir_c3d_to_eurobench_using_predefined_types(
        dir_in=str(subject_dir),
        dir_out=str(out_eurobench),
        predefined_types="ANGLE",
        list_filter_cols=[],
        pattern_c3d=c3d_file.name,
        pattern_subject_condition_run=r"(?P<condition>SUBJ)(?P<subject>\d+)\s*\((?P<run>\d+)\)",
        group_names=["subject", "condition", "run"],
        b_save_data=True,
        b_save_analogs=False,
        b_save_events=False,
        b_save_info=False,
        writing_mode="w",
    )

    angles_csv = out_eurobench / f"{c3d_file.stem}_jointAngles.csv"
    if angles_csv.exists():
        c3d_reader = ezc3d.c3d(str(c3d_file))
        point_scale = c3d_reader["parameters"]["POINT"]["SCALE"]["value"][0]
        angle_scale = (1.0 / point_scale) if point_scale != 0 else None
        pct_angles, data_angles = segment_and_normalize_angles(
            str(angles_csv),
            str(events_yaml),
            angle_scale=angle_scale,
            cycle_mode=angle_cycle_mode,
        )
        save_normalized_angles(out_processed, trial_basename, pct_angles, data_angles)
        save_normalized_angles(out_processed, subject_dir.name, pct_angles, data_angles)
        save_flexion_outputs(out_processed, trial_basename, pct_angles, data_angles)
        save_flexion_outputs(out_processed, subject_dir.name, pct_angles, data_angles)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        default="data/HealthyPiG/138_HealthyPiG/raw",
        help="Root directory with raw subject folders",
    )
    parser.add_argument(
        "--eurobench-root",
        default="data/HealthyPiG/138_HealthyPiG/eurobench",
        help="Output directory for Eurobench trajectories/events",
    )
    parser.add_argument(
        "--processed-root",
        default="data/HealthyPiG/138_HealthyPiG/processed",
        help="Output directory for normalized outputs",
    )
    parser.add_argument("--subject", help="Single subject folder name, e.g. SUBJ01")
    parser.add_argument("--all", action="store_true", help="Process all subjects")
    parser.add_argument(
        "--trial-strategy",
        choices=["trial0", "best_range"],
        default="trial0",
        help="Trial selection strategy",
    )
    parser.add_argument(
        "--angle-cycle-mode",
        choices=["events", "knee_min", "auto"],
        default="auto",
        help="Cycle selection for angles",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    eurobench_root = Path(args.eurobench_root)
    processed_root = Path(args.processed_root)

    if args.all:
        subject_dirs = sorted(p for p in raw_root.glob("SUBJ*") if p.is_dir())
    elif args.subject:
        subject_dirs = [raw_root / args.subject]
    else:
        subject_dirs = [raw_root / "SUBJ01"]

    for subject_dir in subject_dirs:
        try:
            process_subject_trial0(
                subject_dir,
                eurobench_root,
                processed_root,
                trial_strategy=args.trial_strategy,
                angle_cycle_mode=args.angle_cycle_mode,
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"Skipping {subject_dir.name}: {exc}")
            continue


def _select_best_range_trial(subject_dir: Path) -> Path:
    import ezc3d

    best_file = None
    best_range = -1.0
    for c3d_file in sorted(subject_dir.glob("*.c3d")):
        c3d = ezc3d.c3d(str(c3d_file))
        labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
        if "RKneeAngles" not in labels:
            continue
        idx = labels.index("RKneeAngles")
        vals = c3d["data"]["points"][0, idx, :]
        curr_range = float(vals.max() - vals.min())
        if curr_range > best_range:
            best_range = curr_range
            best_file = c3d_file
    if best_file is None:
        raise FileNotFoundError(f"No C3D with RKneeAngles found in {subject_dir}")
    return best_file


if __name__ == "__main__":
    main()
