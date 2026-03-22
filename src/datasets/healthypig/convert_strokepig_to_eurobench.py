import argparse
import os
from pathlib import Path
import re
import site
import sys
import tempfile

import pandas as pd
import yaml


def _ensure_ezc3d_dylib_or_reexec() -> None:
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

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from detect_gait_events_markers import detect_gait_events_markers  # noqa: E402
from mh_toolbox.conversion.c3d.c3d_eurobench import (  # noqa: E402
    convert_dir_c3d_to_eurobench_using_predefined_types,
)
try:
    from src.datasets.build_canonical_gait_profiles import _trajectory_process_group  # noqa: E402
    from src.datasets.marker_standardization import TRAJECTORY_MARKER_STANDARDIZATION  # noqa: E402
except ModuleNotFoundError:
    from datasets.build_canonical_gait_profiles import _trajectory_process_group  # noqa: E402
    from datasets.marker_standardization import TRAJECTORY_MARKER_STANDARDIZATION  # noqa: E402


DATASET_ID = "healthypig_stroke"
TRIAL_PATTERN = r"(?P<subject>TVC\d+)/(?P<condition>gait)/(?P<run>[^/]+)\.c3d$"
TRIAL_GROUPS = ["subject", "condition", "run"]


def _subject_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.name)
    number = int(match.group(1)) if match else 10**9
    return number, path.name


def _trial_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)", path.stem)
    number = int(match.group(1)) if match else 10**9
    return number, path.stem.lower()


def _iter_subject_dirs(raw_root: Path) -> list[Path]:
    return sorted(
        [path for path in raw_root.glob("TVC*") if path.is_dir()],
        key=_subject_sort_key,
    )


def _iter_dynamic_trials(subject_dir: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in subject_dir.glob("*.c3d")
            if path.stem.lower().startswith("bwa")
        ],
        key=_trial_sort_key,
    )


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML payload in {path}")
    return payload


def _marker_names_from_trajectory(traj_path: Path) -> set[str]:
    df = pd.read_csv(traj_path, nrows=1)
    return {
        col[:-2]
        for col in df.columns
        if col.endswith(("_x", "_y", "_z"))
    }


def _harmonize_trajectory_csv(
    src_path: Path,
    dst_path: Path,
    common_markers: list[str],
) -> None:
    df = pd.read_csv(src_path)
    keep_cols = ["time"]
    for marker in common_markers:
        for axis in ["x", "y", "z"]:
            col = f"{marker}_{axis}"
            if col in df.columns:
                keep_cols.append(col)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    df.loc[:, keep_cols].to_csv(dst_path, index=False)


def _valid_event_payload(payload: dict, heel_key: str) -> bool:
    heel_values = payload.get(heel_key, []) or []
    return len(heel_values) >= 2


def _select_gait_events_payload(traj_path: Path, point_events_path: Path | None = None) -> tuple[dict, str]:
    attempts = [
        {"side": "R", "axis_mode": "vertical", "axis_override": "z"},
        {"side": "L", "axis_mode": "vertical", "axis_override": "z"},
        {"side": "R", "axis_mode": "auto", "axis_override": None},
        {"side": "L", "axis_mode": "auto", "axis_override": None},
    ]
    errors: list[str] = []

    for attempt in attempts:
        side = attempt["side"]
        heel_key = f"{side.lower()}_heel_strike"
        toe_key = f"{side.lower()}_toe_off"
        try:
            events, axis = detect_gait_events_markers(
                str(traj_path),
                side=side,
                axis_mode=attempt["axis_mode"],
                axis_override=attempt["axis_override"],
            )
        except Exception as exc:
            errors.append(f"{side}:{attempt['axis_mode']}:{exc}")
            continue

        if not _valid_event_payload(events, heel_key):
            errors.append(f"{side}:{attempt['axis_mode']}:insufficient_heel_strikes")
            continue

        payload = dict(events)
        if heel_key != "r_heel_strike":
            payload["r_heel_strike"] = list(payload.get(heel_key, []))
            payload["r_toe_off"] = list(payload.get(toe_key, []))
        return payload, f"marker_{side.lower()}_{axis}"

    if point_events_path is not None and point_events_path.exists():
        payload = _load_yaml(point_events_path)
        if _valid_event_payload(payload, "r_heel_strike"):
            return payload, "point_events_r"
        if _valid_event_payload(payload, "l_heel_strike"):
            payload["r_heel_strike"] = list(payload.get("l_heel_strike", []))
            payload["r_toe_off"] = list(payload.get("l_toe_off", []))
            return payload, "point_events_l_aliased"

    error_msg = "; ".join(errors) if errors else "no_event_source_available"
    raise RuntimeError(f"Could not derive gait events for {traj_path.name}: {error_msg}")


def _build_trial_manifest(eurobench_root: Path, manifest_path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for subject_dir in _iter_subject_dirs(eurobench_root):
        for traj_path in sorted(subject_dir.glob("*_Trajectories.csv"), key=_trial_sort_key):
            trial_base = traj_path.stem.replace("_Trajectories", "")
            rows.append(
                {
                    "dataset": DATASET_ID,
                    "subject": subject_dir.name,
                    "condition": "",
                    "trial_id": trial_base,
                    "signal_kind": "trajectories",
                    "signal_csv": str(traj_path),
                    "trajectories_csv": str(traj_path),
                    "joint_angles_csv": str(subject_dir / f"{trial_base}_jointAngles.csv")
                    if (subject_dir / f"{trial_base}_jointAngles.csv").exists()
                    else "",
                    "gait_events_yaml": str(subject_dir / f"{trial_base}_gaitEvents.yaml")
                    if (subject_dir / f"{trial_base}_gaitEvents.yaml").exists()
                    else "",
                    "point_gait_events_yaml": str(subject_dir / f"{trial_base}_point_events.yaml")
                    if (subject_dir / f"{trial_base}_point_events.yaml").exists()
                    else "",
                    "info_yaml": str(subject_dir / f"{trial_base}_info.yaml")
                    if (subject_dir / f"{trial_base}_info.yaml").exists()
                    else "",
                }
            )
    df = pd.DataFrame(rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    return df


def _build_subject_manifest(
    canonical_rows: list[dict],
    subject_dirs: list[Path],
    manifest_path: Path,
) -> pd.DataFrame:
    result_by_subject = {row["subject"]: row for row in canonical_rows}
    rows: list[dict] = []
    for subject_dir in subject_dirs:
        subject = subject_dir.name
        result = result_by_subject.get(subject)
        if result is None:
            rows.append(
                {
                    "dataset": DATASET_ID,
                    "subject": subject,
                    "condition": "",
                    "group_id": subject,
                    "angles_csv": "",
                    "trajectories_csv": "",
                    "summary_yaml": "",
                    "plot_png": "",
                    "status": "missing",
                    "n_trials": 0,
                    "candidate_cycles": 0,
                    "selected_cycles": 0,
                }
            )
            continue

        rows.append(
            {
                "dataset": DATASET_ID,
                "subject": subject,
                "condition": "",
                "group_id": subject,
                "angles_csv": result.get("angles_csv", ""),
                "trajectories_csv": result.get("trajectories_csv", ""),
                "summary_yaml": result.get("summary_yaml", ""),
                "plot_png": result.get("plot_png", ""),
                "status": result.get("status", ""),
                "n_trials": int(result.get("n_trials", 0)),
                "candidate_cycles": int(result.get("candidate_cycles", 0)),
                "selected_cycles": int(result.get("selected_cycles", 0)),
            }
        )

    df = pd.DataFrame(rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    return df


def _convert_trial(
    trial_path: Path,
    stage_dir: Path,
    out_dir: Path,
) -> dict:
    stage_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_file = stage_dir / trial_path.name
    if stage_file.exists() or stage_file.is_symlink():
        stage_file.unlink()
    stage_file.symlink_to(trial_path.resolve())

    trial_base = trial_path.stem
    row = {
        "dataset": DATASET_ID,
        "subject": trial_path.parent.name,
        "trial": trial_base,
        "status": "ok",
        "error": "",
        "source_c3d": str(trial_path),
        "trajectories_csv": "",
        "joint_angles_csv": "",
        "point_events_yaml": "",
        "gait_events_yaml": "",
        "info_yaml": "",
        "event_source": "",
    }

    try:
        convert_dir_c3d_to_eurobench_using_predefined_types(
            dir_in=str(stage_dir),
            dir_out=str(out_dir),
            predefined_types="TRAJECTORY",
            b_save_data=True,
            b_save_analogs=False,
            b_save_events=True,
            b_save_info=True,
            pattern_c3d=trial_path.name,
            pattern_subject_condition_run=TRIAL_PATTERN,
            group_names=TRIAL_GROUPS,
            events_suffix="events",
            info_suffix="info",
            writing_mode="w",
            **TRAJECTORY_MARKER_STANDARDIZATION,
        )
        convert_dir_c3d_to_eurobench_using_predefined_types(
            dir_in=str(stage_dir),
            dir_out=str(out_dir),
            predefined_types="ANGLE",
            b_save_data=True,
            b_save_analogs=False,
            b_save_events=False,
            b_save_info=False,
            pattern_c3d=trial_path.name,
            pattern_subject_condition_run=TRIAL_PATTERN,
            group_names=TRIAL_GROUPS,
            info_suffix="info",
            writing_mode="w",
        )

        trajectories_csv = out_dir / f"{trial_base}_Trajectories.csv"
        joint_angles_csv = out_dir / f"{trial_base}_jointAngles.csv"
        point_events_yaml = out_dir / f"{trial_base}_point_events.yaml"
        gait_events_yaml = out_dir / f"{trial_base}_gaitEvents.yaml"
        info_yaml = out_dir / f"{trial_base}_info.yaml"

        payload, event_source = _select_gait_events_payload(
            trajectories_csv,
            point_events_path=point_events_yaml if point_events_yaml.exists() else None,
        )
        _write_yaml(gait_events_yaml, payload)

        row["trajectories_csv"] = str(trajectories_csv) if trajectories_csv.exists() else ""
        row["joint_angles_csv"] = str(joint_angles_csv) if joint_angles_csv.exists() else ""
        row["point_events_yaml"] = str(point_events_yaml) if point_events_yaml.exists() else ""
        row["gait_events_yaml"] = str(gait_events_yaml)
        row["info_yaml"] = str(info_yaml) if info_yaml.exists() else ""
        row["event_source"] = event_source
    except Exception as exc:
        row["status"] = "error"
        row["error"] = str(exc)

    return row


def _canonicalize_subject(
    subject: str,
    eurobench_root: Path,
    processed_root: Path,
    plots_root: Path,
    keep_percentile: float,
    n_points: int,
    min_stride_s: float,
    max_stride_s: float,
) -> dict:
    subject_dir = eurobench_root / subject
    triplets: list[tuple[Path, Path]] = []
    for traj_path in sorted(subject_dir.glob("*_Trajectories.csv"), key=_trial_sort_key):
        trial_base = traj_path.stem.replace("_Trajectories", "")
        gait_path = subject_dir / f"{trial_base}_gaitEvents.yaml"
        if gait_path.exists():
            triplets.append((traj_path, gait_path))

    row = {
        "dataset": DATASET_ID,
        "subject": subject,
        "status": "ok",
        "error": "",
        "n_trials": len(triplets),
        "candidate_cycles": 0,
        "selected_cycles": 0,
        "angles_csv": "",
        "trajectories_csv": "",
        "summary_yaml": "",
        "plot_png": "",
    }

    if not triplets:
        row["status"] = "no_trials"
        return row

    try:
        marker_sets = [_marker_names_from_trajectory(traj_path) for traj_path, _ in triplets]
        common_markers = sorted(set.intersection(*marker_sets)) if marker_sets else []
        if not common_markers:
            raise RuntimeError(f"No common marker set available for {subject}")

        tmp_root = REPO_ROOT / "tmp"
        tmp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"{subject}_canonical_", dir=str(tmp_root)) as tmp_name:
            tmp_dir = Path(tmp_name)
            harmonized_triplets: list[tuple[Path, Path]] = []
            for traj_path, gait_path in triplets:
                harmonized_traj = tmp_dir / traj_path.name
                _harmonize_trajectory_csv(
                    src_path=traj_path,
                    dst_path=harmonized_traj,
                    common_markers=common_markers,
                )
                harmonized_triplets.append((harmonized_traj, gait_path))

            result = _trajectory_process_group(
                dataset_name=DATASET_ID,
                group_label=subject,
                basename=subject,
                triplets=harmonized_triplets,
                out_dir=processed_root / subject,
                plots_dir=plots_root / subject,
                n_points=n_points,
                min_stride_s=min_stride_s,
                max_stride_s=max_stride_s,
                keep_percentile=keep_percentile,
                require_toe_off=False,
            )
        summary_yaml = Path(result["summary_yaml"])
        summary_payload = _load_yaml(summary_yaml)
        row["candidate_cycles"] = int(summary_payload.get("candidate_cycles", 0))
        row["selected_cycles"] = int(summary_payload.get("selected_cycles", 0))
        row["angles_csv"] = str(result.get("angles_csv", ""))
        row["trajectories_csv"] = str(result.get("trajectories_csv", ""))
        row["summary_yaml"] = str(summary_yaml)
        row["plot_png"] = str(plots_root / subject / f"{subject}_canonical_profiles.png")
    except Exception as exc:
        row["status"] = "error"
        row["error"] = str(exc)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert 50_StrokePiG C3D files to Eurobench and canonical profiles.")
    parser.add_argument(
        "--raw-root",
        default="data/HealthyPiG/50_StrokePiG/raw/50_StrokePiG",
        help="Root with extracted TVC subject folders.",
    )
    parser.add_argument(
        "--eurobench-root",
        default="data/HealthyPiG/50_StrokePiG/eurobench",
        help="Output Eurobench root.",
    )
    parser.add_argument(
        "--processed-root",
        default="data/HealthyPiG/50_StrokePiG/processed_canonical",
        help="Output canonical profiles root.",
    )
    parser.add_argument(
        "--plots-root",
        default="data/HealthyPiG/50_StrokePiG/plots_canonical",
        help="Output canonical plots root.",
    )
    parser.add_argument(
        "--trial-manifest",
        default="data/HealthyPiG/50_StrokePiG/trial_manifest.csv",
        help="Trial manifest CSV path.",
    )
    parser.add_argument(
        "--subject-manifest",
        default="data/HealthyPiG/50_StrokePiG/subject_manifest.csv",
        help="Subject manifest CSV path.",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Process a single subject, e.g. TVC03.",
    )
    parser.add_argument(
        "--keep-percentile",
        type=float,
        default=25.0,
        help="Percentile threshold for canonical cycle selection.",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=101,
        help="Number of normalized gait-cycle points.",
    )
    parser.add_argument(
        "--min-stride-s",
        type=float,
        default=0.6,
        help="Minimum stride duration for canonical cycle extraction.",
    )
    parser.add_argument(
        "--max-stride-s",
        type=float,
        default=3.0,
        help="Maximum stride duration for canonical cycle extraction.",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    eurobench_root = Path(args.eurobench_root)
    processed_root = Path(args.processed_root)
    plots_root = Path(args.plots_root)
    trial_manifest_path = Path(args.trial_manifest)
    subject_manifest_path = Path(args.subject_manifest)

    if args.subject:
        subject_dirs = [raw_root / args.subject]
    else:
        subject_dirs = _iter_subject_dirs(raw_root)

    eurobench_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)
    tmp_root = REPO_ROOT / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    trial_rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="strokepig_stage_", dir=str(tmp_root)) as tmp_name:
        stage_root = Path(tmp_name)
        for subject_dir in subject_dirs:
            dynamic_trials = _iter_dynamic_trials(subject_dir)
            out_dir = eurobench_root / subject_dir.name
            stage_dir = stage_root / subject_dir.name / "gait"
            for trial_path in dynamic_trials:
                trial_rows.append(
                    _convert_trial(
                        trial_path=trial_path,
                        stage_dir=stage_dir,
                        out_dir=out_dir,
                    )
                )

    conversion_log_path = eurobench_root / "conversion_log.csv"
    pd.DataFrame(trial_rows).to_csv(conversion_log_path, index=False)

    canonical_rows: list[dict] = []
    for subject_dir in subject_dirs:
        canonical_rows.append(
            _canonicalize_subject(
                subject=subject_dir.name,
                eurobench_root=eurobench_root,
                processed_root=processed_root,
                plots_root=plots_root,
                keep_percentile=args.keep_percentile,
                n_points=args.n_points,
                min_stride_s=args.min_stride_s,
                max_stride_s=args.max_stride_s,
            )
        )

    canonical_log_path = processed_root / "healthypig_stroke_canonical_subjects_summary.csv"
    pd.DataFrame(canonical_rows).to_csv(canonical_log_path, index=False)

    trial_manifest = _build_trial_manifest(eurobench_root=eurobench_root, manifest_path=trial_manifest_path)
    subject_manifest = _build_subject_manifest(
        canonical_rows=canonical_rows,
        subject_dirs=subject_dirs,
        manifest_path=subject_manifest_path,
    )

    summary = {
        "dataset": DATASET_ID,
        "n_subjects_requested": len(subject_dirs),
        "n_trials_attempted": len(trial_rows),
        "n_trials_ok": int(sum(row["status"] == "ok" for row in trial_rows)),
        "n_subjects_ok": int(sum(row["status"] == "ok" for row in canonical_rows)),
        "conversion_log": str(conversion_log_path),
        "canonical_log": str(canonical_log_path),
        "trial_manifest": str(trial_manifest_path),
        "subject_manifest": str(subject_manifest_path),
        "trial_manifest_rows": int(len(trial_manifest)),
        "subject_manifest_rows": int(len(subject_manifest)),
    }
    _write_yaml(processed_root / "healthypig_stroke_conversion_summary.yaml", summary)
    print(processed_root / "healthypig_stroke_conversion_summary.yaml")


if __name__ == "__main__":
    main()
