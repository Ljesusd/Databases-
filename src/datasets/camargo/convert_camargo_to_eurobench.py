import argparse
from collections import Counter
from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml

try:
    import matio
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "Camargo conversion requires the 'matio' package. "
        "Install it with: python3 -m pip install mat-io"
    ) from exc


DATASET_TAG = "camargo"
DEFAULT_FLOAT_FORMAT = "%.6f"
CORE_SENSORS = ("markers", "ik_offset", "ik", "gcLeft", "gcRight", "conditions")
OPTIONAL_SENSOR_SUFFIX = {
    "imu": "_imu.csv",
    "emg": "_emg.csv",
    "gon": "_gon.csv",
    "id": "_id.csv",
    "jp": "_jp.csv",
    "fp": "_fp.csv",
}
MARKER_ALIAS_MAP = {
    "LASI": "L_ASIS",
    "RASI": "R_ASIS",
    "LPSI": "L_PSIS",
    "RPSI": "R_PSIS",
    "LHEE": "L_Heel",
    "RHEE": "R_Heel",
    "LTOE": "L_Toe_Tip",
    "RTOE": "R_Toe_Tip",
}
PELVIS_MARKERS = ("L_ASIS", "R_ASIS", "L_PSIS", "R_PSIS")
OUTPUT_MARKER_ORDER = [
    "PELV",
    "LASI",
    "RASI",
    "LPSI",
    "RPSI",
    "SACR",
    "LTHI",
    "RTHI",
    "LKNE",
    "RKNE",
    "LTIB",
    "RTIB",
    "LANK",
    "RANK",
    "LHEE",
    "RHEE",
    "LTOE",
    "RTOE",
]


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def _flatten_value(value):
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        if value.dtype == object and value.size == 1:
            return _flatten_value(value.reshape(-1)[0])
        if value.size == 1:
            return _flatten_value(value.reshape(-1)[0])
        return [_flatten_value(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        value = value.strip()
        return value if value else None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_mat(path: Path):
    return matio.load_from_mat(path)


def _load_table(path: Path, key: str = "data") -> pd.DataFrame:
    payload = _load_mat(path)
    if key not in payload:
        raise KeyError(f"{path} does not contain '{key}'")
    table = payload[key]
    if not isinstance(table, pd.DataFrame):
        raise TypeError(f"{path} key '{key}' is not a DataFrame")
    return table.copy()


def _sample_rate_hz(time_values: np.ndarray) -> float | None:
    if time_values.size < 2:
        return None
    diffs = np.diff(time_values)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return None
    return float(1.0 / np.median(diffs))


def _mean_markers(arrays: list[np.ndarray | None]) -> np.ndarray | None:
    valid = [arr for arr in arrays if arr is not None]
    if not valid:
        return None
    stack = np.stack(valid, axis=0).astype(float, copy=False)
    finite = np.isfinite(stack)
    counts = finite.sum(axis=0)
    totals = np.where(finite, stack, 0.0).sum(axis=0)
    out = np.full(stack.shape[1:], np.nan, dtype=float)
    np.divide(totals, counts, out=out, where=counts > 0)
    return out


def _relative_time(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float, float | None]:
    if "Header" not in df.columns:
        raise KeyError("Missing 'Header' time column")
    out = df.copy()
    time_abs = pd.to_numeric(out["Header"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(time_abs)):
        raise ValueError("Invalid time values in Header column")
    t0 = float(time_abs[0])
    out["time"] = time_abs - t0
    out = out.drop(columns=["Header"])
    return out, t0, float(time_abs[-1]), _sample_rate_hz(time_abs)


def _standardize_joint_angles(df: pd.DataFrame) -> pd.DataFrame:
    work, _, _, _ = _relative_time(df)
    n = len(work)

    def num(col: str) -> pd.Series:
        if col not in work.columns:
            return pd.Series(np.zeros(n, dtype=float))
        return pd.to_numeric(work[col], errors="coerce")

    out = pd.DataFrame(
        {
            "LHipAngles_x": num("hip_flexion_l"),
            "LHipAngles_y": num("hip_adduction_l"),
            "LHipAngles_z": num("hip_rotation_l"),
            "LKneeAngles_x": num("knee_angle_l"),
            "LKneeAngles_y": pd.Series(np.zeros(n, dtype=float)),
            "LKneeAngles_z": pd.Series(np.zeros(n, dtype=float)),
            "LAnkleAngles_x": num("ankle_angle_l"),
            "LAnkleAngles_y": num("subtalar_angle_l"),
            "LAnkleAngles_z": num("mtp_angle_l"),
            "RHipAngles_x": num("hip_flexion_r"),
            "RHipAngles_y": num("hip_adduction_r"),
            "RHipAngles_z": num("hip_rotation_r"),
            "RKneeAngles_x": num("knee_angle_r"),
            "RKneeAngles_y": pd.Series(np.zeros(n, dtype=float)),
            "RKneeAngles_z": pd.Series(np.zeros(n, dtype=float)),
            "RAnkleAngles_x": num("ankle_angle_r"),
            "RAnkleAngles_y": num("subtalar_angle_r"),
            "RAnkleAngles_z": num("mtp_angle_r"),
            "L5S1Angles_x": num("lumbar_extension"),
            "L5S1Angles_y": num("lumbar_bending"),
            "L5S1Angles_z": num("lumbar_rotation"),
            "time": pd.to_numeric(work["time"], errors="coerce"),
        }
    )
    return out


def _marker_xyz(df: pd.DataFrame, marker: str) -> np.ndarray | None:
    cols = [f"{marker}_{axis}" for axis in ("x", "y", "z")]
    if not all(col in df.columns for col in cols):
        return None
    return np.column_stack([pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float) for col in cols])


def _median_component(vector: np.ndarray | None, axis: int) -> float:
    if vector is None:
        return 0.0
    return float(np.nanmedian(vector[:, axis]))


def _frame_from_markers(marker_map: dict[str, np.ndarray | None]) -> dict:
    pelvis = marker_map.get("PELV")
    heels = _mean_markers([marker_map.get("LHEE"), marker_map.get("RHEE")])
    if pelvis is None or heels is None:
        raise ValueError("Camargo frame alignment requires pelvis and heel markers.")

    vertical_vec = pelvis - heels
    vertical_axis = int(np.nanargmax(np.abs(np.nanmedian(vertical_vec, axis=0))))
    vertical_sign = 1 if _median_component(vertical_vec, vertical_axis) >= 0.0 else -1

    remaining_axes = [axis for axis in (0, 1, 2) if axis != vertical_axis]
    pelvis_width = _mean_markers(
        [
            marker_map.get("RASI") - marker_map.get("LASI") if marker_map.get("RASI") is not None and marker_map.get("LASI") is not None else None,
            marker_map.get("RPSI") - marker_map.get("LPSI") if marker_map.get("RPSI") is not None and marker_map.get("LPSI") is not None else None,
        ]
    )
    lateral_axis = max(remaining_axes, key=lambda axis: abs(_median_component(pelvis_width, axis)))
    lateral_sign = 1 if _median_component(pelvis_width, lateral_axis) >= 0.0 else -1

    forward_axis = next(axis for axis in remaining_axes if axis != lateral_axis)
    toe_minus_heel = _mean_markers(
        [
            marker_map.get("RTOE") - marker_map.get("RHEE") if marker_map.get("RTOE") is not None and marker_map.get("RHEE") is not None else None,
            marker_map.get("LTOE") - marker_map.get("LHEE") if marker_map.get("LTOE") is not None and marker_map.get("LHEE") is not None else None,
        ]
    )
    forward_sign = 1 if _median_component(toe_minus_heel, forward_axis) >= 0.0 else -1

    return {
        "order": [forward_axis, lateral_axis, vertical_axis],
        "signs": [forward_sign, lateral_sign, vertical_sign],
        "axis_labels": {"x": "forward", "y": "lateral_right", "z": "vertical_up"},
    }


def _canonicalize_marker_map(df: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict]:
    source = {marker: _marker_xyz(df, marker) for marker in sorted({col[:-2] for col in df.columns if col.endswith("_x")})}
    raw_map: dict[str, np.ndarray | None] = {
        "LASI": source.get("L_ASIS"),
        "RASI": source.get("R_ASIS"),
        "LPSI": source.get("L_PSIS"),
        "RPSI": source.get("R_PSIS"),
        "SACR": _mean_markers([source.get("L_PSIS"), source.get("R_PSIS")]),
        "PELV": _mean_markers([source.get("L_ASIS"), source.get("R_ASIS"), source.get("L_PSIS"), source.get("R_PSIS")]),
        "LTHI": _mean_markers([source.get("L_Thigh_Upper"), source.get("L_Thigh_Front"), source.get("L_Thigh_Rear")]),
        "RTHI": _mean_markers([source.get("R_Thigh_Upper"), source.get("R_Thigh_Front"), source.get("R_Thigh_Rear")]),
        "LKNE": source.get("L_Knee_Lat"),
        "RKNE": source.get("R_Knee_Lat"),
        "LTIB": _mean_markers([source.get("L_Shank_Upper"), source.get("L_Shank_Front"), source.get("L_Shank_Rear")]),
        "RTIB": _mean_markers([source.get("R_Shank_Upper"), source.get("R_Shank_Front"), source.get("R_Shank_Rear")]),
        "LANK": source.get("L_Ankle_Lat"),
        "RANK": source.get("R_Ankle_Lat"),
        "LHEE": source.get("L_Heel"),
        "RHEE": source.get("R_Heel"),
        "LTOE": source.get("L_Toe_Tip"),
        "RTOE": source.get("R_Toe_Tip"),
    }
    frame = _frame_from_markers(raw_map)
    order = frame["order"]
    signs = np.asarray(frame["signs"], dtype=float)

    canonical: dict[str, np.ndarray] = {}
    for name in OUTPUT_MARKER_ORDER:
        arr = raw_map.get(name)
        if arr is None:
            continue
        canonical[name] = arr[:, order] * signs

    frame_meta = {
        "source_axis_order": {"x": int(order[0]), "y": int(order[1]), "z": int(order[2])},
        "sign_flips": {"x": int(signs[0]), "y": int(signs[1]), "z": int(signs[2])},
        "axis_labels": frame["axis_labels"],
        "marker_notes": {
            "LKNE": "approximated from lateral knee marker",
            "RKNE": "approximated from lateral knee marker",
            "LTIB": "mean of upper/front/rear shank markers",
            "RTIB": "mean of upper/front/rear shank markers",
            "LANK": "approximated from lateral ankle marker",
            "RANK": "approximated from lateral ankle marker",
            "LTHI": "mean of upper/front/rear thigh markers",
            "RTHI": "mean of upper/front/rear thigh markers",
        },
    }
    return canonical, frame_meta


def _standardize_markers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    work, _, _, _ = _relative_time(df)
    marker_map, frame_meta = _canonicalize_marker_map(work)
    out = pd.DataFrame({"time": pd.to_numeric(work["time"], errors="coerce")})
    for name in OUTPUT_MARKER_ORDER:
        arr = marker_map.get(name)
        if arr is None:
            continue
        out[f"{name}_x"] = arr[:, 0]
        out[f"{name}_y"] = arr[:, 1]
        out[f"{name}_z"] = arr[:, 2]
    return out, frame_meta


def _event_times_from_gc(df: pd.DataFrame, column: str) -> list[float]:
    if "Header" not in df.columns or column not in df.columns:
        return []
    time_abs = pd.to_numeric(df["Header"], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    if time_abs.size == 0:
        return []
    t0 = float(time_abs[0])
    indices: list[int] = []
    if np.isfinite(values[0]) and abs(values[0]) <= 1e-9:
        indices.append(0)
    wraps = np.where(np.isfinite(values[:-1]) & np.isfinite(values[1:]) & (np.diff(values) < -50.0))[0] + 1
    indices.extend(int(i) for i in wraps.tolist())
    times = sorted({round(float(time_abs[idx] - t0), 6) for idx in indices})
    return times


def _conditions_summary(payload: dict) -> dict:
    info: dict = {}
    for key, value in payload.items():
        if key.startswith("__"):
            continue
        flat = _flatten_value(value)
        if isinstance(flat, pd.DataFrame):
            if {"Header", "Label"}.issubset(flat.columns):
                labels = flat["Label"].astype(str).tolist()
                seq: list[str] = []
                for label in labels:
                    if not seq or seq[-1] != label:
                        seq.append(label)
                info[f"{key}_labels"] = sorted(set(labels))
                info[f"{key}_label_sequence"] = seq
                info[f"{key}_n_rows"] = int(len(flat))
            elif {"Header", "Speed"}.issubset(flat.columns):
                speed = pd.to_numeric(flat["Speed"], errors="coerce")
                info[f"{key}_n_rows"] = int(len(flat))
                info[f"{key}_speed_min"] = float(speed.min()) if speed.notna().any() else None
                info[f"{key}_speed_max"] = float(speed.max()) if speed.notna().any() else None
                info[f"{key}_speed_median"] = float(speed.median()) if speed.notna().any() else None
            else:
                info[f"{key}_columns"] = [str(col) for col in flat.columns]
                info[f"{key}_n_rows"] = int(len(flat))
        else:
            info[key] = flat
    return info


def _available_modalities(sensor_files: dict[str, Path]) -> list[str]:
    return sorted(name for name, path in sensor_files.items() if path is not None and path.exists())


def _trial_output_name(subject: str, session: str, trial_stem: str) -> str:
    clean_session = re.sub(r"[^0-9A-Za-z]+", "_", session).strip("_")
    return f"{subject}_{clean_session}_{trial_stem}"


def _trial_sensor_paths(trial_stem: str, mode_dir: Path) -> dict[str, Path | None]:
    out: dict[str, Path | None] = {}
    for sensor in CORE_SENSORS + tuple(OPTIONAL_SENSOR_SUFFIX.keys()):
        candidate = mode_dir / sensor / f"{trial_stem}.mat"
        out[sensor] = candidate if candidate.exists() else None
    return out


def _write_optional_table(sensor: str, source_path: Path, out_path: Path) -> tuple[int, float | None]:
    df = _load_table(source_path)
    rel_df, _, _, rate = _relative_time(df)
    rel_df.to_csv(out_path, index=False, float_format=DEFAULT_FLOAT_FORMAT)
    return int(len(rel_df)), rate


def convert_trial(
    subject: str,
    session: str,
    mode: str,
    trial_stem: str,
    sensor_files: dict[str, Path | None],
    out_dir: Path,
    save_optional: set[str],
    overwrite: bool,
) -> dict:
    basename = _trial_output_name(subject, session, trial_stem)
    traj_csv = out_dir / f"{basename}_Trajectories.csv"
    joint_csv = out_dir / f"{basename}_jointAngles.csv"
    gait_yaml = out_dir / f"{basename}_gaitEvents.yaml"
    point_yaml = out_dir / f"{basename}_point_gaitEvents.yaml"
    info_yaml = out_dir / f"{basename}_info.yaml"

    row = {
        "subject": subject,
        "session": session,
        "mode": mode,
        "trial": trial_stem,
        "status": "ok",
        "error": "",
        "out_trajectories": str(traj_csv),
        "out_joint_angles": str(joint_csv),
        "out_gait_events": str(gait_yaml),
        "out_point_gait_events": str(point_yaml),
        "out_info": str(info_yaml),
    }

    if not overwrite and all(path.exists() for path in (traj_csv, joint_csv, gait_yaml, point_yaml, info_yaml)):
        row["status"] = "skipped_exists"
        return row

    conditions_path = sensor_files.get("conditions")
    markers_path = sensor_files.get("markers")
    joint_path = sensor_files.get("ik_offset") or sensor_files.get("ik")
    gc_left_path = sensor_files.get("gcLeft")
    gc_right_path = sensor_files.get("gcRight")
    missing = [
        name
        for name, path in {
            "conditions": conditions_path,
            "markers": markers_path,
            "joint_angles": joint_path,
            "gcLeft": gc_left_path,
            "gcRight": gc_right_path,
        }.items()
        if path is None
    ]
    if missing:
        row["status"] = "missing_core"
        row["error"] = ",".join(missing)
        return row

    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        conditions = _load_mat(conditions_path)
        conditions_info = _conditions_summary(conditions)

        marker_df = _load_table(markers_path)
        marker_out, frame_meta = _standardize_markers(marker_df)
        marker_t0 = float(pd.to_numeric(marker_df["Header"], errors="coerce").iloc[0])
        marker_t1 = float(pd.to_numeric(marker_df["Header"], errors="coerce").iloc[-1])
        marker_rate = _sample_rate_hz(pd.to_numeric(marker_df["Header"], errors="coerce").to_numpy(dtype=float))
        marker_out.to_csv(traj_csv, index=False, float_format=DEFAULT_FLOAT_FORMAT)

        raw_joint_df = _load_table(joint_path)
        joint_out = _standardize_joint_angles(raw_joint_df)
        joint_t0 = float(pd.to_numeric(raw_joint_df["Header"], errors="coerce").iloc[0])
        joint_t1 = float(pd.to_numeric(raw_joint_df["Header"], errors="coerce").iloc[-1])
        joint_rate = _sample_rate_hz(pd.to_numeric(raw_joint_df["Header"], errors="coerce").to_numpy(dtype=float))
        joint_out.to_csv(joint_csv, index=False, float_format=DEFAULT_FLOAT_FORMAT)

        gc_left_df = _load_table(gc_left_path)
        gc_right_df = _load_table(gc_right_path)
        events = {
            "l_heel_strike": _event_times_from_gc(gc_left_df, "HeelStrike"),
            "l_toe_off": _event_times_from_gc(gc_left_df, "ToeOff"),
            "r_heel_strike": _event_times_from_gc(gc_right_df, "HeelStrike"),
            "r_toe_off": _event_times_from_gc(gc_right_df, "ToeOff"),
        }
        _write_yaml(gait_yaml, events)
        _write_yaml(point_yaml, events)

        optional_outputs = {}
        optional_rates = {}
        optional_rows = {}
        for sensor in sorted(save_optional):
            source_path = sensor_files.get(sensor)
            if source_path is None:
                continue
            out_path = out_dir / f"{basename}{OPTIONAL_SENSOR_SUFFIX[sensor]}"
            n_rows, rate = _write_optional_table(sensor, source_path, out_path)
            optional_outputs[f"{sensor}_file"] = out_path.name
            optional_rates[f"{sensor}_sample_rate_hz"] = rate
            optional_rows[f"n_{sensor}_rows"] = n_rows

        marker_labels = sorted({col[:-2] for col in marker_out.columns if col.endswith("_x")})
        info_payload = {
            "dataset": DATASET_TAG,
            "subject": subject,
            "participant": subject,
            "session": session,
            "mode": mode,
            "trial": trial_stem,
            "condition": mode,
            "features": ["Trajectories", "jointAngles"] + [sensor for sensor in sorted(save_optional) if sensor_files.get(sensor)],
            "trajectory_file": traj_csv.name,
            "joint_angles_file": joint_csv.name,
            "gait_events_file": gait_yaml.name,
            "point_gait_events_file": point_yaml.name,
            "info_file": info_yaml.name,
            "source_files": {
                "conditions": str(conditions_path),
                "markers": str(markers_path),
                "joint_angles": str(joint_path),
                "gcLeft": str(gc_left_path),
                "gcRight": str(gc_right_path),
            },
            "source_time_start_s": marker_t0,
            "source_time_end_s": marker_t1,
            "source_joint_time_start_s": joint_t0,
            "source_joint_time_end_s": joint_t1,
            "trajectory_sample_rate_hz": marker_rate,
            "joint_angle_sample_rate_hz": joint_rate,
            "n_marker_frames": int(len(marker_out)),
            "n_joint_angle_frames": int(len(joint_out)),
            "n_markers": int(len(marker_labels)),
            "marker_labels": marker_labels,
            "trajectory_unit": "mm",
            "joint_angle_unit": "deg",
            "joint_angle_source": "ik_offset" if sensor_files.get("ik_offset") else "ik",
            "coordinate_frame": frame_meta,
            "available_modalities": _available_modalities(sensor_files),
            "conditions_summary": conditions_info,
        }
        info_payload.update(optional_outputs)
        info_payload.update(optional_rates)
        info_payload.update(optional_rows)
        _write_yaml(info_yaml, info_payload)

        row["n_marker_frames"] = int(len(marker_out))
        row["n_joint_angle_frames"] = int(len(joint_out))
        row["n_event_types"] = int(sum(bool(v) for v in events.values()))
        row["n_events_total"] = int(sum(len(v) for v in events.values()))
    except Exception as exc:  # noqa: BLE001
        row["status"] = "error"
        row["error"] = str(exc)
    return row


def iter_trials(raw_root: Path):
    subjects_root = raw_root / "subjects"
    for subject_dir in sorted(p for p in subjects_root.iterdir() if p.is_dir()):
        subject = subject_dir.name
        for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir() and p.name != "osimxml"):
            session = session_dir.name
            for mode_dir in sorted(p for p in session_dir.iterdir() if p.is_dir() and (p / "conditions").exists()):
                mode = mode_dir.name
                for cond_path in sorted((mode_dir / "conditions").glob("*.mat")):
                    yield subject, session, mode, cond_path.stem, _trial_sensor_paths(cond_path.stem, mode_dir)


def convert_all(
    raw_root: Path,
    out_root: Path,
    save_optional: set[str],
    overwrite: bool,
    subjects_filter: set[str] | None,
) -> pd.DataFrame:
    rows = []
    for subject, session, mode, trial_stem, sensor_files in iter_trials(raw_root):
        if subjects_filter and subject not in subjects_filter:
            continue
        out_dir = out_root / subject
        rows.append(
            convert_trial(
                subject=subject,
                session=session,
                mode=mode,
                trial_stem=trial_stem,
                sensor_files=sensor_files,
                out_dir=out_dir,
                save_optional=save_optional,
                overwrite=overwrite,
            )
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the Camargo dataset to Eurobench-style CSV/YAML files.")
    parser.add_argument("--raw-root", default="data/camargo/raw", help="Camargo raw root with archives/subjects/scripts.")
    parser.add_argument("--out-root", default="data/camargo/eurobench", help="Output Eurobench root.")
    parser.add_argument(
        "--subjects",
        default="",
        help="Comma-separated subject IDs to convert. Default: all extracted subjects.",
    )
    parser.add_argument("--save-imu", action="store_true", help="Export per-trial IMU CSV files.")
    parser.add_argument("--save-emg", action="store_true", help="Export per-trial EMG CSV files.")
    parser.add_argument("--save-gon", action="store_true", help="Export per-trial goniometer CSV files.")
    parser.add_argument("--save-id", action="store_true", help="Export per-trial inverse-dynamics CSV files.")
    parser.add_argument("--save-jp", action="store_true", help="Export per-trial joint-power CSV files.")
    parser.add_argument("--save-fp", action="store_true", help="Export per-trial force-plate CSV files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    save_optional = {
        sensor
        for sensor, enabled in {
            "imu": args.save_imu,
            "emg": args.save_emg,
            "gon": args.save_gon,
            "id": args.save_id,
            "jp": args.save_jp,
            "fp": args.save_fp,
        }.items()
        if enabled
    }
    subjects_filter = {token.strip() for token in args.subjects.split(",") if token.strip()} or None

    log_df = convert_all(
        raw_root=raw_root,
        out_root=out_root,
        save_optional=save_optional,
        overwrite=args.overwrite,
        subjects_filter=subjects_filter,
    )
    log_path = out_root / "conversion_log.csv"
    log_df.to_csv(log_path, index=False)

    status_counts = Counter(log_df["status"].fillna("unknown").tolist()) if not log_df.empty else Counter()
    summary = {
        "dataset": DATASET_TAG,
        "raw_root": str(raw_root),
        "out_root": str(out_root),
        "subjects_filter": sorted(subjects_filter) if subjects_filter else None,
        "saved_optional_modalities": sorted(save_optional),
        "n_trials_seen": int(len(log_df)),
        "n_subjects_seen": int(log_df["subject"].nunique()) if not log_df.empty else 0,
        "status_counts": dict(status_counts),
        "mode_counts": {str(k): int(v) for k, v in log_df["mode"].value_counts().sort_index().items()} if not log_df.empty else {},
        "log_file": str(log_path),
    }
    summary_path = out_root / "conversion_summary.yaml"
    _write_yaml(summary_path, summary)

    print(log_path)
    print(summary_path)
    print(yaml.safe_dump(summary, sort_keys=False, allow_unicode=False))


if __name__ == "__main__":
    main()
