#!/usr/bin/env python3
"""Convert the Multimodal video and IMU kinematic dataset to a canonical EUROBENCH-style layout."""

from __future__ import annotations

import argparse
import csv
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

import yaml


DATASET_ROOT = Path("data/Multimodal video and IMU kinematic")
DATASET_TAG = "Multimodal video and IMU kinematic"
CSV_SAMPLE_RATE = 50.0
PROCESSED_DELIMITER = ";"
WALK_CONDITIONS = {
    "A01": "walk_forward",
    "A02": "walk_backward",
    "A03": "walk_along",
}
KINVIS_MARKER_ALIASES: dict[str, str | tuple[str, ...]] = {
    "LASI": "left_hip",
    "RASI": "right_hip",
    "LPSI": "left_hip",
    "RPSI": "right_hip",
    "L_FME": "left_knee",
    "R_FME": "right_knee",
    "L_FAL": "left_ankle",
    "R_FAL": "right_ankle",
    "L_FM1": "left_big_toe",
    "R_FM1": "right_big_toe",
    "L_FM5": "left_small_toe",
    "R_FM5": "right_small_toe",
    "T10": "torso",
    "C7": "neck",
    "SJN": "neck",
    "SXS": "torso",
    "CLAV": ("left_shoulder", "right_shoulder"),
    "STRN": "torso",
    "SACR": "pelvis",
    "LSHO": "left_shoulder",
    "RSHO": "right_shoulder",
    "LELB": "left_elbow",
    "RELB": "right_elbow",
    "LWRA": "left_wrist",
    "RWRA": "right_wrist",
    "LWRB": "left_pinky_knuckle",
    "RWRB": "right_pinky_knuckle",
    "LFIN": "left_middle_tip",
    "RFIN": "right_middle_tip",
    "LWR": "left_wrist",
    "RWR": "right_wrist",
    "LAC": "left_elbow",
    "RAC": "right_elbow",
    "LEP": "left_elbow",
    "REP": "right_elbow",
    "LFHD": "left_eye",
    "RFHD": "right_eye",
    "LBHD": "left_ear",
    "RBHD": "right_ear",
    "HEDA": "nose",
    "HEDL": "left_ear",
    "HEDP": "neck",
    "HEDO": "right_ear",
    "L_FM2": "left_big_toe",
    "R_FM2": "right_big_toe",
    "L_FCC": "left_heel",
    "R_FCC": "right_heel",
    "L_TAM": "left_ankle",
    "R_TAM": "right_ankle",
    "L_FLE": "left_knee",
    "R_FLE": "right_knee",
    "L_SIA": "left_hip",
    "R_SIA": "right_hip",
    "L_IPS": "left_hip",
    "R_IPS": "right_hip",
    "TV10": "torso",
    "CV7": "neck",
}


def _alias_source_is_available(source: str | tuple[str, ...], marker_triplets: dict[str, tuple[int, int, int]]) -> bool:
    if isinstance(source, str):
        return source in marker_triplets
    return any(marker in marker_triplets for marker in source)


def _resolve_alias_source(
    source: str | tuple[str, ...],
    transformed_markers: dict[str, tuple[float, float, float]],
) -> tuple[float, float, float] | None:
    if isinstance(source, str):
        return transformed_markers.get(source)

    values = [transformed_markers.get(marker) for marker in source]
    values = [value for value in values if value is not None]
    if not values:
        return None

    return (
        float(statistics.mean(value[0] for value in values)),
        float(statistics.mean(value[1] for value in values)),
        float(statistics.mean(value[2] for value in values)),
    )


def _to_number(value: str):
    value = value.strip()
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _normalize_label(value: str) -> str:
    return value.strip().strip("\ufeff")


def _format_float(value: float) -> str:
    text = f"{value:.10f}"
    text = text.rstrip("0").rstrip(".")
    return text if text else "0"


def _format_time(value: float) -> str:
    return _format_float(value)


def _infer_rate_from_times(times: list[float]) -> float | None:
    diffs = [t2 - t1 for t1, t2 in zip(times, times[1:]) if t2 > t1]
    if not diffs:
        return None
    diff = diffs[0] if len(diffs) == 1 else statistics.median(diffs)
    if diff <= 0:
        return None
    return 1.0 / diff


def _save_yaml(meta: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(meta, sort_keys=False, allow_unicode=False), encoding="utf-8")


def _load_body_measurements(root: Path) -> dict[str, dict[str, object]]:
    measurements_path = root / "dataset" / "bodyMeasurements.csv"
    if not measurements_path.exists():
        return {}

    with measurements_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        out: dict[str, dict[str, object]] = {}
        for row in reader:
            subject = row.get("Subject")
            if not subject:
                continue
            cleaned = {
                f"meta_{key.lower().replace(' ', '_').replace('(', '').replace(')', '')}": _to_number(str(value))
                for key, value in row.items()
                if key != "Subject"
            }
            out[subject] = cleaned
        return out


def _parse_name(filename: str, prefix: str | None = None) -> tuple[str, str, str] | None:
    stem = Path(filename).stem
    if "_Npose" in stem:
        return None
    if prefix and stem.startswith(prefix):
        stem = stem[len(prefix) :]
    if stem.startswith("ik_"):
        stem = stem[3:]
    parts = stem.split("_")
    if len(parts) != 3:
        return None
    subject, condition, run = parts
    if not subject.startswith("S") or not condition.startswith("A") or not run.startswith("T"):
        return None
    return subject, condition, f"{int(run[1:]):03d}"


def _trial_base_name(subject: str, condition: str, run: str) -> str:
    return f"subject_{subject}_cond_{condition}_run_{run}"


def _copy_raw_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _detect_local_extrema(
    times: list[float],
    values: list[float | None],
    kind: str,
    min_separation_s: float = 0.35,
    min_prominence_mm: float = 20.0,
) -> list[float]:
    candidates: list[tuple[float, float]] = []
    for idx in range(1, len(values) - 1):
        prev_value = values[idx - 1]
        value = values[idx]
        next_value = values[idx + 1]
        if prev_value is None or value is None or next_value is None:
            continue
        if kind == "max":
            is_extremum = value >= prev_value and value > next_value
            prominence = min(value - prev_value, value - next_value)
        else:
            is_extremum = value <= prev_value and value < next_value
            prominence = min(prev_value - value, next_value - value)
        if not is_extremum or prominence < min_prominence_mm:
            continue
        candidates.append((times[idx], prominence))

    selected: list[tuple[float, float]] = []
    for time_value, prominence in candidates:
        if not selected or (time_value - selected[-1][0]) >= min_separation_s:
            selected.append((time_value, prominence))
            continue
        if prominence > selected[-1][1]:
            selected[-1] = (time_value, prominence)

    return [time_value for time_value, _ in selected]


def _infer_gait_events(
    times: list[float],
    marker_series: dict[str, list[tuple[float, float, float] | None]],
) -> dict[str, list[float]]:
    empty = {
        "l_heel_strike": [],
        "l_toe_off": [],
        "r_heel_strike": [],
        "r_toe_off": [],
    }
    if not times:
        return empty

    pelvis_series = marker_series.get("pelvis", [])
    pelvis_points = [point for point in pelvis_series if point is not None]
    if not pelvis_points:
        return empty

    pelvis_x = [point[0] for point in pelvis_points]
    pelvis_y = [point[1] for point in pelvis_points]
    progression_axis = 1 if (max(pelvis_y) - min(pelvis_y)) >= (max(pelvis_x) - min(pelvis_x)) else 0

    pelvis_axis_values = [point[progression_axis] for point in pelvis_series if point is not None]
    direction = 1.0
    if pelvis_axis_values and pelvis_axis_values[-1] < pelvis_axis_values[0]:
        direction = -1.0

    out = dict(empty)
    side_map = (("left", "l"), ("right", "r"))
    for side_name, side_prefix in side_map:
        heel_series = marker_series.get(f"{side_name}_heel", [])
        toe_series = marker_series.get(f"{side_name}_big_toe", [])
        heel_relative: list[float | None] = []
        toe_relative: list[float | None] = []
        for idx in range(len(times)):
            pelvis_point = pelvis_series[idx] if idx < len(pelvis_series) else None
            heel_point = heel_series[idx] if idx < len(heel_series) else None
            toe_point = toe_series[idx] if idx < len(toe_series) else None
            if pelvis_point is None or heel_point is None:
                heel_relative.append(None)
            else:
                heel_relative.append(direction * (heel_point[progression_axis] - pelvis_point[progression_axis]))
            if pelvis_point is None or toe_point is None:
                toe_relative.append(None)
            else:
                toe_relative.append(direction * (toe_point[progression_axis] - pelvis_point[progression_axis]))

        out[f"{side_prefix}_heel_strike"] = _detect_local_extrema(times, heel_relative, kind="max")
        out[f"{side_prefix}_toe_off"] = _detect_local_extrema(times, toe_relative, kind="min")

    return out


def _collect_runs(dataset_dir: Path) -> dict[tuple[str, str, str], dict[str, object]]:
    runs: dict[tuple[str, str, str], dict[str, object]] = {}

    for partition in ("videonly", "videoandimus"):
        partition_dir = dataset_dir / partition
        if not partition_dir.exists():
            continue

        for in_path in sorted(partition_dir.glob("S*/S*_A*_T*.csv")):
            parsed = _parse_name(in_path.name)
            if parsed is None:
                continue
            subject, condition, run = parsed
            key = (subject, condition, run)
            record = runs.setdefault(
                key,
                {
                    "subject": subject,
                    "condition": condition,
                    "run": run,
                    "source_partitions": set(),
                },
            )
            record["trajectory_csv"] = in_path
            record["trajectory_partition"] = partition
            record["source_partitions"].add(partition)

        if partition != "videoandimus":
            continue

        for in_path in sorted(partition_dir.glob("S*/ik_S*_A*_T*.mot")):
            parsed = _parse_name(in_path.name, prefix="ik_")
            if parsed is None:
                continue
            subject, condition, run = parsed
            key = (subject, condition, run)
            record = runs.setdefault(
                key,
                {
                    "subject": subject,
                    "condition": condition,
                    "run": run,
                    "source_partitions": set(),
                },
            )
            record["joint_angles_mot"] = in_path
            record["source_partitions"].add(partition)

        for in_path in sorted(partition_dir.glob("S*/S*_A*_T*.sto")):
            parsed = _parse_name(in_path.name)
            if parsed is None:
                continue
            subject, condition, run = parsed
            key = (subject, condition, run)
            record = runs.setdefault(
                key,
                {
                    "subject": subject,
                    "condition": condition,
                    "run": run,
                    "source_partitions": set(),
                },
            )
            record["imu_sto"] = in_path
            record["source_partitions"].add(partition)

    return runs


def _subject_infos(
    runs: dict[tuple[str, str, str], dict[str, object]],
    body_meta: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    by_subject: dict[str, dict[str, object]] = {}
    for record in runs.values():
        subject = str(record["subject"])
        info = by_subject.setdefault(
            subject,
            {
                "dataset": DATASET_TAG,
                "subject": subject,
                "subject_id": f"subject_{subject}",
                "conditions": set(),
                "modalities": set(),
                "source_partitions": set(),
                "n_runs": 0,
            },
        )
        info["conditions"].add(str(record["condition"]))
        info["n_runs"] += 1
        info["source_partitions"].update(record.get("source_partitions", set()))
        if "trajectory_csv" in record:
            info["modalities"].add("markers")
        if "joint_angles_mot" in record:
            info["modalities"].add("joint_angles")
        if "imu_sto" in record:
            info["modalities"].add("imu")

    out: dict[str, dict[str, object]] = {}
    for subject, info in by_subject.items():
        finalized = {
            "dataset": info["dataset"],
            "subject": info["subject"],
            "subject_id": info["subject_id"],
            "conditions": sorted(info["conditions"]),
            "modalities": sorted(info["modalities"]),
            "source_partitions": sorted(info["source_partitions"]),
            "n_runs": info["n_runs"],
        }
        finalized.update(body_meta.get(subject, {}))
        out[subject] = finalized
    return out


def _convert_trajectory_csv(
    in_path: Path,
    out_csv: Path,
    subject: str,
    condition: str,
    run: str,
) -> dict[str, object]:
    coord_offset = {"x": 0.0, "y": 0.0, "z": 0.0}
    offset_source = "none"
    marker_series = {
        "pelvis": [],
        "left_heel": [],
        "right_heel": [],
        "left_big_toe": [],
        "right_big_toe": [],
    }
    times: list[float] = []

    with in_path.open("r", encoding="utf-8") as file_in, out_csv.open("w", newline="", encoding="utf-8") as file_out:
        reader = csv.reader(file_in)
        writer = csv.writer(file_out, delimiter=PROCESSED_DELIMITER)

        header = next(reader, None)
        if not header:
            return {"n_frames": 0, "sample_rate": CSV_SAMPLE_RATE, "labels": []}
        header = [_normalize_label(column) for column in header]
        header_index = {column: idx for idx, column in enumerate(header)}
        axis_indices = {
            "x": [idx for idx, column in enumerate(header) if column.endswith("_x")],
            "y": [idx for idx, column in enumerate(header) if column.endswith("_y")],
            "z": [idx for idx, column in enumerate(header) if column.endswith("_z")],
        }
        marker_triplets: dict[str, tuple[int, int, int]] = {}
        for column in header:
            if not column.endswith("_x"):
                continue
            marker = column[:-2]
            y_column = f"{marker}_y"
            z_column = f"{marker}_z"
            if y_column in header_index and z_column in header_index:
                marker_triplets[marker] = (header_index[column], header_index[y_column], header_index[z_column])

        pelvis_indices = {
            "x": header_index.get("pelvis_x"),
            "y": header_index.get("pelvis_y"),
            "z": header_index.get("pelvis_z"),
        }
        offset_ready = False

        alias_pairs = [
            (alias, source)
            for alias, source in KINVIS_MARKER_ALIASES.items()
            if _alias_source_is_available(source, marker_triplets)
        ]
        alias_columns = [f"{alias}_{axis}" for alias, _ in alias_pairs for axis in ("x", "y", "z")]
        output_header = header + alias_columns + ["time"]
        writer.writerow(output_header)

        n_frames = 0
        for frame_idx, row in enumerate(reader):
            if not row:
                continue
            row = [cell.strip() for cell in row]
            if len(row) < len(header):
                row.extend([""] * (len(header) - len(row)))
            elif len(row) > len(header):
                row = row[: len(header)]

            if not offset_ready:
                has_pelvis = all(pelvis_indices[axis] is not None for axis in ("x", "y", "z"))
                if has_pelvis:
                    try:
                        coord_offset = {
                            axis: float(row[int(pelvis_indices[axis])])
                            for axis in ("x", "y", "z")
                        }
                        offset_source = "pelvis_first_frame"
                        offset_ready = True
                    except (ValueError, TypeError):
                        pass
                if not offset_ready:
                    try:
                        coord_offset = {
                            axis: statistics.mean(float(row[idx]) for idx in axis_indices[axis]) if axis_indices[axis] else 0.0
                            for axis in ("x", "y", "z")
                        }
                        offset_source = "mean_first_frame"
                        offset_ready = True
                    except (ValueError, TypeError):
                        coord_offset = {"x": 0.0, "y": 0.0, "z": 0.0}
                        offset_source = "none"

            out_row = list(row)
            transformed_markers: dict[str, tuple[float, float, float]] = {}
            for marker, (idx_x, idx_y, idx_z) in marker_triplets.items():
                value_x = out_row[idx_x]
                value_y = out_row[idx_y]
                value_z = out_row[idx_z]
                if value_x == "" or value_y == "" or value_z == "":
                    continue
                try:
                    src_x = float(value_x) - coord_offset["x"]
                    src_y = float(value_y) - coord_offset["y"]
                    src_z = float(value_z) - coord_offset["z"]
                except ValueError:
                    continue

                kinvis_x = src_x
                kinvis_y = src_z
                kinvis_z = -src_y
                transformed_markers[marker] = (kinvis_x, kinvis_y, kinvis_z)
                out_row[idx_x] = _format_float(kinvis_x)
                out_row[idx_y] = _format_float(kinvis_y)
                out_row[idx_z] = _format_float(kinvis_z)

            alias_values: list[str] = []
            alias_frame: dict[str, tuple[float, float, float] | None] = {}
            for alias, source in alias_pairs:
                values = _resolve_alias_source(source, transformed_markers)
                alias_frame[alias] = values
                if values is None:
                    alias_values.extend(["", "", ""])
                    continue
                alias_values.extend([_format_float(value) for value in values])

            current_time = frame_idx / CSV_SAMPLE_RATE
            writer.writerow(out_row + alias_values + [_format_time(current_time)])
            times.append(current_time)
            for tracked_marker in marker_series:
                marker_series[tracked_marker].append(
                    transformed_markers.get(tracked_marker) or alias_frame.get(tracked_marker)
                )
            n_frames += 1

    labels: list[str] = []
    seen = set()
    for column in output_header:
        if not column.endswith("_x"):
            continue
        label = column[:-2]
        if label in seen:
            continue
        labels.append(label)
        seen.add(label)

    metadata = {
        "dataset": DATASET_TAG,
        "subject": subject,
        "subject_id": f"subject_{subject}",
        "condition": condition,
        "condition_label": WALK_CONDITIONS.get(condition, condition),
        "run": run,
        "feature": "Trajectories",
        "sample_rate": CSV_SAMPLE_RATE,
        "n_frames": n_frames,
        "units": "mm",
        "labels": labels,
        "n_markers": len(labels),
        "source_file": str(in_path.resolve()),
        "source_filename": in_path.name,
        "mode": "multimodal_trajectory",
        "coordinate_transform": "translated_to_origin_then_kinvis_axes",
        "coordinate_offset_mm": coord_offset,
        "coordinate_offset_source": offset_source,
        "axis_mapping": {"x": "x_centered", "y": "z_centered", "z": "-y_centered"},
        "kinvis_alias_markers_added": [alias for alias, _ in alias_pairs],
    }
    metadata["gait_events"] = _infer_gait_events(times, marker_series) if condition in WALK_CONDITIONS else None
    return metadata


def _convert_joint_angles_mot(
    in_path: Path,
    out_csv: Path,
    subject: str,
    condition: str,
    run: str,
) -> dict[str, object]:
    with in_path.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    try:
        end_header = lines.index("endheader\n") + 1
    except ValueError:
        try:
            end_header = lines.index("endheader\r\n") + 1
        except ValueError as exc:
            raise ValueError(f"Invalid .mot file, missing endheader: {in_path}") from exc

    data_lines = [line for line in lines[end_header:] if line.strip()]
    if not data_lines:
        return {"n_frames": 0, "sample_rate": 1.0, "labels": []}

    header = data_lines[0].split()
    if not header or header[0].lower() != "time":
        raise ValueError(f"Invalid .mot header in {in_path}")

    times: list[float] = []
    rows: list[list[float]] = []
    for line in data_lines[1:]:
        fields = line.strip().split()
        if len(fields) != len(header):
            continue
        parsed = [float(field) if field else float("nan") for field in fields]
        rows.append(parsed)
        times.append(parsed[0])

    sample_rate = _infer_rate_from_times(times) or 1.0
    with out_csv.open("w", newline="", encoding="utf-8") as file_out:
        writer = csv.writer(file_out, delimiter=PROCESSED_DELIMITER)
        writer.writerow(header)
        for parsed in rows:
            writer.writerow([_format_time(value) if idx == 0 else _format_float(value) for idx, value in enumerate(parsed)])

    return {
        "dataset": DATASET_TAG,
        "subject": subject,
        "subject_id": f"subject_{subject}",
        "condition": condition,
        "condition_label": WALK_CONDITIONS.get(condition, condition),
        "run": run,
        "feature": "jointAngles",
        "sample_rate": float(sample_rate),
        "n_frames": len(rows),
        "units": "degrees",
        "labels": header[1:],
        "n_signals": len(header[1:]),
        "source_file": str(in_path.resolve()),
        "source_filename": in_path.name,
        "mode": "video_and_imu",
    }


def _write_testbed_label(
    out_path: Path,
    subject: str,
    condition: str,
    run: str,
    record: dict[str, object],
    processed_outputs: list[str],
    raw_outputs: list[str],
) -> None:
    partition_names = sorted(str(value) for value in record.get("source_partitions", set()))
    testbed = {
        "dataset": DATASET_TAG,
        "subject": subject,
        "subject_id": f"subject_{subject}",
        "condition": condition,
        "condition_label": WALK_CONDITIONS.get(condition, condition),
        "run": run,
        "task_group": "walking" if condition in WALK_CONDITIONS else "other",
        "source_partitions": partition_names,
        "acquisition_mode": "video_and_imu" if "videoandimus" in partition_names else "video_only",
        "processed_outputs": processed_outputs,
        "raw_outputs": raw_outputs,
    }
    _save_yaml(testbed, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the Multimodal video and IMU kinematic dataset to canonical EUROBENCH outputs.")
    parser.add_argument(
        "--source-root",
        default=str(DATASET_ROOT),
        help="Root folder of the dataset containing the dataset/ folder.",
    )
    parser.add_argument(
        "--out-root",
        default=str(DATASET_ROOT / "eurobench"),
        help="Output folder for EUROBENCH-style files.",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root)
    out_root = Path(args.out_root)
    dataset_dir = source_root / "dataset"
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Missing dataset folder: {dataset_dir}")

    body_meta = _load_body_measurements(source_root)
    runs = _collect_runs(dataset_dir)
    if not runs:
        raise RuntimeError("No compatible source runs were found.")

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    raw_root = out_root / "raw_data"
    raw_root.mkdir(parents=True, exist_ok=True)

    subject_infos = _subject_infos(runs, body_meta)
    for subject, info in sorted(subject_infos.items()):
        _save_yaml(info, out_root / f"subject_{subject}_info.yaml")

    counts = defaultdict(int)
    log_rows: list[list[str | int]] = []

    for key in sorted(runs):
        record = runs[key]
        subject = str(record["subject"])
        condition = str(record["condition"])
        run = str(record["run"])
        base_name = _trial_base_name(subject, condition, run)
        processed_outputs: list[str] = []
        raw_outputs: list[str] = []

        trajectory_path = record.get("trajectory_csv")
        if isinstance(trajectory_path, Path):
            out_csv = out_root / f"{base_name}_Trajectories.csv"
            metadata = _convert_trajectory_csv(trajectory_path, out_csv, subject, condition, run)
            processed_outputs.append(out_csv.name)
            counts["Trajectories"] += 1
            log_rows.append(["Trajectories", str(trajectory_path), out_csv.name, int(metadata["n_frames"]), "ok"])

            raw_markers = raw_root / f"{base_name}_markers.csv"
            _copy_raw_file(trajectory_path, raw_markers)
            raw_outputs.append(raw_markers.relative_to(out_root).as_posix())
            counts["raw_markers"] += 1
            log_rows.append(["raw_markers", str(trajectory_path), raw_markers.relative_to(out_root).as_posix(), "", "ok"])

            gait_events = metadata.get("gait_events")
            if condition in WALK_CONDITIONS and isinstance(gait_events, dict):
                gait_path = out_root / f"{base_name}_gaitEvents.yaml"
                _save_yaml(gait_events, gait_path)
                processed_outputs.append(gait_path.name)
                counts["gaitEvents"] += 1
                log_rows.append(["gaitEvents", str(trajectory_path), gait_path.name, "", "ok"])

        joint_angles_path = record.get("joint_angles_mot")
        if isinstance(joint_angles_path, Path):
            out_csv = out_root / f"{base_name}_jointAngles.csv"
            metadata = _convert_joint_angles_mot(joint_angles_path, out_csv, subject, condition, run)
            processed_outputs.append(out_csv.name)
            counts["jointAngles"] += 1
            log_rows.append(["jointAngles", str(joint_angles_path), out_csv.name, int(metadata["n_frames"]), "ok"])

            raw_joint_angles = raw_root / f"{base_name}_jointAngles.mot"
            _copy_raw_file(joint_angles_path, raw_joint_angles)
            raw_outputs.append(raw_joint_angles.relative_to(out_root).as_posix())
            counts["raw_jointAngles"] += 1
            log_rows.append(["raw_jointAngles", str(joint_angles_path), raw_joint_angles.relative_to(out_root).as_posix(), "", "ok"])

        imu_path = record.get("imu_sto")
        if isinstance(imu_path, Path):
            raw_imu = raw_root / f"{base_name}_imu.sto"
            _copy_raw_file(imu_path, raw_imu)
            raw_outputs.append(raw_imu.relative_to(out_root).as_posix())
            counts["raw_imu"] += 1
            log_rows.append(["raw_imu", str(imu_path), raw_imu.relative_to(out_root).as_posix(), "", "ok"])

        testbed_path = out_root / f"{base_name}_testbedLabel.yaml"
        _write_testbed_label(testbed_path, subject, condition, run, record, processed_outputs, raw_outputs)
        counts["testbedLabel"] += 1
        log_rows.append(["testbedLabel", base_name, testbed_path.name, "", "ok"])

    counts["subject_info"] = len(subject_infos)
    summary_path = out_root / "conversion_log.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.writer(log_file, delimiter=PROCESSED_DELIMITER)
        writer.writerow(["feature", "input", "output", "n_frames", "status"])
        writer.writerows(log_rows)

    print("Done.")
    for key in sorted(counts):
        print(f"{key}: {counts[key]}")
    print(f"Log saved to: {summary_path}")


if __name__ == "__main__":
    main()
