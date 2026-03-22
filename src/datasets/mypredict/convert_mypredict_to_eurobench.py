import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from detect_gait_events_markers import detect_gait_events_markers


DATASET_TAG = "mypredict"
DAY_RE = re.compile(r"^Day_(?P<day>\d+)$")
TRIAL_RE = re.compile(r"^Trial_(?P<trial>\d+)$")

SOURCE_MARKER_KEYS = [
    "Mrk_C7SpinalProcess",
    "Mrk_IJ",
    "Mrk_Left_ASI",
    "Mrk_Right_ASI",
    "Mrk_Sacrum",
    "Mrk_T4SpinalProcess",
    "Mrk_T12SpinalProcess",
    "Mrk_Left_Acromion",
    "Mrk_Right_Acromion",
    "Mrk_Left_GreaterTrochanter",
    "Mrk_Right_GreaterTrochanter",
    "Mrk_Left_KneeLatEpicondyle",
    "Mrk_Left_KneeMedEpicondyle",
    "Mrk_Right_KneeLatEpicondyle",
    "Mrk_Right_KneeMedEpicondyle",
    "Mrk_Left_LatMalleolus",
    "Mrk_Left_MedMalleolus",
    "Mrk_Right_LatMalleolus",
    "Mrk_Right_MedMalleolus",
    "Mrk_Left_HeelFoot",
    "Mrk_Right_HeelFoot",
    "Mrk_Left_Toe",
    "Mrk_Right_Toe",
    "Mrk_Left_FirstMetatarsal",
    "Mrk_Left_FifthMetatarsal",
    "Mrk_Right_FirstMetatarsal",
    "Mrk_Right_FifthMetatarsal",
]

OUTPUT_MARKER_ORDER = [
    "C7",
    "IJ",
    "CLAV",
    "STRN",
    "T10",
    "LASI",
    "LPSI",
    "RASI",
    "RPSI",
    "SACR",
    "PELV",
    "LSHO",
    "RSHO",
    "LTHI",
    "RTHI",
    "LKNE",
    "RKNE",
    "LANK",
    "RANK",
    "LHEE",
    "RHEE",
    "LTOE",
    "RTOE",
    "LMT1",
    "LMT5",
    "RMT1",
    "RMT5",
]

JOINT_ANGLE_MAP = {
    "Ang_Left_Hip": "LHipAngles",
    "Ang_Left_Knee": "LKneeAngles",
    "Ang_Left_Ankle": "LAnkleAngles",
    "Ang_Right_Hip": "RHipAngles",
    "Ang_Right_Knee": "RKneeAngles",
    "Ang_Right_Ankle": "RAnkleAngles",
    "Ang_L5S1": "L5S1Angles",
}


def _to_builtin(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _format_label_value(value: float) -> str:
    value = float(value)
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.12g}"


def _relative_time(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    return arr - float(arr[0])


def _sample_rate_hz(values: np.ndarray) -> float | None:
    time = np.asarray(values, dtype=float).reshape(-1)
    if time.size < 2:
        return None
    dt = np.diff(time)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return None
    return float(1.0 / np.median(dt))


def _sorted_day_keys(root: h5py.File) -> list[str]:
    def _key(name: str) -> int:
        match = DAY_RE.match(name)
        return int(match.group("day")) if match else 10**9

    return sorted((k for k in root.keys() if DAY_RE.match(k)), key=_key)


def _sorted_trial_keys(day_group: h5py.Group) -> list[str]:
    def _key(name: str) -> int:
        match = TRIAL_RE.match(name)
        return int(match.group("trial")) if match else 10**9

    return sorted((k for k in day_group.keys() if TRIAL_RE.match(k)), key=_key)


def _day_tag(day_key: str) -> str:
    match = DAY_RE.match(day_key)
    if match is None:
        return day_key.replace("_", "")
    return f"Day{int(match.group('day'))}"


def _trial_run(trial_key: str) -> str:
    match = TRIAL_RE.match(trial_key)
    if match is None:
        return trial_key
    return match.group("trial")


def _label_segments(labels: np.ndarray, time: np.ndarray) -> list[dict]:
    labels_arr = np.asarray(labels, dtype=float).reshape(-1)
    time_arr = np.asarray(time, dtype=float).reshape(-1)
    if labels_arr.size == 0 or time_arr.size == 0:
        return []
    if labels_arr.size != time_arr.size:
        raise ValueError("Label array length does not match time array length.")

    segments: list[dict] = []
    start_idx = 0
    current = float(labels_arr[0])
    for idx in range(1, labels_arr.size):
        if not np.isclose(labels_arr[idx], current, atol=1e-9, rtol=0.0):
            end_idx = idx
            segments.append(
                {
                    "label": _format_label_value(current),
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx - 1),
                    "n_samples": int(end_idx - start_idx),
                    "start_time_s": round(float(time_arr[start_idx]), 6),
                    "end_time_s": round(float(time_arr[end_idx - 1]), 6),
                    "duration_s": round(float(time_arr[end_idx - 1] - time_arr[start_idx]), 6),
                }
            )
            current = float(labels_arr[idx])
            start_idx = idx

    end_idx = labels_arr.size
    segments.append(
        {
            "label": _format_label_value(current),
            "start_idx": int(start_idx),
            "end_idx": int(end_idx - 1),
            "n_samples": int(end_idx - start_idx),
            "start_time_s": round(float(time_arr[start_idx]), 6),
            "end_time_s": round(float(time_arr[end_idx - 1]), 6),
            "duration_s": round(float(time_arr[end_idx - 1] - time_arr[start_idx]), 6),
        }
    )
    return segments


def _label_histogram(labels: np.ndarray) -> dict[str, int]:
    unique_vals, counts = np.unique(np.asarray(labels, dtype=float).reshape(-1), return_counts=True)
    return {_format_label_value(val): int(count) for val, count in zip(unique_vals, counts)}


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def _mean_markers(arrays: list[np.ndarray | None]) -> np.ndarray | None:
    valid = [arr for arr in arrays if arr is not None]
    if not valid:
        return None
    return np.mean(np.stack(valid, axis=0), axis=0)


def _build_trajectories(markers_group: h5py.Group) -> tuple[pd.DataFrame, list[str]]:
    if "Time" not in markers_group:
        raise KeyError("Markers group is missing Time.")

    time = _relative_time(markers_group["Time"][:])
    df = pd.DataFrame({"time": time})

    source_markers: dict[str, np.ndarray | None] = {}
    missing: list[str] = []
    for src_name in SOURCE_MARKER_KEYS:
        if src_name not in markers_group:
            missing.append(src_name)
            source_markers[src_name] = None
            continue
        arr = np.asarray(markers_group[src_name][:], dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"{src_name} must have shape (n, 3), got {arr.shape}")
        source_markers[src_name] = arr

    derived_markers = {
        "C7": source_markers.get("Mrk_C7SpinalProcess"),
        "IJ": source_markers.get("Mrk_IJ"),
        "CLAV": source_markers.get("Mrk_IJ"),
        "STRN": source_markers.get("Mrk_IJ"),
        "T10": _mean_markers(
            [
                source_markers.get("Mrk_T4SpinalProcess"),
                source_markers.get("Mrk_T12SpinalProcess"),
            ]
        ),
        "LASI": source_markers.get("Mrk_Left_ASI"),
        "LPSI": source_markers.get("Mrk_Sacrum"),
        "RASI": source_markers.get("Mrk_Right_ASI"),
        "RPSI": source_markers.get("Mrk_Sacrum"),
        "SACR": source_markers.get("Mrk_Sacrum"),
        "PELV": _mean_markers(
            [
                source_markers.get("Mrk_Left_ASI"),
                source_markers.get("Mrk_Right_ASI"),
                source_markers.get("Mrk_Sacrum"),
            ]
        ),
        "LSHO": source_markers.get("Mrk_Left_Acromion"),
        "RSHO": source_markers.get("Mrk_Right_Acromion"),
        "LTHI": source_markers.get("Mrk_Left_GreaterTrochanter"),
        "RTHI": source_markers.get("Mrk_Right_GreaterTrochanter"),
        "LKNE": _mean_markers(
            [
                source_markers.get("Mrk_Left_KneeLatEpicondyle"),
                source_markers.get("Mrk_Left_KneeMedEpicondyle"),
            ]
        ),
        "RKNE": _mean_markers(
            [
                source_markers.get("Mrk_Right_KneeLatEpicondyle"),
                source_markers.get("Mrk_Right_KneeMedEpicondyle"),
            ]
        ),
        "LANK": _mean_markers(
            [
                source_markers.get("Mrk_Left_LatMalleolus"),
                source_markers.get("Mrk_Left_MedMalleolus"),
            ]
        ),
        "RANK": _mean_markers(
            [
                source_markers.get("Mrk_Right_LatMalleolus"),
                source_markers.get("Mrk_Right_MedMalleolus"),
            ]
        ),
        "LHEE": source_markers.get("Mrk_Left_HeelFoot"),
        "RHEE": source_markers.get("Mrk_Right_HeelFoot"),
        "LTOE": source_markers.get("Mrk_Left_Toe"),
        "RTOE": source_markers.get("Mrk_Right_Toe"),
        "LMT1": source_markers.get("Mrk_Left_FirstMetatarsal"),
        "LMT5": source_markers.get("Mrk_Left_FifthMetatarsal"),
        "RMT1": source_markers.get("Mrk_Right_FirstMetatarsal"),
        "RMT5": source_markers.get("Mrk_Right_FifthMetatarsal"),
    }

    for out_name in OUTPUT_MARKER_ORDER:
        arr = derived_markers.get(out_name)
        if arr is None:
            continue
        df[f"{out_name}_x"] = arr[:, 0]
        df[f"{out_name}_y"] = arr[:, 1]
        df[f"{out_name}_z"] = arr[:, 2]

    return df, missing


def _build_joint_angles(trial_group: h5py.Group) -> pd.DataFrame:
    if "Time" not in trial_group:
        raise KeyError("Trial group is missing Time.")

    time = _relative_time(trial_group["Time"][:])
    n_samples = time.shape[0]
    df = pd.DataFrame()

    for src_name, out_prefix in JOINT_ANGLE_MAP.items():
        if src_name not in trial_group:
            raise KeyError(f"Missing required angle dataset: {src_name}")
        arr = np.asarray(trial_group[src_name][:], dtype=float)
        if arr.shape != (n_samples, 3):
            raise ValueError(f"{src_name} must have shape {(n_samples, 3)}, got {arr.shape}")
        df[f"{out_prefix}_x"] = arr[:, 0]
        df[f"{out_prefix}_y"] = arr[:, 1]
        df[f"{out_prefix}_z"] = arr[:, 2]

    df["time"] = time
    return df


def _build_imu(trial_group: h5py.Group) -> pd.DataFrame:
    time = _relative_time(trial_group["Time"][:])
    df = pd.DataFrame({"time": time})
    for key in sorted(k for k in trial_group.keys() if k.startswith("Acc_") or k.startswith("Gyr_")):
        arr = np.asarray(trial_group[key][:], dtype=float)
        if arr.ndim != 2 or arr.shape[0] != time.shape[0] or arr.shape[1] != 3:
            raise ValueError(f"{key} must have shape ({time.shape[0]}, 3), got {arr.shape}")
        df[f"{key}_x"] = arr[:, 0]
        df[f"{key}_y"] = arr[:, 1]
        df[f"{key}_z"] = arr[:, 2]
    if "Label" in trial_group:
        df["label"] = np.asarray(trial_group["Label"][:], dtype=float)
    return df


def _build_emg(trial_group: h5py.Group) -> pd.DataFrame:
    time = _relative_time(trial_group["Time"][:])
    df = pd.DataFrame({"time": time})
    for key in sorted(k for k in trial_group.keys() if k.startswith("EMG_")):
        arr = np.asarray(trial_group[key][:], dtype=float).reshape(-1)
        if arr.shape[0] != time.shape[0]:
            raise ValueError(f"{key} must have length {time.shape[0]}, got {arr.shape}")
        df[key] = arr
    if "Label" in trial_group:
        df["label"] = np.asarray(trial_group["Label"][:], dtype=float)
    return df


def _detect_events(traj_csv: Path) -> tuple[dict[str, list[float]], dict[str, dict]]:
    events: dict[str, list[float]] = {}
    meta: dict[str, dict] = {}

    for side in ("L", "R"):
        try:
            side_events, axis = detect_gait_events_markers(
                str(traj_csv),
                side=side,
                pelvis_marker="SACR",
                heel_marker="HEE",
                toe_marker="TOE",
            )
            events.update({k: [round(float(v), 6) for v in vals] for k, vals in side_events.items() if vals})
            side_key = side.lower()
            meta[side_key] = {
                "status": "ok",
                "axis": axis,
                "n_heel_strike": int(len(side_events.get(f"{side_key}_heel_strike", []))),
                "n_toe_off": int(len(side_events.get(f"{side_key}_toe_off", []))),
            }
        except Exception as exc:  # noqa: BLE001
            side_key = side.lower()
            meta[side_key] = {
                "status": "error",
                "axis": None,
                "n_heel_strike": 0,
                "n_toe_off": 0,
                "error": str(exc),
            }

    return events, meta


def _subject_metadata(meta_group: h5py.Group | None) -> dict:
    if meta_group is None:
        return {}
    out: dict = {}
    for key in sorted(meta_group.keys()):
        out[key] = _to_builtin(meta_group[key][()])
    return out


def _build_info_payload(
    subject: str,
    day_key: str,
    trial_key: str,
    source_path: Path,
    source_meta: dict,
    joint_df: pd.DataFrame,
    traj_df: pd.DataFrame,
    labels: np.ndarray,
    label_segments: list[dict],
    events_meta: dict[str, dict],
    out_names: dict[str, str | None],
    missing_markers: list[str],
    source_time_start_s: float,
    marker_time_start_s: float,
) -> dict:
    unique_labels = sorted({_format_label_value(v) for v in np.asarray(labels, dtype=float).reshape(-1)})
    angle_series = sorted({col.rsplit("_", 1)[0] for col in joint_df.columns if col != "time"})
    marker_labels = sorted({col.rsplit("_", 1)[0] for col in traj_df.columns if col != "time"})
    features = ["Trajectories", "jointAngles"]
    if out_names["imu"]:
        features.append("imu")
    if out_names["emg"]:
        features.append("emg")

    return {
        "dataset": DATASET_TAG,
        "subject": subject,
        "participant": subject,
        "session": _day_tag(day_key),
        "day": day_key,
        "trial": trial_key,
        "run": _trial_run(trial_key),
        "condition": "mixed_protocol",
        "features": features,
        "trajectory_file": out_names["traj"],
        "point_gait_events_file": out_names["point_events"],
        "joint_angles_file": out_names["joint"],
        "gait_events_file": out_names["events"],
        "info_file": out_names["info"],
        "imu_file": out_names["imu"],
        "emg_file": out_names["emg"],
        "source_file": str(source_path.resolve()),
        "source_group": f"{day_key}/{trial_key}",
        "source_time_start_s": round(float(source_time_start_s), 6),
        "source_marker_time_start_s": round(float(marker_time_start_s), 6),
        "trajectory_sample_rate_hz": _sample_rate_hz(traj_df["time"].to_numpy()),
        "joint_angle_sample_rate_hz": _sample_rate_hz(joint_df["time"].to_numpy()),
        "n_marker_frames": int(len(traj_df)),
        "n_joint_angle_frames": int(len(joint_df)),
        "n_markers": int(len(marker_labels)),
        "marker_labels": marker_labels,
        "missing_markers": missing_markers,
        "trajectory_unit": "mm",
        "joint_angle_unit": "deg",
        "joint_angle_series": angle_series,
        "label_values": unique_labels,
        "label_histogram": _label_histogram(labels),
        "label_segments": label_segments,
        "event_detection": events_meta,
        "subject_metadata": source_meta,
        "notes": [
            "Trial and marker time were shifted to start at 0.0 s for Eurobench outputs.",
            "Marker trajectories were exported at the native marker rate and joint angles at the native angle rate.",
            "Marker and angle units were inferred from value ranges because the HDF5 source has no unit attributes.",
            "Canonical Eurobench markers such as PELV, LTHI/RTHI, LKNE/RKNE, and LANK/RANK were derived from source anatomical markers.",
            "The source Label semantics are not documented locally; raw values were preserved as compressed segments.",
        ],
    }


def convert_file(
    h5_path: Path,
    eurobench_root: Path,
    overwrite: bool = False,
    save_imu: bool = False,
    save_emg: bool = False,
    trial_limit: int | None = None,
) -> list[dict]:
    rows: list[dict] = []
    written_trials = 0

    with h5py.File(h5_path, "r") as handle:
        meta = _subject_metadata(handle["Meta"] if "Meta" in handle else None)
        subject = str(meta.get("Code") or h5_path.stem).strip() or h5_path.stem
        out_dir = eurobench_root / subject
        out_dir.mkdir(parents=True, exist_ok=True)

        for day_key in _sorted_day_keys(handle):
            day_group = handle[day_key]
            for trial_key in _sorted_trial_keys(day_group):
                if trial_limit is not None and written_trials >= int(trial_limit):
                    break

                row = {
                    "subject": subject,
                    "day": day_key,
                    "trial": trial_key,
                    "status": "ok",
                    "error": "",
                    "source_file": str(h5_path.resolve()),
                    "n_marker_frames": 0,
                    "n_joint_angle_frames": 0,
                    "trajectory_sample_rate_hz": np.nan,
                    "joint_angle_sample_rate_hz": np.nan,
                    "n_label_values": 0,
                    "n_segments": 0,
                    "n_left_hs": 0,
                    "n_left_to": 0,
                    "n_right_hs": 0,
                    "n_right_to": 0,
                    "out_traj": "",
                    "out_point_events": "",
                    "out_joint": "",
                    "out_events": "",
                    "out_info": "",
                    "out_imu": "",
                    "out_emg": "",
                }

                try:
                    trial_group = day_group[trial_key]
                    markers_group = trial_group["Markers"]
                    day_tag = _day_tag(day_key)
                    base = f"{subject}_{day_tag}_{trial_key}"

                    out_traj = out_dir / f"{base}_Trajectories.csv"
                    out_point_events = out_dir / f"{base}_point_gaitEvents.yaml"
                    out_joint = out_dir / f"{base}_jointAngles.csv"
                    out_events = out_dir / f"{base}_gaitEvents.yaml"
                    out_info = out_dir / f"{base}_info.yaml"
                    out_imu = out_dir / f"{base}_imu.csv"
                    out_emg = out_dir / f"{base}_emg.csv"

                    needed_outputs = [out_traj, out_point_events, out_joint, out_events, out_info]
                    if save_imu:
                        needed_outputs.append(out_imu)
                    if save_emg:
                        needed_outputs.append(out_emg)

                    if not overwrite and all(path.exists() for path in needed_outputs):
                        row["status"] = "skip_exists"
                        row["out_traj"] = str(out_traj)
                        row["out_point_events"] = str(out_point_events)
                        row["out_joint"] = str(out_joint)
                        row["out_events"] = str(out_events)
                        row["out_info"] = str(out_info)
                        row["out_imu"] = str(out_imu) if save_imu else ""
                        row["out_emg"] = str(out_emg) if save_emg else ""
                        rows.append(row)
                        continue

                    traj_df, missing_markers = _build_trajectories(markers_group)
                    joint_df = _build_joint_angles(trial_group)

                    labels = np.asarray(trial_group["Label"][:], dtype=float).reshape(-1)
                    joint_time = joint_df["time"].to_numpy()
                    if labels.shape[0] != joint_time.shape[0]:
                        raise ValueError("Label length does not match joint-angle time length.")
                    label_segments = _label_segments(labels=labels, time=joint_time)

                    traj_df.to_csv(out_traj, index=False)
                    events, events_meta = _detect_events(out_traj)
                    _write_yaml(out_point_events, events)
                    joint_df.to_csv(out_joint, index=False)
                    _write_yaml(out_events, events)

                    if save_imu:
                        imu_df = _build_imu(trial_group)
                        imu_df.to_csv(out_imu, index=False)
                    if save_emg:
                        emg_df = _build_emg(trial_group)
                        emg_df.to_csv(out_emg, index=False)

                    info = _build_info_payload(
                        subject=subject,
                        day_key=day_key,
                        trial_key=trial_key,
                        source_path=h5_path,
                        source_meta=meta,
                        joint_df=joint_df,
                        traj_df=traj_df,
                        labels=labels,
                        label_segments=label_segments,
                        events_meta=events_meta,
                        out_names={
                            "traj": out_traj.name,
                            "point_events": out_point_events.name,
                            "joint": out_joint.name,
                            "events": out_events.name,
                            "info": out_info.name,
                            "imu": out_imu.name if save_imu else None,
                            "emg": out_emg.name if save_emg else None,
                        },
                        missing_markers=missing_markers,
                        source_time_start_s=float(trial_group["Time"][0]),
                        marker_time_start_s=float(markers_group["Time"][0]),
                    )
                    _write_yaml(out_info, info)

                    row["n_marker_frames"] = int(len(traj_df))
                    row["n_joint_angle_frames"] = int(len(joint_df))
                    row["trajectory_sample_rate_hz"] = _sample_rate_hz(traj_df["time"].to_numpy())
                    row["joint_angle_sample_rate_hz"] = _sample_rate_hz(joint_df["time"].to_numpy())
                    row["n_label_values"] = int(len(np.unique(labels)))
                    row["n_segments"] = int(len(label_segments))
                    row["n_left_hs"] = int(len(events.get("l_heel_strike", [])))
                    row["n_left_to"] = int(len(events.get("l_toe_off", [])))
                    row["n_right_hs"] = int(len(events.get("r_heel_strike", [])))
                    row["n_right_to"] = int(len(events.get("r_toe_off", [])))
                    row["out_traj"] = str(out_traj)
                    row["out_point_events"] = str(out_point_events)
                    row["out_joint"] = str(out_joint)
                    row["out_events"] = str(out_events)
                    row["out_info"] = str(out_info)
                    row["out_imu"] = str(out_imu) if save_imu else ""
                    row["out_emg"] = str(out_emg) if save_emg else ""
                except Exception as exc:  # noqa: BLE001
                    row["status"] = "error"
                    row["error"] = str(exc)

                rows.append(row)
                written_trials += 1

            if trial_limit is not None and written_trials >= int(trial_limit):
                break

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MyPredict HDF5 files to a Eurobench-style layout.")
    parser.add_argument(
        "--raw-root",
        default="data/mypredict/raw",
        help="Folder containing MyPredict .hdf5 files.",
    )
    parser.add_argument(
        "--eurobench-root",
        default="data/mypredict/eurobench",
        help="Output folder for Eurobench-style files.",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Optional subject code or file stem filter, for example MP101 or mp101.",
    )
    parser.add_argument(
        "--limit-trials",
        type=int,
        default=None,
        help="Optional cap on converted trials, useful for debugging.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outputs that already exist.",
    )
    parser.add_argument(
        "--save-imu",
        action="store_true",
        help="Also export native IMU signals to *_imu.csv.",
    )
    parser.add_argument(
        "--save-emg",
        action="store_true",
        help="Also export native EMG signals to *_emg.csv.",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    eurobench_root = Path(args.eurobench_root)
    eurobench_root.mkdir(parents=True, exist_ok=True)

    h5_files = sorted(raw_root.glob("*.hdf5"))
    if args.subject:
        wanted = args.subject.strip().lower()
        h5_files = [p for p in h5_files if p.stem.lower() == wanted or p.name.lower() == f"{wanted}.hdf5"]
    if not h5_files:
        raise FileNotFoundError(f"No MyPredict .hdf5 files found under {raw_root}")

    rows: list[dict] = []
    converted_trials = 0
    for h5_path in h5_files:
        remaining = None
        if args.limit_trials is not None:
            remaining = max(int(args.limit_trials) - converted_trials, 0)
            if remaining == 0:
                break
        file_rows = convert_file(
            h5_path=h5_path,
            eurobench_root=eurobench_root,
            overwrite=args.overwrite,
            save_imu=args.save_imu,
            save_emg=args.save_emg,
            trial_limit=remaining,
        )
        rows.extend(file_rows)
        converted_trials = len(rows)
        if args.limit_trials is not None and converted_trials >= int(args.limit_trials):
            break

    log_df = pd.DataFrame(rows)
    log_path = eurobench_root / "conversion_log.csv"
    log_df.to_csv(log_path, index=False)

    ok_df = log_df[log_df["status"] == "ok"] if not log_df.empty else pd.DataFrame()
    summary = {
        "dataset": DATASET_TAG,
        "hdf5_files_seen": int(len(h5_files)),
        "trials_processed": int(len(log_df)),
        "trials_written": int(len(ok_df)),
        "trials_skipped": int((log_df["status"] == "skip_exists").sum()) if not log_df.empty else 0,
        "trials_failed": int((log_df["status"] == "error").sum()) if not log_df.empty else 0,
        "subjects_written": int(ok_df["subject"].nunique()) if not ok_df.empty else 0,
        "trajectory_sample_rate_hz": {
            "min": float(ok_df["trajectory_sample_rate_hz"].min()) if not ok_df.empty else None,
            "max": float(ok_df["trajectory_sample_rate_hz"].max()) if not ok_df.empty else None,
        },
        "joint_angle_sample_rate_hz": {
            "min": float(ok_df["joint_angle_sample_rate_hz"].min()) if not ok_df.empty else None,
            "max": float(ok_df["joint_angle_sample_rate_hz"].max()) if not ok_df.empty else None,
        },
        "save_imu": bool(args.save_imu),
        "save_emg": bool(args.save_emg),
        "log_file": str(log_path),
    }
    summary_path = eurobench_root / "conversion_summary.yaml"
    _write_yaml(summary_path, summary)

    print(f"hdf5_files_seen={summary['hdf5_files_seen']}")
    print(f"trials_processed={summary['trials_processed']}")
    print(f"trials_written={summary['trials_written']}")
    print(f"trials_skipped={summary['trials_skipped']}")
    print(f"trials_failed={summary['trials_failed']}")
    print(f"log={log_path}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
