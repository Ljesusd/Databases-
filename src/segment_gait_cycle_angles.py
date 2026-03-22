from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))


ANGLE_COLUMNS = {
    "hip": ["RHipAngles_x", "RHipAngles_y", "RHipAngles_z"],
    "knee": ["RKneeAngles_x", "RKneeAngles_y", "RKneeAngles_z"],
    "ankle": ["RAnkleAngles_x", "RAnkleAngles_y", "RAnkleAngles_z"],
}


def _load_events(events_yaml_path: str) -> dict:
    with open(events_yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Events YAML must contain a dictionary of event lists.")
    return data


def _pick_cycle_times(events: dict) -> tuple[float, float]:
    hs = np.array(events.get("r_heel_strike", []) or [], dtype=float)
    to = np.array(events.get("r_toe_off", []) or [], dtype=float)

    if hs.size >= 2:
        hs = np.sort(hs)
        return float(hs[0]), float(hs[1])

    if hs.size == 1:
        hs_time = float(hs[0])
        if to.size == 0:
            raise ValueError("Only one heel strike found and no toe off available.")
        to = np.sort(to)
        to_after = to[to > hs_time]
        if to_after.size == 0:
            raise ValueError("Toe off occurs before heel strike; cannot simulate cycle.")
        to_time = float(to_after[0])
        stride = 2.0 * (to_time - hs_time)
        if stride <= 0:
            raise ValueError("Invalid stride duration derived from heel strike and toe off.")
        return hs_time, hs_time + stride

    raise ValueError("No right heel strikes found in events.")


def _normalize_segment(time_seg: np.ndarray, data_seg: np.ndarray, n_points: int) -> np.ndarray:
    if time_seg.size < 2:
        raise ValueError("Segment must contain at least two samples.")
    if time_seg[-1] == time_seg[0]:
        raise ValueError("Segment duration is zero; cannot normalize.")

    t_norm = (time_seg - time_seg[0]) / (time_seg[-1] - time_seg[0])
    t_target = np.linspace(0.0, 1.0, n_points)

    out = np.zeros((n_points, data_seg.shape[1]))
    for i in range(data_seg.shape[1]):
        out[:, i] = np.interp(t_target, t_norm, data_seg[:, i])
    return out


def extract_angles(csv_path, angles=("hip", "knee", "ankle"), angle_scale=None):
    df = pd.read_csv(csv_path)
    time = df["time"].values
    data = {}
    for ang in angles:
        cols = ANGLE_COLUMNS[ang]
        mat = df[cols].values
        if angle_scale is not None:
            mat = mat * angle_scale
        data[ang] = mat
    return time, data


def _pick_cycle_times_from_knee(
    time: np.ndarray, knee: np.ndarray, min_interval_s: float = 0.8
) -> tuple[float, float]:
    if knee.size < 3:
        raise ValueError("Not enough knee samples for cycle detection.")
    idx = np.where((knee[1:-1] < knee[:-2]) & (knee[1:-1] <= knee[2:]))[0] + 1
    if idx.size < 2:
        raise ValueError("Not enough knee minima to define a gait cycle.")
    idx = np.sort(idx)
    start = idx[0]
    end = None
    for j in idx[1:]:
        if time[j] - time[start] >= min_interval_s:
            end = j
            break
    if end is None and idx.size >= 3:
        end = idx[2]
    if end is None:
        raise ValueError("Unable to find a suitable knee cycle window.")
    return float(time[start]), float(time[end])


def _knee_peak_fraction(knee_seg: np.ndarray) -> float | None:
    if knee_seg.size < 2:
        return None
    peak_idx = int(np.argmax(knee_seg))
    return peak_idx / float(knee_seg.size - 1)


def segment_and_normalize_angles(
    csv_path: str,
    events_yaml_path: str,
    angles=("hip", "knee", "ankle"),
    n_points: int = 101,
    angle_scale: float | None = None,
    cycle_mode: str = "events",
):
    time, data = extract_angles(csv_path, angles=angles, angle_scale=angle_scale)
    if cycle_mode == "knee_min":
        start_t, end_t = _pick_cycle_times_from_knee(time, data["knee"][:, 0])
    elif cycle_mode == "auto":
        try:
            events = _load_events(events_yaml_path)
            start_t, end_t = _pick_cycle_times(events)
            mask = (time >= start_t) & (time <= end_t)
            knee_seg = data["knee"][mask][:, 0]
            peak_frac = _knee_peak_fraction(knee_seg)
            if peak_frac is None or not (0.45 <= peak_frac <= 0.9):
                raise ValueError("Knee peak outside expected window; falling back to knee_min.")
        except (ValueError, KeyError) as exc:
            warnings.warn(
                f"Using knee_min cycle selection for angles: {exc}",
                RuntimeWarning,
            )
            start_t, end_t = _pick_cycle_times_from_knee(time, data["knee"][:, 0])
    else:
        events = _load_events(events_yaml_path)
        start_t, end_t = _pick_cycle_times(events)

    if end_t > time[-1]:
        warnings.warn(
            "Simulated end time exceeds data range; clamping to last sample.",
            RuntimeWarning,
        )
        end_t = time[-1]

    mask = (time >= start_t) & (time <= end_t)
    time_seg = time[mask]

    if time_seg.size < 2:
        raise ValueError("Not enough samples inside the selected gait cycle window.")

    pct = np.linspace(0.0, 100.0, n_points)
    data_norm = {}
    for ang in angles:
        seg = data[ang][mask]
        data_norm[ang] = _normalize_segment(time_seg, seg, n_points)

    return pct, data_norm


def save_normalized_angles(out_dir, basename, pct, data):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "pct": pct,
            "RHipAngles_x": data["hip"][:, 0],
            "RHipAngles_y": data["hip"][:, 1],
            "RHipAngles_z": data["hip"][:, 2],
            "RKneeAngles_x": data["knee"][:, 0],
            "RKneeAngles_y": data["knee"][:, 1],
            "RKneeAngles_z": data["knee"][:, 2],
            "RAnkleAngles_x": data["ankle"][:, 0],
            "RAnkleAngles_y": data["ankle"][:, 1],
            "RAnkleAngles_z": data["ankle"][:, 2],
        }
    )
    df.to_csv(out_dir / f"{basename}_angles_norm101.csv", index=False)

    np.savez(
        out_dir / f"{basename}_angles_norm101.npz",
        pct=pct,
        hip=data["hip"],
        knee=data["knee"],
        ankle=data["ankle"],
    )


def save_flexion_outputs(out_dir, basename, pct, data):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "pct": pct,
            "hip_flexion": data["hip"][:, 0],
            "knee_flexion": data["knee"][:, 0],
            "ankle_dorsiflexion": data["ankle"][:, 0],
        }
    )
    df.to_csv(out_dir / f"{basename}_flexion_norm101.csv", index=False)

    np.savez(
        out_dir / f"{basename}_flexion_norm101.npz",
        pct=pct,
        hip_flexion=data["hip"][:, 0],
        knee_flexion=data["knee"][:, 0],
        ankle_dorsiflexion=data["ankle"][:, 0],
    )


def main():
    csv_path = "data/HealthyPiG/138_HealthyPiG/eurobench/SUBJ01/SUBJ1 (0)_jointAngles.csv"
    events_yaml = "data/HealthyPiG/138_HealthyPiG/eurobench/SUBJ01/SUBJ1 (0)_gaitEvents.yaml"
    pct, data_norm = segment_and_normalize_angles(csv_path, events_yaml)
    save_normalized_angles(
        out_dir="data/HealthyPiG/138_HealthyPiG/processed/SUBJ01",
        basename="SUBJ1_0",
        pct=pct,
        data=data_norm,
    )


if __name__ == "__main__":
    main()
