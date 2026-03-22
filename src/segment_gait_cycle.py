from pathlib import Path
import sys
import warnings

import numpy as np
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from datasets.healthypig.extract_landmarks import extract_landmarks


def save_normalized_outputs(out_dir, basename, pct, data):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = np.column_stack(
        [
            pct,
            data["hip"],
            data["knee"],
            data["ankle"],
        ]
    )
    df = np.asarray(df)
    columns = [
        "pct",
        "RTHI_x",
        "RTHI_y",
        "RTHI_z",
        "RKNE_x",
        "RKNE_y",
        "RKNE_z",
        "RANK_x",
        "RANK_y",
        "RANK_z",
    ]
    import pandas as pd

    pd.DataFrame(df, columns=columns).to_csv(
        out_dir / f"{basename}_norm101.csv", index=False
    )

    np.savez(
        out_dir / f"{basename}_norm101.npz",
        pct=pct,
        hip=data["hip"],
        knee=data["knee"],
        ankle=data["ankle"],
    )


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


def segment_and_normalize(
    csv_path: str,
    events_yaml_path: str,
    landmarks=("hip", "knee", "ankle"),
    n_points: int = 101,
):
    """
    Returns normalized trajectories for one gait cycle (right side).
    """
    events = _load_events(events_yaml_path)
    start_t, end_t = _pick_cycle_times(events)

    time, data = extract_landmarks(csv_path, landmarks=landmarks)

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

    time_norm = np.linspace(0.0, 100.0, n_points)
    data_norm = {}
    for lm in landmarks:
        seg = data[lm][mask]
        data_norm[lm] = _normalize_segment(time_seg, seg, n_points)

    return time_norm, data_norm


def main():
    csv_path = "data/_test_eurobench/SUBJ1 (0)_Trajectories.csv"
    events_yaml = "data/_test_eurobench/SUBJ1 (0)_gaitEvents.yaml"
    time_norm, data_norm = segment_and_normalize(csv_path, events_yaml)
    save_normalized_outputs(
        out_dir="data/HealthyPiG/138_HealthyPiG/processed/SUBJ01",
        basename="SUBJ1_0",
        pct=time_norm,
        data=data_norm,
    )
    print(time_norm.shape)
    print(data_norm["hip"].shape)
    print(data_norm["knee"].shape)
    print(data_norm["ankle"].shape)


if __name__ == "__main__":
    main()
