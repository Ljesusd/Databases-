import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from segment_gait_cycle import _load_events, _pick_cycle_times


def _infer_sagittal_axes(pelv: np.ndarray) -> tuple[int, int]:
    ranges = pelv.max(axis=0) - pelv.min(axis=0)
    forward_idx = int(np.argmax(ranges))
    lateral_idx = int(np.argmin(ranges))
    vertical_idx = int({0, 1, 2}.difference({forward_idx, lateral_idx}).pop())
    return forward_idx, vertical_idx


def _project(vec: np.ndarray, forward_idx: int, vertical_idx: int) -> np.ndarray:
    return np.stack([vec[:, forward_idx], vec[:, vertical_idx]], axis=1)


def _angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return np.degrees(np.arctan2(cross, dot))


def _angle_between_3d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cosang = np.clip(num / den, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def _ensure_positive(angle: np.ndarray) -> np.ndarray:
    if np.nanmedian(angle) < 0:
        return -angle
    return angle


def _normalize_1d(time_seg: np.ndarray, values: np.ndarray, n_points: int) -> np.ndarray:
    t_norm = (time_seg - time_seg[0]) / (time_seg[-1] - time_seg[0])
    t_target = np.linspace(0.0, 1.0, n_points)
    return np.interp(t_target, t_norm, values)


def _pick_cycle_times_from_knee(
    time: np.ndarray, knee: np.ndarray, min_interval_s: float = 0.8
) -> tuple[float, float]:
    if knee.size < 3:
        raise ValueError("Not enough knee samples for cycle detection.")
    idx = np.where((knee[1:-1] < knee[:-2]) & (knee[1:-1] <= knee[2:]))[0] + 1
    if idx.size < 2:
        raise ValueError("Not enough knee minima to define a gait cycle.")
    idx = np.sort(idx)

    best_start = None
    best_end = None
    best_range = -np.inf

    for i, start in enumerate(idx[:-1]):
        end = None
        for j in idx[i + 1 :]:
            if time[j] - time[start] >= min_interval_s:
                end = j
                break
        if end is None:
            continue
        seg = knee[(time >= time[start]) & (time <= time[end])]
        curr_range = float(seg.max() - seg.min())
        if curr_range > best_range:
            best_range = curr_range
            best_start = start
            best_end = end

    if best_start is None or best_end is None:
        raise ValueError("Unable to find a suitable knee cycle window.")
    return float(time[best_start]), float(time[best_end])


def compute_marker_angles(
    csv_path: str,
    forward_idx: int | None = None,
    vertical_idx: int | None = None,
    angle_mode: str = "3d",
    hip_relative: bool = False,
    ankle_zero_90: bool = False,
    hip_absolute: bool = False,
    hip_sagittal: bool = False,
):
    df = pd.read_csv(csv_path)
    if "time" in df.columns:
        time = df["time"].values
    else:
        time = np.arange(len(df), dtype=float)

    pelv = df[["PELV_x", "PELV_y", "PELV_z"]].values
    rthi = df[["RTHI_x", "RTHI_y", "RTHI_z"]].values
    rkne = df[["RKNE_x", "RKNE_y", "RKNE_z"]].values
    rank = df[["RANK_x", "RANK_y", "RANK_z"]].values
    rtoe = df[["RTOE_x", "RTOE_y", "RTOE_z"]].values

    if forward_idx is None or vertical_idx is None:
        forward_idx, vertical_idx = _infer_sagittal_axes(pelv)

    # Vectores de segmentos (apuntan proximal -> distal salvo donde se invierte explícito):
    thigh = rkne - rthi          # trocánter -> rodilla
    shank = rank - rkne          # rodilla -> tobillo
    foot = rtoe - rank           # tobillo -> punta
    pelvis_vec = rthi - pelv     # centro pelvis -> trocánter (para modo relativo)

    # Alternativas “hacia arriba” para definiciones absolutas
    thigh_up = rthi - rkne       # rodilla -> trocánter
    shank_up = rkne - rank       # tobillo -> rodilla

    # Índice lateral para proyección sagital si forward/vertical están definidos
    lateral_idx = None
    if forward_idx is not None and vertical_idx is not None:
        lateral_idx = int({0, 1, 2}.difference({forward_idx, vertical_idx}).pop())

    if angle_mode == "2d":
        if forward_idx is None or vertical_idx is None:
            forward_idx, vertical_idx = _infer_sagittal_axes(pelv)
        thigh_2d = _project(thigh, forward_idx, vertical_idx)
        shank_2d = _project(shank, forward_idx, vertical_idx)
        foot_2d = _project(foot, forward_idx, vertical_idx)

        vertical_down = np.tile([0.0, -1.0], (thigh_2d.shape[0], 1))
        if hip_absolute:
            vertical_up = np.tile([0.0, 1.0], (thigh_2d.shape[0], 1))
            thigh_up = -thigh_2d  # rodilla->cadera para apuntar arriba
            hip = _ensure_positive(_angle_between(vertical_up, thigh_up))
        elif hip_relative:
            pelvis_2d = _project(pelvis_vec, forward_idx, vertical_idx)
            hip = _ensure_positive(_angle_between(pelvis_2d, thigh_2d))
        else:
            hip = _ensure_positive(_angle_between(vertical_down, thigh_2d))
        knee = _ensure_positive(_angle_between(thigh_2d, shank_2d))
        ankle = _ensure_positive(_angle_between(shank_2d, foot_2d))
    else:
        if vertical_idx is None:
            _, vertical_idx = _infer_sagittal_axes(pelv)
        vertical_up = np.zeros((thigh.shape[0], 3))
        vertical_up[:, vertical_idx] = 1.0
        vertical_down = -vertical_up

        if hip_absolute:
            hip = _ensure_positive(_angle_between_3d(vertical_up, thigh_up))
        elif hip_relative:
            hip = _ensure_positive(_angle_between_3d(pelvis_vec, thigh))
        elif hip_sagittal and lateral_idx is not None:
            thigh_sag = thigh.copy()
            thigh_sag[:, lateral_idx] = 0.0
            hip = np.degrees(np.arctan2(thigh_sag[:, forward_idx], thigh_sag[:, vertical_idx]))
        else:
            hip = _ensure_positive(_angle_between_3d(vertical_down, thigh))

        knee = _ensure_positive(_angle_between_3d(thigh, shank))

        ankle_raw = _ensure_positive(_angle_between_3d(shank, foot))
        ankle = ankle_raw

    if ankle_zero_90:
        # Re-referenciar tobillo: pie neutro = 0°, dorsiflexión (+), plantarflexión (-).
        ankle = ankle - 90.0

    return time, hip, knee, ankle


def segment_and_normalize_marker_angles(
    csv_path: str,
    events_yaml_path: str,
    n_points: int = 101,
    forward_idx: int | None = None,
    vertical_idx: int | None = None,
    angle_mode: str = "3d",
    cycle_mode: str = "events",
    hip_relative: bool = False,
    ankle_zero_90: bool = False,
    hip_absolute: bool = False,
    hip_sagittal: bool = False,
):
    time, hip, knee, ankle = compute_marker_angles(
        csv_path,
        forward_idx=forward_idx,
        vertical_idx=vertical_idx,
        angle_mode=angle_mode,
        hip_relative=hip_relative,
        ankle_zero_90=ankle_zero_90,
        hip_absolute=hip_absolute,
        hip_sagittal=hip_sagittal,
    )

    if cycle_mode == "knee_min":
        start_t, end_t = _pick_cycle_times_from_knee(time, knee)
    else:
        events = _load_events(events_yaml_path)
        start_t, end_t = _pick_cycle_times(events)

    mask = (time >= start_t) & (time <= end_t)
    time_seg = time[mask]
    if time_seg.size < 2:
        raise ValueError("Not enough samples inside the selected gait cycle window.")

    pct = np.linspace(0.0, 100.0, n_points)
    hip_norm = _normalize_1d(time_seg, hip[mask], n_points)
    knee_norm = _normalize_1d(time_seg, knee[mask], n_points)
    ankle_norm = _normalize_1d(time_seg, ankle[mask], n_points)

    return pct, hip_norm, knee_norm, ankle_norm


def save_marker_angles(out_dir: Path, basename: str, pct, hip, knee, ankle):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "pct": pct,
            "hip_flexion": hip,
            "knee_flexion": knee,
            "ankle_dorsiflexion": ankle,
        }
    )
    df.to_csv(out_dir / f"{basename}_marker_angles_norm101.csv", index=False)

    np.savez(
        out_dir / f"{basename}_marker_angles_norm101.npz",
        pct=pct,
        hip_flexion=hip,
        knee_flexion=knee,
        ankle_dorsiflexion=ankle,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input trajectories CSV")
    parser.add_argument("--events", required=True, help="Gait events YAML")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--basename", required=True, help="Output basename")
    parser.add_argument("--forward-idx", type=int, help="Forward axis index (0,1,2)")
    parser.add_argument("--vertical-idx", type=int, help="Vertical axis index (0,1,2)")
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
    parser.add_argument(
        "--ankle-zero-90",
        action="store_true",
        help="Restar 90° al tobillo para que 0 sea pie neutro (dorsi +, plantar -)",
    )
    parser.add_argument(
        "--hip-absolute",
        action="store_true",
        help="Hip flexion respecto a vertical global (rodilla->cadera vs vertical up)",
    )
    parser.add_argument(
        "--hip-sagittal",
        action="store_true",
        help="Project thigh to sagittal plane and compute flexion with atan2 (plane-stable)",
    )
    args = parser.parse_args()

    pct, hip, knee, ankle = segment_and_normalize_marker_angles(
        args.csv,
        args.events,
        forward_idx=args.forward_idx,
        vertical_idx=args.vertical_idx,
        angle_mode=args.angle_mode,
        cycle_mode=args.cycle_mode,
        ankle_zero_90=args.ankle_zero_90,
        hip_absolute=args.hip_absolute,
        hip_sagittal=args.hip_sagittal,
    )
    save_marker_angles(Path(args.out_dir), args.basename, pct, hip, knee, ankle)
    print(Path(args.out_dir) / f"{args.basename}_marker_angles_norm101.csv")


if __name__ == "__main__":
    main()
