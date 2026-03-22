import argparse
import math
import re
import sys
import zipfile
from collections import Counter, defaultdict
from io import TextIOWrapper
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from segment_gait_cycle_marker_angles import segment_and_normalize_marker_angles


WALK_TASKS = {"Gait", "FastGait", "SlowGait", "2minWalk"}
TASK_REGEX = re.compile(r"P\d+_S\d+_(?P<task>[^_]+)_(?P<run>\d+)_Trajectories\.csv$")
BENCHMARK_LEVEL_WALKING_MODE = 1
DEFAULT_N_POINTS = 101


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML payload in {path}")
    return payload


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return np.column_stack([pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float) for col in cols])


def _smooth(curve: np.ndarray, window: int = 5) -> np.ndarray:
    return pd.Series(np.asarray(curve, dtype=float)).rolling(
        window=window,
        center=True,
        min_periods=1,
    ).mean().to_numpy()


def _zscore(curve: np.ndarray) -> np.ndarray:
    curve = np.asarray(curve, dtype=float)
    std = float(np.nanstd(curve))
    if not np.isfinite(std) or std < 1e-6:
        return np.zeros_like(curve)
    return (curve - float(np.nanmean(curve))) / std


def _shape_distance(curve: np.ndarray, template: np.ndarray) -> float:
    curve_s = _zscore(_smooth(curve))
    templ_s = _zscore(template)
    rmse = float(np.sqrt(np.nanmean((curve_s - templ_s) ** 2)))
    corr = np.corrcoef(curve_s, templ_s)[0, 1]
    if not np.isfinite(corr):
        corr = 0.0
    return 0.65 * rmse + 0.35 * (1.0 - corr)


def _interp_template(knots: list[tuple[float, float]], pct: np.ndarray) -> np.ndarray:
    x = np.asarray([point[0] for point in knots], dtype=float)
    y = np.asarray([point[1] for point in knots], dtype=float)
    return _smooth(np.interp(pct, x, y), window=5)


def _canonical_templates(pct: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "hip": _interp_template(
            [
                (0, 30),
                (10, 28),
                (20, 18),
                (35, 5),
                (50, -8),
                (60, -12),
                (68, -6),
                (78, 10),
                (90, 24),
                (100, 30),
            ],
            pct,
        ),
        "knee": _interp_template(
            [
                (0, 5),
                (8, 15),
                (18, 8),
                (30, 2),
                (45, 3),
                (58, 15),
                (68, 45),
                (73, 60),
                (82, 35),
                (92, 10),
                (100, 5),
            ],
            pct,
        ),
        "ankle": _interp_template(
            [
                (0, 0),
                (8, -6),
                (18, -2),
                (35, 8),
                (50, 10),
                (60, 0),
                (67, -18),
                (80, -2),
                (90, 2),
                (100, 0),
            ],
            pct,
        ),
    }


def _normalize_matrix(time_seg: np.ndarray, values: np.ndarray, n_points: int) -> np.ndarray:
    if time_seg.size < 2:
        raise ValueError("Segment must contain at least two samples.")
    dt = float(time_seg[-1] - time_seg[0])
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Segment duration must be positive.")
    t_norm = (time_seg - time_seg[0]) / dt
    target = np.linspace(0.0, 1.0, n_points)
    out = np.empty((n_points, values.shape[1]), dtype=float)
    for idx in range(values.shape[1]):
        out[:, idx] = np.interp(target, t_norm, values[:, idx])
    return out


def _normalize_vector(time_seg: np.ndarray, values: np.ndarray, n_points: int) -> np.ndarray:
    return _normalize_matrix(time_seg, np.asarray(values, dtype=float)[:, None], n_points)[:, 0]


def _wrap180(curve: np.ndarray) -> np.ndarray:
    curve = np.asarray(curve, dtype=float)
    return (curve + 180.0) % 360.0 - 180.0


def _unwrap_deg(curve: np.ndarray) -> np.ndarray:
    return np.degrees(np.unwrap(np.radians(np.asarray(curve, dtype=float))))


def _recenter_curve(curve: np.ndarray, center_step: float = 180.0) -> np.ndarray:
    curve = np.asarray(curve, dtype=float)
    median = float(np.nanmedian(curve))
    shift = round(median / center_step) * center_step
    return curve - shift


def _curve_variants(curve: np.ndarray) -> dict[str, np.ndarray]:
    base = np.asarray(curve, dtype=float)
    wrapped = _wrap180(base)
    unwrapped = _unwrap_deg(base)
    return {
        "raw": base,
        "wrap180": wrapped,
        "wrap180_center": _recenter_curve(wrapped),
        "unwrap": unwrapped,
        "unwrap_center": _recenter_curve(unwrapped),
    }


def _score_joint_curve(joint_name: str, curve: np.ndarray, template: np.ndarray) -> tuple[float, dict]:
    dist = _shape_distance(curve, template)
    rom = float(np.ptp(curve))
    max_step = float(np.max(np.abs(np.diff(curve)))) if curve.size >= 2 else 0.0
    peak_idx = int(np.argmax(curve))
    trough_idx = int(np.argmin(curve))

    if joint_name == "hip":
        penalty = 0.0
        penalty += max(0.0, 18.0 - rom) / 5.0
        penalty += max(0.0, rom - 70.0) / 4.0
        penalty += max(0.0, max_step - 20.0) / 2.0
        penalty += abs(curve[0] - 30.0) / 9.0
        penalty += abs(curve[55] + 12.0) / 9.0
        penalty += abs(curve[-1] - 30.0) / 9.0
        penalty += abs(trough_idx - 58) / 8.0
        score = dist + 0.18 * penalty
    elif joint_name == "knee":
        penalty = 0.0
        penalty += max(0.0, 25.0 - rom) / 5.0
        penalty += max(0.0, rom - 90.0) / 5.0
        penalty += max(0.0, max_step - 12.0) / 2.0
        penalty += abs(curve[0] - 5.0) / 7.0
        penalty += abs(curve[10] - 15.0) / 7.0
        penalty += abs(curve[30] - 2.0) / 7.0
        penalty += abs(curve[72] - 55.0) / 9.0
        penalty += abs(curve[-1] - 5.0) / 7.0
        penalty += abs(peak_idx - 72) / 8.0
        score = dist + 0.16 * penalty
    elif joint_name == "ankle":
        penalty = 0.0
        penalty += max(0.0, 12.0 - rom) / 4.0
        penalty += max(0.0, rom - 45.0) / 4.0
        penalty += max(0.0, max_step - 10.0) / 2.0
        penalty += abs(curve[0] - 0.0) / 6.0
        penalty += abs(curve[10] + 6.0) / 6.0
        penalty += abs(curve[45] - 10.0) / 6.0
        penalty += abs(curve[67] + 18.0) / 7.0
        penalty += abs(curve[-1] - 0.0) / 6.0
        penalty += abs(trough_idx - 67) / 8.0
        score = dist + 0.16 * penalty
    else:
        raise ValueError(f"Unsupported joint '{joint_name}'")

    return score, {
        "shape_distance": float(dist),
        "rom_deg": float(rom),
        "max_step_deg": float(max_step),
        "peak_pct": float(peak_idx),
        "trough_pct": float(trough_idx),
    }


def _angle_between_3d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    den = np.where(den == 0.0, np.nan, den)
    cosang = np.clip(num / den, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def _angle_between_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return np.degrees(np.arctan2(cross, dot))


def _project_sagittal(vec: np.ndarray) -> np.ndarray:
    return np.stack([vec[:, 0], vec[:, 2]], axis=1)


def _available_markers(df: pd.DataFrame) -> list[str]:
    return sorted({col[:-2] for col in df.columns if col.endswith("_x")})


def _extract_marker_data(df: pd.DataFrame) -> dict[str, np.ndarray]:
    markers = _available_markers(df)
    marker_data = {
        marker: _safe_numeric(df, [f"{marker}_x", f"{marker}_y", f"{marker}_z"])
        for marker in markers
    }
    if "PELV" not in marker_data:
        pelvis_parts = [marker_data[name] for name in ["LASI", "RASI", "LPSI", "RPSI"] if name in marker_data]
        if pelvis_parts:
            marker_data["PELV"] = np.nanmean(np.stack(pelvis_parts, axis=0), axis=0)
    return marker_data


def _mean_existing(marker_data: dict[str, np.ndarray], names: list[str]) -> np.ndarray | None:
    parts = [marker_data[name] for name in names if name in marker_data]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return np.nanmean(np.stack(parts, axis=0), axis=0)


def _first_existing(marker_data: dict[str, np.ndarray], names: list[str]) -> np.ndarray | None:
    for name in names:
        if name in marker_data:
            return marker_data[name]
    return None


def _standardize_cycle_frame(marker_data: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict]:
    pelv = marker_data.get("PELV")
    if pelv is None:
        pelv = _mean_existing(marker_data, ["LASI", "RASI", "LPSI", "RPSI"])
    if pelv is None:
        raise KeyError("Missing pelvis markers required for frame alignment.")

    forward_candidates = [pelv]
    heel = _first_existing(marker_data, ["RHEE", "LHEE", "RANK", "LANK"])
    if heel is not None:
        forward_candidates.append(heel)

    best_idx = 0
    best_net = 0.0
    for axis in [0, 1]:
        net = 0.0
        for arr in forward_candidates:
            val = float(arr[-1, axis] - arr[0, axis])
            if abs(val) > abs(best_net if axis == best_idx else net):
                net = val
        if abs(net) > abs(best_net):
            best_idx = axis
            best_net = net
    forward_idx = int(best_idx)
    lateral_idx = 1 if forward_idx == 0 else 0
    vertical_idx = 2

    right_lateral = _mean_existing(marker_data, ["RASI", "RPSI", "RTHI", "RANK", "RHEE"])
    left_lateral = _mean_existing(marker_data, ["LASI", "LPSI", "LTHI", "LANK", "LHEE"])
    lateral_span = 1.0
    if right_lateral is not None and left_lateral is not None:
        lateral_span = float(np.nanmedian(right_lateral[:, lateral_idx] - left_lateral[:, lateral_idx]))

    upper = _first_existing(marker_data, ["C7", "CLAV", "STRN", "T10", "RBHD", "LFHD"])
    vertical_span = 1.0
    if upper is not None:
        vertical_span = float(np.nanmedian(upper[:, vertical_idx] - pelv[:, vertical_idx]))

    signs = np.array(
        [
            1.0 if best_net >= 0.0 else -1.0,
            1.0 if lateral_span >= 0.0 else -1.0,
            1.0 if vertical_span >= 0.0 else -1.0,
        ],
        dtype=float,
    )
    order = np.array([forward_idx, lateral_idx, vertical_idx], dtype=int)
    origin = pelv[0, order] * signs

    standardized: dict[str, np.ndarray] = {}
    for marker, values in marker_data.items():
        standardized[marker] = values[:, order] * signs - origin

    frame_meta = {
        "source_axis_order": {
            "x": int(forward_idx),
            "y": int(lateral_idx),
            "z": int(vertical_idx),
        },
        "sign_flips": {
            "x": int(signs[0]),
            "y": int(signs[1]),
            "z": int(signs[2]),
        },
    }
    return standardized, frame_meta


def _cycle_windows(
    events: dict,
    min_stride_s: float,
    max_stride_s: float,
    require_toe_off: bool,
    heel_key: str = "r_heel_strike",
    toe_key: str = "r_toe_off",
) -> tuple[list[dict], dict[str, int]]:
    hs = np.sort(np.asarray(events.get(heel_key, []) or [], dtype=float))
    toe = np.sort(np.asarray(events.get(toe_key, []) or [], dtype=float))

    windows: list[dict] = []
    stats = {
        "pairs_seen": 0,
        "pairs_outside_stride_range": 0,
        "pairs_without_toe_off": 0,
        "pairs_kept": 0,
    }
    if hs.size < 2:
        return windows, stats

    for cycle_idx, (start_t, end_t) in enumerate(zip(hs[:-1], hs[1:]), start=1):
        stats["pairs_seen"] += 1
        stride_time = float(end_t - start_t)
        if not (min_stride_s <= stride_time <= max_stride_s):
            stats["pairs_outside_stride_range"] += 1
            continue
        toe_inside = toe[(toe > start_t) & (toe < end_t)]
        if require_toe_off and toe_inside.size == 0:
            stats["pairs_without_toe_off"] += 1
            continue
        windows.append(
            {
                "cycle_idx": int(cycle_idx),
                "start_time_s": float(start_t),
                "end_time_s": float(end_t),
                "stride_time_s": stride_time,
                "toe_off_count": int(toe_inside.size),
            }
        )
        stats["pairs_kept"] += 1
    return windows, stats


def _align_events_to_trial_time(time: np.ndarray, events: dict[str, list[float]]) -> dict[str, list[float]]:
    finite_time = time[np.isfinite(time)]
    if finite_time.size == 0:
        return events

    t_min = float(finite_time.min())
    t_max = float(finite_time.max())
    all_events = [value for values in events.values() for value in values if np.isfinite(value)]
    if not all_events:
        return events

    def _ratio_in_range(values: list[float]) -> float:
        if not values:
            return 0.0
        count = sum(1 for value in values if t_min <= value <= t_max)
        return float(count) / float(len(values))

    orig_ratio = _ratio_in_range(all_events)
    offset = min(all_events) - t_min
    shifted = {key: [float(value - offset) for value in values] for key, values in events.items()}
    shifted_values = [value for values in shifted.values() for value in values]
    shifted_ratio = _ratio_in_range(shifted_values)
    if shifted_ratio > orig_ratio + 0.2:
        return shifted
    return events


def _build_right_joint_candidates(marker_data: dict[str, np.ndarray]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {"hip": [], "knee": [], "ankle": []}

    knee_center = _mean_existing(marker_data, ["RKJC", "RKNE", "RKNM"])
    ankle_center = _mean_existing(marker_data, ["RAJC", "RANK", "RMED"])
    toe = _mean_existing(marker_data, ["RTOE", "RFMH"])
    heel = _first_existing(marker_data, ["RHEE"])

    hip_anchor_defs = [
        ("rpelv", _mean_existing(marker_data, ["RASI", "RPSI"])),
        ("midASIS", _first_existing(marker_data, ["midASIS"])),
        ("pelv", _first_existing(marker_data, ["PELV"])),
        ("rasi", _first_existing(marker_data, ["RASI"])),
        ("rpsi", _first_existing(marker_data, ["RPSI"])),
    ]
    distal_defs = [
        ("rthi", _first_existing(marker_data, ["RTHI"])),
        ("rknee", knee_center),
    ]

    for prox_name, prox in hip_anchor_defs:
        if prox is None:
            continue
        for dist_name, dist in distal_defs:
            if dist is None:
                continue
            vec = dist - prox
            curve = np.degrees(np.arctan2(vec[:, 0], -vec[:, 2]))
            out["hip"].append({"source": f"{prox_name}_to_{dist_name}", "curve": curve})

    if knee_center is not None and ankle_center is not None:
        shank = ankle_center - knee_center
        thigh_defs = [
            ("rthi_to_rknee", None),
            ("rpelv_to_rknee", _mean_existing(marker_data, ["RASI", "RPSI"])),
            ("pelv_to_rknee", _first_existing(marker_data, ["PELV"])),
        ]
        thigh_marker = _first_existing(marker_data, ["RTHI"])
        if thigh_marker is not None:
            thigh = knee_center - thigh_marker
            out["knee"].append({"source": "angle3d_rthi", "curve": _angle_between_3d(thigh, shank)})
            out["knee"].append(
                {"source": "angle2d_rthi", "curve": np.abs(_angle_between_2d(_project_sagittal(thigh), _project_sagittal(shank)))}
            )
        for source_name, prox in thigh_defs[1:]:
            if prox is None:
                continue
            thigh = knee_center - prox
            out["knee"].append({"source": f"angle3d_{source_name}", "curve": _angle_between_3d(thigh, shank)})
            out["knee"].append(
                {
                    "source": f"angle2d_{source_name}",
                    "curve": np.abs(_angle_between_2d(_project_sagittal(thigh), _project_sagittal(shank))),
                }
            )

        if toe is not None:
            foot_defs: list[tuple[str, np.ndarray]] = []
            if heel is not None:
                foot_defs.append(("toe_minus_heel", toe - heel))
            foot_defs.append(("toe_minus_ankle", toe - ankle_center))
            shank_up = knee_center - ankle_center
            for foot_name, foot in foot_defs:
                raw_3d = _angle_between_3d(shank_up, foot)
                raw_2d = _angle_between_2d(_project_sagittal(shank_up), _project_sagittal(foot))
                out["ankle"].append({"source": f"90_minus_3d_{foot_name}", "curve": 90.0 - raw_3d})
                out["ankle"].append({"source": f"3d_minus_90_{foot_name}", "curve": raw_3d - 90.0})
                out["ankle"].append({"source": f"angle2d_{foot_name}", "curve": raw_2d})
                out["ankle"].append({"source": f"neg_angle2d_{foot_name}", "curve": -raw_2d})

    return out


def _pick_best_joint_curve(
    joint_name: str,
    raw_candidates: list[dict],
    time_seg: np.ndarray,
    n_points: int,
    template: np.ndarray,
) -> dict:
    best: dict | None = None
    for raw_item in raw_candidates:
        raw_curve = _normalize_vector(time_seg, raw_item["curve"], n_points)
        for variant_name, variant_curve in _curve_variants(raw_curve).items():
            for sign in [1, -1]:
                signed_curve = sign * variant_curve
                offsets = {
                    "none": 0.0,
                    "start": float(signed_curve[0] - template[0]),
                    "median": float(np.nanmedian(signed_curve) - np.nanmedian(template)),
                    "mid": float(signed_curve[len(signed_curve) // 2] - template[len(template) // 2]),
                }
                for offset_name, offset in offsets.items():
                    candidate_curve = signed_curve - offset
                    score, qc = _score_joint_curve(joint_name, candidate_curve, template)
                    item = {
                        "joint": joint_name,
                        "source": raw_item["source"],
                        "variant": variant_name,
                        "sign": int(sign),
                        "offset_kind": offset_name,
                        "offset_deg": float(offset),
                        "curve": candidate_curve,
                        "score": float(score),
                        "rom_deg": float(qc["rom_deg"]),
                        "max_step_deg": float(qc["max_step_deg"]),
                        "shape_distance": float(qc["shape_distance"]),
                        "peak_pct": float(qc["peak_pct"]),
                        "trough_pct": float(qc["trough_pct"]),
                    }
                    if best is None or item["score"] < best["score"]:
                        best = item
    if best is None:
        raise ValueError(f"No valid candidate curves found for {joint_name}.")
    return best


def _pick_best_existing_curve(
    joint_name: str,
    raw_curve: np.ndarray,
    template: np.ndarray,
    source: str,
) -> dict:
    best: dict | None = None
    for variant_name, variant_curve in _curve_variants(raw_curve).items():
        for sign in [1, -1]:
            signed_curve = sign * variant_curve
            offsets = {
                "none": 0.0,
                "start": float(signed_curve[0] - template[0]),
                "median": float(np.nanmedian(signed_curve) - np.nanmedian(template)),
                "mid": float(signed_curve[len(signed_curve) // 2] - template[len(template) // 2]),
            }
            for offset_name, offset in offsets.items():
                candidate_curve = signed_curve - offset
                score, qc = _score_joint_curve(joint_name, candidate_curve, template)
                item = {
                    "joint": joint_name,
                    "source": source,
                    "variant": variant_name,
                    "sign": int(sign),
                    "offset_kind": offset_name,
                    "offset_deg": float(offset),
                    "curve": candidate_curve,
                    "score": float(score),
                    "rom_deg": float(qc["rom_deg"]),
                    "max_step_deg": float(qc["max_step_deg"]),
                    "shape_distance": float(qc["shape_distance"]),
                    "peak_pct": float(qc["peak_pct"]),
                    "trough_pct": float(qc["trough_pct"]),
                }
                if best is None or item["score"] < best["score"]:
                    best = item
    if best is None:
        raise ValueError(f"No valid precomputed curve found for {joint_name}.")
    return best


def _load_norm101_matrix(path: Path, n_points: int) -> tuple[np.ndarray, list[str], np.ndarray]:
    df = pd.read_csv(path)
    if "pct" not in df.columns:
        raise KeyError(f"Missing pct column in {path.name}")
    pct = pd.to_numeric(df["pct"], errors="coerce").to_numpy(dtype=float)
    traj_columns = [col for col in df.columns if col not in {"pct", "time"}]
    if not traj_columns:
        raise ValueError(f"No trajectory columns found in {path.name}")
    matrix = np.column_stack([pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float) for col in traj_columns])
    if pct.size != n_points or not np.allclose(pct, np.linspace(0.0, 100.0, n_points)):
        source_pct = pct.copy()
        target_pct = np.linspace(0.0, 100.0, n_points)
        resampled = np.empty((n_points, matrix.shape[1]), dtype=float)
        for col_idx in range(matrix.shape[1]):
            resampled[:, col_idx] = np.interp(target_pct, source_pct, matrix[:, col_idx])
        matrix = resampled
        pct = target_pct
    return pct, traj_columns, matrix


def _plot_angle_overlay(
    out_path: Path,
    pct: np.ndarray,
    stacks: dict[str, np.ndarray],
    medians: dict[str, np.ndarray],
    templates: dict[str, np.ndarray],
    title: str,
    subtitle: str,
) -> None:
    joints = [("ankle", "ANKLE", "ankle_dorsiflexion"), ("knee", "KNEE", "knee_flexion"), ("hip", "HIP", "hip_flexion")]
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 9.5), sharex=True, constrained_layout=True)
    legend_handles = None

    for ax, (key, panel_title, ylabel) in zip(axes, joints):
        stack = stacks.get(key)
        selected_handle = None
        if stack is not None:
            draw_stack = stack
            if draw_stack.shape[0] > 200:
                idx = np.linspace(0, draw_stack.shape[0] - 1, 200, dtype=int)
                draw_stack = draw_stack[idx]
            for curve in draw_stack:
                line = ax.plot(
                    pct,
                    curve,
                    color="#8e8e8e",
                    alpha=0.55,
                    linewidth=3.0,
                    linestyle=(0, (5, 4)),
                    zorder=1,
                )[0]
                if selected_handle is None:
                    selected_handle = line
        template_handle = ax.plot(
            pct,
            templates[key],
            color="#2f6f7e",
            linewidth=1.6,
            linestyle=(0, (8, 4)),
            alpha=0.95,
            zorder=2,
        )[0]
        median_handle = ax.plot(pct, medians[key], color="black", linewidth=2.0, zorder=3)[0]
        if legend_handles is None:
            legend_handles = (selected_handle, template_handle, median_handle)
        ax.set_title(panel_title)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.0, 100.0)
        ax.grid(alpha=0.18)
        ax.axhline(0.0, color="#6f7e86", linewidth=0.7, alpha=0.5)
    axes[-1].set_xlabel("PERCENT OF GAIT CYCLE (%)")
    if legend_handles is not None and legend_handles[0] is not None:
        fig.legend(
            legend_handles,
            ["Selected cycles", "Reference template", "Canonical median"],
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.99),
        )
    fig.suptitle(f"{title}\n{subtitle}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _multisensor_legacy_groups(
    legacy_processed_root: Path,
    eurobench_root: Path,
    summary_csv: Path,
) -> dict[str, list[dict]]:
    df = pd.read_csv(summary_csv)
    required = {"user", "trial", "kept"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"{summary_csv.name} missing columns: {missing}")

    groups: dict[str, list[dict]] = defaultdict(list)
    kept_df = df[df["kept"].fillna(False).astype(bool)].copy()
    kept_df = kept_df[~kept_df["trial"].astype(str).str.contains("_auto_")]
    for _, row in kept_df.sort_values(["user", "trial"]).iterrows():
        subject = str(row["user"])
        trial_name = str(row["trial"])
        traj_path = eurobench_root / subject / trial_name.replace("_marker_angles_norm101.csv", "_Trajectories.csv")
        events_path = eurobench_root / subject / trial_name.replace("_marker_angles_norm101.csv", "_gaitEvents.yaml")
        legacy_angles_path = legacy_processed_root / subject / trial_name
        legacy_norm_path = legacy_processed_root / subject / trial_name.replace("_marker_angles_norm101.csv", "_norm101.csv")
        if not legacy_angles_path.exists():
            continue
        groups[subject].append(
            {
                "trial_name": trial_name,
                "traj_path": traj_path,
                "events_path": events_path,
                "legacy_angles_path": legacy_angles_path,
                "legacy_norm_path": legacy_norm_path if legacy_norm_path.exists() else None,
            }
        )
    return groups


def _multisensor_fixed_process_group(
    dataset_name: str,
    group_label: str,
    basename: str,
    trial_specs: list[dict],
    out_dir: Path,
    plots_dir: Path,
    n_points: int,
    keep_percentile: float,
) -> dict:
    pct = np.linspace(0.0, 100.0, n_points)
    norm_time = np.linspace(0.0, 1.0, n_points)
    templates = _canonical_templates(pct)
    candidates: list[dict] = []
    cycle_rows: list[dict] = []
    traj_columns_ref: list[str] | None = None

    for spec in trial_specs:
        row = {
            "dataset": dataset_name,
            "group": group_label,
            "trial": spec["trial_name"],
            "selected": False,
        }
        row_index = len(cycle_rows)
        cycle_rows.append(row)
        try:
            if spec["traj_path"].exists() and spec["events_path"].exists():
                _, hip_raw, knee_raw, ankle_raw = segment_and_normalize_marker_angles(
                    str(spec["traj_path"]),
                    str(spec["events_path"]),
                    n_points=n_points,
                    angle_mode="3d",
                    cycle_mode="knee_min",
                    ankle_zero_90=True,
                    hip_sagittal=True,
                )
                source_pipeline = "fixed_3d_knee_min_hip_sagittal"
            else:
                legacy_df = pd.read_csv(spec["legacy_angles_path"])
                required = {"pct", "hip_flexion", "knee_flexion", "ankle_dorsiflexion"}
                if not required.issubset(legacy_df.columns):
                    raise KeyError(f"{spec['legacy_angles_path'].name} missing columns: {sorted(required.difference(legacy_df.columns))}")
                hip_raw = pd.to_numeric(legacy_df["hip_flexion"], errors="coerce").to_numpy(dtype=float)
                knee_raw = pd.to_numeric(legacy_df["knee_flexion"], errors="coerce").to_numpy(dtype=float)
                ankle_raw = pd.to_numeric(legacy_df["ankle_dorsiflexion"], errors="coerce").to_numpy(dtype=float)
                source_pipeline = "legacy_angles_fallback"

            hip_choice = _pick_best_existing_curve("hip", hip_raw, templates["hip"], source=source_pipeline)
            knee_choice = _pick_best_existing_curve("knee", knee_raw, templates["knee"], source=source_pipeline)
            ankle_choice = _pick_best_existing_curve("ankle", ankle_raw, templates["ankle"], source=source_pipeline)
            total_score = float(hip_choice["score"] + knee_choice["score"] + ankle_choice["score"])

            traj_matrix = None
            if spec["legacy_norm_path"] is not None:
                _, traj_columns, traj_matrix = _load_norm101_matrix(spec["legacy_norm_path"], n_points=n_points)
                if traj_columns_ref is None:
                    traj_columns_ref = traj_columns
                elif traj_columns != traj_columns_ref:
                    raise ValueError(f"Trajectory column mismatch in {spec['legacy_norm_path'].name}")

            cycle_rows[row_index].update(
                {
                    "status": "candidate",
                    "source_pipeline": source_pipeline,
                    "total_score": total_score,
                }
            )
            for joint_name, chosen in [("hip", hip_choice), ("knee", knee_choice), ("ankle", ankle_choice)]:
                cycle_rows[row_index][f"{joint_name}_source"] = chosen["source"]
                cycle_rows[row_index][f"{joint_name}_variant"] = chosen["variant"]
                cycle_rows[row_index][f"{joint_name}_sign"] = int(chosen["sign"])
                cycle_rows[row_index][f"{joint_name}_offset_kind"] = chosen["offset_kind"]
                cycle_rows[row_index][f"{joint_name}_offset_deg"] = float(chosen["offset_deg"])
                cycle_rows[row_index][f"{joint_name}_score"] = float(chosen["score"])
                cycle_rows[row_index][f"{joint_name}_rom_deg"] = float(chosen["rom_deg"])
                cycle_rows[row_index][f"{joint_name}_max_step_deg"] = float(chosen["max_step_deg"])
                cycle_rows[row_index][f"{joint_name}_shape_distance"] = float(chosen["shape_distance"])

            candidates.append(
                {
                    "row_index": row_index,
                    "traj_matrix": traj_matrix,
                    "hip_curve": hip_choice["curve"],
                    "knee_curve": knee_choice["curve"],
                    "ankle_curve": ankle_choice["curve"],
                    "score": total_score,
                    "hip_source": hip_choice["source"],
                    "hip_variant": hip_choice["variant"],
                    "hip_sign": int(hip_choice["sign"]),
                    "knee_source": knee_choice["source"],
                    "knee_variant": knee_choice["variant"],
                    "knee_sign": int(knee_choice["sign"]),
                    "ankle_source": ankle_choice["source"],
                    "ankle_variant": ankle_choice["variant"],
                    "ankle_sign": int(ankle_choice["sign"]),
                }
            )
        except Exception as exc:
            cycle_rows[row_index]["status"] = "fail"
            cycle_rows[row_index]["error"] = str(exc)

    if not candidates:
        raise RuntimeError(f"No valid candidate trials found for {group_label}")

    candidate_scores = np.asarray([candidate["score"] for candidate in candidates], dtype=float)
    score_threshold = float(np.percentile(candidate_scores, keep_percentile))
    keep_idx = np.where(candidate_scores <= score_threshold)[0]
    if keep_idx.size == 0:
        keep_idx = np.array([int(np.argmin(candidate_scores))], dtype=int)
    selected_rows = [candidates[idx] for idx in sorted(set(keep_idx.tolist()))]

    mode_info = _selection_modes(selected_rows, ["hip", "knee", "ankle"])
    mode_filtered_rows = [
        row
        for row in selected_rows
        if all(
            row[f"{joint_name}_source"] == mode_info[joint_name]["source"]
            and row[f"{joint_name}_variant"] == mode_info[joint_name]["variant"]
            and row[f"{joint_name}_sign"] == mode_info[joint_name]["sign"]
            for joint_name in mode_info
        )
    ]
    mode_filter_applied = False
    if len(selected_rows) >= 5 and len(mode_filtered_rows) >= max(3, int(math.floor(0.6 * len(selected_rows)))):
        selected_rows = mode_filtered_rows
        mode_filter_applied = True

    for row in selected_rows:
        cycle_rows[row["row_index"]]["selected"] = True
        cycle_rows[row["row_index"]]["status"] = "selected"

    hip_stack = np.stack([row["hip_curve"] for row in selected_rows], axis=0)
    knee_stack = np.stack([row["knee_curve"] for row in selected_rows], axis=0)
    ankle_stack = np.stack([row["ankle_curve"] for row in selected_rows], axis=0)
    hip_median = np.nanmedian(hip_stack, axis=0)
    knee_median = np.nanmedian(knee_stack, axis=0)
    ankle_median = np.nanmedian(ankle_stack, axis=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    traj_csv = None
    traj_stack_rows = [row["traj_matrix"] for row in selected_rows if row["traj_matrix"] is not None]
    if traj_stack_rows and traj_columns_ref is not None:
        traj_stack = np.stack(traj_stack_rows, axis=0)
        traj_median = np.nanmedian(traj_stack, axis=0)
        traj_csv = out_dir / f"{basename}_canonical_Trajectories.csv"
        traj_columns = ["time", "pct"] + traj_columns_ref
        pd.DataFrame(np.column_stack([norm_time, pct, traj_median]), columns=traj_columns).to_csv(traj_csv, index=False)

    angle_csv = out_dir / f"{basename}_canonical_marker_angles_norm101.csv"
    pd.DataFrame(
        {
            "pct": pct,
            "hip_flexion": hip_median,
            "knee_flexion": knee_median,
            "ankle_dorsiflexion": ankle_median,
        }
    ).to_csv(angle_csv, index=False)

    cycle_metrics_csv = out_dir / f"{basename}_canonical_cycle_metrics.csv"
    pd.DataFrame(cycle_rows).to_csv(cycle_metrics_csv, index=False)

    plot_path = plots_dir / f"{basename}_canonical_profiles.png"
    _plot_angle_overlay(
        plot_path,
        pct=pct,
        stacks={"hip": hip_stack, "knee": knee_stack, "ankle": ankle_stack},
        medians={"hip": hip_median, "knee": knee_median, "ankle": ankle_median},
        templates=templates,
        title=group_label,
        subtitle=f"selected_cycles={len(selected_rows)} / candidates={len(candidates)}",
    )

    summary = {
        "dataset": dataset_name,
        "group": group_label,
        "basename": basename,
        "n_trials": int(len(trial_specs)),
        "candidate_cycles": int(len(candidates)),
        "selected_cycles": int(len(selected_rows)),
        "score_threshold": float(score_threshold),
        "keep_percentile": float(keep_percentile),
        "mode_filter_applied": bool(mode_filter_applied),
        "selection_modes": mode_info,
        "source_pipeline": "legacy_qc_trials + fixed_3d_knee_min_hip_sagittal",
        "outputs": {
            "trajectories_csv": str(traj_csv) if traj_csv is not None else None,
            "angles_csv": str(angle_csv),
            "cycle_metrics_csv": str(cycle_metrics_csv),
            "plot_png": str(plot_path),
        },
    }
    summary_path = out_dir / f"{basename}_canonical_summary.yaml"
    _write_yaml(summary_path, summary)
    return {
        "dataset": dataset_name,
        "group": group_label,
        "basename": basename,
        "n_trials": int(len(trial_specs)),
        "candidate_cycles": int(len(candidates)),
        "selected_cycles": int(len(selected_rows)),
        "score_threshold": float(score_threshold),
        "mode_filter_applied": bool(mode_filter_applied),
        "angles_csv": str(angle_csv),
        "trajectories_csv": str(traj_csv) if traj_csv is not None else "",
        "summary_yaml": str(summary_path),
        "status": "ok",
    }


def _selection_modes(rows: list[dict], joints: list[str]) -> dict[str, dict]:
    out = {}
    for joint_name in joints:
        counter = Counter(
            (
                row.get(f"{joint_name}_source"),
                row.get(f"{joint_name}_variant"),
                row.get(f"{joint_name}_sign"),
            )
            for row in rows
        )
        if not counter:
            continue
        (source, variant, sign), count = counter.most_common(1)[0]
        out[joint_name] = {
            "source": source,
            "variant": variant,
            "sign": int(sign),
            "count": int(count),
            "ratio": float(count / max(1, len(rows))),
        }
    return out


def _resolve_events_path(traj_path: Path) -> Path | None:
    base = traj_path.stem.replace("_Trajectories", "")
    point = traj_path.with_name(f"{base}_point_gaitEvents.yaml")
    plain = traj_path.with_name(f"{base}_gaitEvents.yaml")
    if point.exists():
        return point
    if plain.exists():
        return plain
    return None


def _trajectory_process_group(
    dataset_name: str,
    group_label: str,
    basename: str,
    triplets: list[tuple[Path, Path]],
    out_dir: Path,
    plots_dir: Path,
    n_points: int,
    min_stride_s: float,
    max_stride_s: float,
    keep_percentile: float,
    require_toe_off: bool,
) -> dict:
    pct = np.linspace(0.0, 100.0, n_points)
    norm_time = np.linspace(0.0, 1.0, n_points)
    templates = _canonical_templates(pct)
    markers_ref: list[str] | None = None
    candidates: list[dict] = []
    cycle_rows: list[dict] = []
    frame_examples: list[dict] = []
    n_trials = 0
    n_pairs_seen = 0
    n_pairs_kept = 0

    for traj_path, events_path in triplets:
        n_trials += 1
        traj_df = pd.read_csv(traj_path)
        if "time" not in traj_df.columns:
            raise KeyError(f"Missing time column in {traj_path}")
        traj_time = pd.to_numeric(traj_df["time"], errors="coerce").to_numpy(dtype=float)
        marker_data_all = _extract_marker_data(traj_df)
        markers = sorted(marker_data_all.keys())
        if markers_ref is None:
            markers_ref = markers
        elif markers != markers_ref:
            raise ValueError(f"Marker set mismatch in {traj_path.name}")

        raw_events = _load_yaml(events_path)
        event_lists = {
            key: [float(value) for value in values]
            for key, values in raw_events.items()
            if isinstance(values, list)
        }
        aligned_events = _align_events_to_trial_time(traj_time, event_lists)
        events = dict(raw_events)
        events.update(aligned_events)
        windows, stats = _cycle_windows(
            events,
            min_stride_s=min_stride_s,
            max_stride_s=max_stride_s,
            require_toe_off=require_toe_off,
        )
        n_pairs_seen += int(stats["pairs_seen"])
        n_pairs_kept += int(stats["pairs_kept"])

        for window in windows:
            start_t = float(window["start_time_s"])
            end_t = float(window["end_time_s"])
            traj_mask = (traj_time >= start_t) & (traj_time <= end_t)
            traj_samples = int(np.count_nonzero(traj_mask))
            row = {
                "dataset": dataset_name,
                "group": group_label,
                "trial": traj_path.name,
                "cycle_idx": int(window["cycle_idx"]),
                "start_time_s": start_t,
                "end_time_s": end_t,
                "stride_time_s": float(window["stride_time_s"]),
                "toe_off_count": int(window["toe_off_count"]),
                "trajectory_samples": traj_samples,
                "selected": False,
            }
            row_index = len(cycle_rows)
            cycle_rows.append(row)
            if traj_samples < 5:
                cycle_rows[row_index]["status"] = "skip_short_segment"
                continue

            try:
                time_seg = traj_time[traj_mask]
                marker_seg = {marker: values[traj_mask] for marker, values in marker_data_all.items()}
                standardized, frame_meta = _standardize_cycle_frame(marker_seg)
                marker_parts = [_normalize_matrix(time_seg, standardized[marker], n_points) for marker in markers_ref]
                marker_matrix = np.concatenate(marker_parts, axis=1)
                raw_joint_candidates = _build_right_joint_candidates(standardized)

                chosen: dict[str, dict] = {}
                total_score = 0.0
                for joint_name in ["hip", "knee", "ankle"]:
                    if not raw_joint_candidates.get(joint_name):
                        raise ValueError(f"No right-side marker candidates for {joint_name}.")
                    picked = _pick_best_joint_curve(
                        joint_name=joint_name,
                        raw_candidates=raw_joint_candidates[joint_name],
                        time_seg=time_seg,
                        n_points=n_points,
                        template=templates[joint_name],
                    )
                    chosen[joint_name] = picked
                    total_score += float(picked["score"])

                cycle_rows[row_index].update(
                    {
                        "status": "candidate",
                        "total_score": float(total_score),
                    }
                )
                cycle_rows[row_index].update(
                    {
                        "frame_forward_axis": int(frame_meta["source_axis_order"]["x"]),
                        "frame_lateral_axis": int(frame_meta["source_axis_order"]["y"]),
                        "frame_vertical_axis": int(frame_meta["source_axis_order"]["z"]),
                        "frame_sign_x": int(frame_meta["sign_flips"]["x"]),
                        "frame_sign_y": int(frame_meta["sign_flips"]["y"]),
                        "frame_sign_z": int(frame_meta["sign_flips"]["z"]),
                    }
                )
                for joint_name in ["hip", "knee", "ankle"]:
                    cycle_rows[row_index][f"{joint_name}_source"] = chosen[joint_name]["source"]
                    cycle_rows[row_index][f"{joint_name}_variant"] = chosen[joint_name]["variant"]
                    cycle_rows[row_index][f"{joint_name}_sign"] = int(chosen[joint_name]["sign"])
                    cycle_rows[row_index][f"{joint_name}_score"] = float(chosen[joint_name]["score"])
                    cycle_rows[row_index][f"{joint_name}_rom_deg"] = float(chosen[joint_name]["rom_deg"])
                    cycle_rows[row_index][f"{joint_name}_max_step_deg"] = float(chosen[joint_name]["max_step_deg"])
                    cycle_rows[row_index][f"{joint_name}_shape_distance"] = float(chosen[joint_name]["shape_distance"])

                candidates.append(
                    {
                        "row_index": row_index,
                        "marker_matrix": marker_matrix,
                        "hip_curve": chosen["hip"]["curve"],
                        "knee_curve": chosen["knee"]["curve"],
                        "ankle_curve": chosen["ankle"]["curve"],
                        "score": float(total_score),
                        "hip_source": chosen["hip"]["source"],
                        "hip_variant": chosen["hip"]["variant"],
                        "hip_sign": int(chosen["hip"]["sign"]),
                        "knee_source": chosen["knee"]["source"],
                        "knee_variant": chosen["knee"]["variant"],
                        "knee_sign": int(chosen["knee"]["sign"]),
                        "ankle_source": chosen["ankle"]["source"],
                        "ankle_variant": chosen["ankle"]["variant"],
                        "ankle_sign": int(chosen["ankle"]["sign"]),
                    }
                )
                if len(frame_examples) < 3:
                    frame_examples.append(frame_meta)
            except Exception as exc:
                cycle_rows[row_index]["status"] = "fail"
                cycle_rows[row_index]["error"] = str(exc)

    if not candidates or markers_ref is None:
        raise RuntimeError(f"No valid candidate cycles found for {group_label}")

    candidate_scores = np.asarray([candidate["score"] for candidate in candidates], dtype=float)
    score_threshold = float(np.percentile(candidate_scores, keep_percentile))
    keep_idx = np.where(candidate_scores <= score_threshold)[0]
    if keep_idx.size == 0:
        keep_idx = np.array([int(np.argmin(candidate_scores))], dtype=int)
    selected_rows = [candidates[idx] for idx in sorted(set(keep_idx.tolist()))]

    mode_info = _selection_modes(selected_rows, ["hip", "knee", "ankle"])
    mode_filtered_rows = [
        row
        for row in selected_rows
        if all(
            row[f"{joint_name}_source"] == mode_info[joint_name]["source"]
            and row[f"{joint_name}_variant"] == mode_info[joint_name]["variant"]
            and row[f"{joint_name}_sign"] == mode_info[joint_name]["sign"]
            for joint_name in mode_info
        )
    ]
    mode_filter_applied = False
    if len(selected_rows) >= 5 and len(mode_filtered_rows) >= max(3, int(math.floor(0.6 * len(selected_rows)))):
        selected_rows = mode_filtered_rows
        mode_filter_applied = True

    for row in selected_rows:
        cycle_rows[row["row_index"]]["selected"] = True
        cycle_rows[row["row_index"]]["status"] = "selected"

    traj_blocks: list[np.ndarray] = []
    markers_out: list[str] = []
    for marker_idx, marker in enumerate(markers_ref):
        block = np.stack([row["marker_matrix"][:, marker_idx * 3 : (marker_idx + 1) * 3] for row in selected_rows], axis=0)
        if np.any(np.isfinite(block)):
            traj_blocks.append(block)
            markers_out.append(marker)
    if not traj_blocks:
        raise RuntimeError(f"No finite marker trajectories available for {group_label}")

    traj_stack = np.concatenate(traj_blocks, axis=2)
    hip_stack = np.stack([row["hip_curve"] for row in selected_rows], axis=0)
    knee_stack = np.stack([row["knee_curve"] for row in selected_rows], axis=0)
    ankle_stack = np.stack([row["ankle_curve"] for row in selected_rows], axis=0)

    traj_median = np.nanmedian(traj_stack, axis=0)
    hip_median = np.nanmedian(hip_stack, axis=0)
    knee_median = np.nanmedian(knee_stack, axis=0)
    ankle_median = np.nanmedian(ankle_stack, axis=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    traj_columns = ["time", "pct"]
    for marker in markers_out:
        traj_columns.extend([f"{marker}_x", f"{marker}_y", f"{marker}_z"])

    traj_csv = out_dir / f"{basename}_canonical_Trajectories.csv"
    pd.DataFrame(np.column_stack([norm_time, pct, traj_median]), columns=traj_columns).to_csv(traj_csv, index=False)

    angle_csv = out_dir / f"{basename}_canonical_marker_angles_norm101.csv"
    pd.DataFrame(
        {
            "pct": pct,
            "hip_flexion": hip_median,
            "knee_flexion": knee_median,
            "ankle_dorsiflexion": ankle_median,
        }
    ).to_csv(angle_csv, index=False)

    cycle_metrics_csv = out_dir / f"{basename}_canonical_cycle_metrics.csv"
    pd.DataFrame(cycle_rows).to_csv(cycle_metrics_csv, index=False)

    plot_path = plots_dir / f"{basename}_canonical_profiles.png"
    _plot_angle_overlay(
        plot_path,
        pct=pct,
        stacks={"hip": hip_stack, "knee": knee_stack, "ankle": ankle_stack},
        medians={"hip": hip_median, "knee": knee_median, "ankle": ankle_median},
        templates=templates,
        title=group_label,
        subtitle=f"selected_cycles={len(selected_rows)} / candidates={len(candidates)}",
    )

    summary = {
        "dataset": dataset_name,
        "group": group_label,
        "basename": basename,
        "n_trials": int(n_trials),
        "event_pairs_seen": int(n_pairs_seen),
        "candidate_cycles": int(len(candidates)),
        "selected_cycles": int(len(selected_rows)),
        "score_threshold": float(score_threshold),
        "keep_percentile": float(keep_percentile),
        "mode_filter_applied": bool(mode_filter_applied),
        "selection_modes": mode_info,
        "markers": markers_out,
        "frame_examples": frame_examples,
        "outputs": {
            "trajectories_csv": str(traj_csv),
            "angles_csv": str(angle_csv),
            "cycle_metrics_csv": str(cycle_metrics_csv),
            "plot_png": str(plot_path),
        },
    }
    summary_path = out_dir / f"{basename}_canonical_summary.yaml"
    _write_yaml(summary_path, summary)
    return {
        "dataset": dataset_name,
        "group": group_label,
        "basename": basename,
        "n_trials": int(n_trials),
        "candidate_cycles": int(len(candidates)),
        "selected_cycles": int(len(selected_rows)),
        "score_threshold": float(score_threshold),
        "mode_filter_applied": bool(mode_filter_applied),
        "angles_csv": str(angle_csv),
        "trajectories_csv": str(traj_csv),
        "summary_yaml": str(summary_path),
        "status": "ok",
    }


def _read_mode_source(source_file: str) -> pd.Series:
    if "::" in source_file:
        zip_path_str, member = source_file.split("::", 1)
        with zipfile.ZipFile(Path(zip_path_str), "r") as zf:
            with zf.open(member, "r") as fh:
                df = pd.read_csv(TextIOWrapper(fh, encoding="utf-8"), usecols=["Mode"])
    else:
        df = pd.read_csv(source_file, usecols=["Mode"])
    return pd.to_numeric(df["Mode"], errors="coerce")


def _benchmark_cycle_candidates(
    joint_csv: Path,
    events_yaml: Path,
    info_yaml: Path,
    n_points: int,
    purity_threshold: float,
    min_duration_s: float,
    max_duration_s: float,
) -> tuple[list[dict], list[dict]]:
    info = _load_yaml(info_yaml)
    source_file = str(info.get("source_file") or "").strip()
    if not source_file:
        raise ValueError(f"Missing source_file in {info_yaml.name}")

    df = pd.read_csv(joint_csv)
    required = ["time", "RKneeAngles_x", "RAnkleAngles_x", "RHipAngles_x"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{joint_csv.name} missing columns: {missing}")

    time = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
    hip = pd.to_numeric(df["RHipAngles_x"], errors="coerce").to_numpy(dtype=float)
    knee = pd.to_numeric(df["RKneeAngles_x"], errors="coerce").to_numpy(dtype=float)
    ankle = pd.to_numeric(df["RAnkleAngles_x"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(time) & np.isfinite(hip) & np.isfinite(knee) & np.isfinite(ankle)
    time = time[valid]
    hip = hip[valid]
    knee = knee[valid]
    ankle = ankle[valid]

    mode = _read_mode_source(source_file).to_numpy(dtype=float)
    if mode.size != time.size:
        n = min(mode.size, time.size)
        mode = mode[:n]
        time = time[:n]
        hip = hip[:n]
        knee = knee[:n]
        ankle = ankle[:n]

    events = _load_yaml(events_yaml)
    windows, _ = _cycle_windows(
        events,
        min_stride_s=min_duration_s,
        max_stride_s=max_duration_s,
        require_toe_off=False,
    )
    pct = np.linspace(0.0, 100.0, n_points)
    cycles: list[dict] = []
    skipped: list[dict] = []
    for window in windows:
        start = float(window["start_time_s"])
        end = float(window["end_time_s"])
        mask = (time >= start) & (time <= end)
        if np.count_nonzero(mask) < 3:
            skipped.append({"file": joint_csv.name, "cycle_index": window["cycle_idx"], "reason": "not_enough_samples"})
            continue

        mode_seg = mode[mask]
        finite_mode = mode_seg[np.isfinite(mode_seg)]
        if finite_mode.size == 0:
            skipped.append({"file": joint_csv.name, "cycle_index": window["cycle_idx"], "reason": "mode_missing"})
            continue
        dominant_mode = int(pd.Series(finite_mode).mode().iloc[0])
        level_ratio = float(np.mean(finite_mode == BENCHMARK_LEVEL_WALKING_MODE))
        if dominant_mode != BENCHMARK_LEVEL_WALKING_MODE or level_ratio < purity_threshold:
            skipped.append(
                {
                    "file": joint_csv.name,
                    "cycle_index": window["cycle_idx"],
                    "reason": "not_level_walking",
                    "dominant_mode": dominant_mode,
                    "level_ratio": level_ratio,
                }
            )
            continue

        t_seg = time[mask]
        cycles.append(
            {
                "trial": joint_csv.stem.replace("_jointAngles", ""),
                "cycle_idx": int(window["cycle_idx"]),
                "start_time_s": start,
                "end_time_s": end,
                "stride_time_s": float(window["stride_time_s"]),
                "level_walking_ratio": level_ratio,
                "pct": pct,
                "hip_raw": _normalize_vector(t_seg, hip[mask], n_points),
                "knee_raw": _normalize_vector(t_seg, knee[mask], n_points),
                "ankle_raw": _normalize_vector(t_seg, ankle[mask], n_points),
            }
        )
    return cycles, skipped


def _benchmark_process_subject(
    subject_dir: Path,
    out_root: Path,
    plots_root: Path,
    n_points: int,
    keep_percentile: float,
    purity_threshold: float,
    min_duration_s: float,
    max_duration_s: float,
) -> dict:
    pct = np.linspace(0.0, 100.0, n_points)
    templates = _canonical_templates(pct)
    out_dir = out_root / subject_dir.name
    basename = subject_dir.name
    candidates: list[dict] = []
    cycle_rows: list[dict] = []
    n_trials = 0

    for joint_csv in sorted(subject_dir.glob("*_jointAngles.csv")):
        base = joint_csv.stem.replace("_jointAngles", "")
        events_yaml = subject_dir / f"{base}_gaitEvents.yaml"
        info_yaml = subject_dir / f"{base}_info.yaml"
        if not events_yaml.exists() or not info_yaml.exists():
            continue
        n_trials += 1
        try:
            cycles, skipped = _benchmark_cycle_candidates(
                joint_csv,
                events_yaml,
                info_yaml,
                n_points=n_points,
                purity_threshold=purity_threshold,
                min_duration_s=min_duration_s,
                max_duration_s=max_duration_s,
            )
            for item in skipped:
                cycle_rows.append(
                    {
                        "dataset": "benchmark_bilateral_lower_limb",
                        "group": subject_dir.name,
                        "trial": joint_csv.name,
                        "status": item["reason"],
                        "cycle_idx": item.get("cycle_index"),
                        "selected": False,
                    }
                )
            for cycle in cycles:
                row = {
                    "dataset": "benchmark_bilateral_lower_limb",
                    "group": subject_dir.name,
                    "trial": cycle["trial"],
                    "cycle_idx": cycle["cycle_idx"],
                    "start_time_s": cycle["start_time_s"],
                    "end_time_s": cycle["end_time_s"],
                    "stride_time_s": cycle["stride_time_s"],
                    "level_walking_ratio": cycle["level_walking_ratio"],
                    "selected": False,
                }
                row_index = len(cycle_rows)
                cycle_rows.append(row)
                raw_map = {
                    "knee": [{"source": "RKneeAngles_x", "curve": cycle["knee_raw"]}],
                    "ankle": [{"source": "RAnkleAngles_x", "curve": cycle["ankle_raw"]}],
                }
                if float(np.nanstd(cycle["hip_raw"])) > 1e-6:
                    raw_map["hip"] = [{"source": "RHipAngles_x", "curve": cycle["hip_raw"]}]

                chosen: dict[str, dict] = {}
                total_score = 0.0
                for joint_name in raw_map:
                    chosen_item = _pick_best_joint_curve(
                        joint_name=joint_name,
                        raw_candidates=raw_map[joint_name],
                        time_seg=np.linspace(0.0, 1.0, len(cycle[f"{joint_name}_raw"])),
                        n_points=n_points,
                        template=templates[joint_name],
                    )
                    chosen[joint_name] = chosen_item
                    total_score += float(chosen_item["score"])

                cycle_rows[row_index].update({"status": "candidate", "total_score": float(total_score)})
                for joint_name, picked in chosen.items():
                    cycle_rows[row_index][f"{joint_name}_source"] = picked["source"]
                    cycle_rows[row_index][f"{joint_name}_variant"] = picked["variant"]
                    cycle_rows[row_index][f"{joint_name}_sign"] = picked["sign"]
                    cycle_rows[row_index][f"{joint_name}_score"] = picked["score"]
                    cycle_rows[row_index][f"{joint_name}_rom_deg"] = picked["rom_deg"]
                    cycle_rows[row_index][f"{joint_name}_max_step_deg"] = picked["max_step_deg"]
                    cycle_rows[row_index][f"{joint_name}_shape_distance"] = picked["shape_distance"]

                candidates.append(
                    {
                        "row_index": row_index,
                        "score": float(total_score),
                        "hip_curve": chosen["hip"]["curve"] if "hip" in chosen else np.zeros(n_points, dtype=float),
                        "knee_curve": chosen["knee"]["curve"],
                        "ankle_curve": chosen["ankle"]["curve"],
                        "hip_available": "hip" in chosen,
                        "knee_source": chosen["knee"]["source"],
                        "knee_variant": chosen["knee"]["variant"],
                        "knee_sign": int(chosen["knee"]["sign"]),
                        "ankle_source": chosen["ankle"]["source"],
                        "ankle_variant": chosen["ankle"]["variant"],
                        "ankle_sign": int(chosen["ankle"]["sign"]),
                    }
                )
        except Exception as exc:
            cycle_rows.append(
                {
                    "dataset": "benchmark_bilateral_lower_limb",
                    "group": subject_dir.name,
                    "trial": joint_csv.name,
                    "status": "fail",
                    "error": str(exc),
                    "selected": False,
                }
            )

    if not candidates:
        raise RuntimeError(f"No valid benchmark candidates for {subject_dir.name}")

    candidate_scores = np.asarray([candidate["score"] for candidate in candidates], dtype=float)
    score_threshold = float(np.percentile(candidate_scores, keep_percentile))
    keep_idx = np.where(candidate_scores <= score_threshold)[0]
    if keep_idx.size == 0:
        keep_idx = np.array([int(np.argmin(candidate_scores))], dtype=int)
    selected_rows = [candidates[idx] for idx in sorted(set(keep_idx.tolist()))]
    mode_info = _selection_modes(selected_rows, ["knee", "ankle"])
    mode_filtered_rows = [
        row
        for row in selected_rows
        if all(
            row[f"{joint_name}_source"] == mode_info[joint_name]["source"]
            and row[f"{joint_name}_variant"] == mode_info[joint_name]["variant"]
            and row[f"{joint_name}_sign"] == mode_info[joint_name]["sign"]
            for joint_name in mode_info
        )
    ]
    mode_filter_applied = False
    if len(selected_rows) >= 5 and len(mode_filtered_rows) >= max(3, int(math.floor(0.6 * len(selected_rows)))):
        selected_rows = mode_filtered_rows
        mode_filter_applied = True

    for row in selected_rows:
        cycle_rows[row["row_index"]]["selected"] = True
        cycle_rows[row["row_index"]]["status"] = "selected"

    hip_stack = np.stack([row["hip_curve"] for row in selected_rows], axis=0)
    knee_stack = np.stack([row["knee_curve"] for row in selected_rows], axis=0)
    ankle_stack = np.stack([row["ankle_curve"] for row in selected_rows], axis=0)
    hip_median = np.nanmedian(hip_stack, axis=0)
    knee_median = np.nanmedian(knee_stack, axis=0)
    ankle_median = np.nanmedian(ankle_stack, axis=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    angle_csv = out_dir / f"{basename}_canonical_marker_angles_norm101.csv"
    pd.DataFrame(
        {
            "pct": pct,
            "hip_flexion": hip_median,
            "knee_flexion": knee_median,
            "ankle_dorsiflexion": ankle_median,
        }
    ).to_csv(angle_csv, index=False)

    cycle_metrics_csv = out_dir / f"{basename}_canonical_cycle_metrics.csv"
    pd.DataFrame(cycle_rows).to_csv(cycle_metrics_csv, index=False)

    plot_path = plots_root / f"{basename}_canonical_profiles.png"
    _plot_angle_overlay(
        plot_path,
        pct=pct,
        stacks={"hip": hip_stack, "knee": knee_stack, "ankle": ankle_stack},
        medians={"hip": hip_median, "knee": knee_median, "ankle": ankle_median},
        templates=templates,
        title=subject_dir.name,
        subtitle=f"selected_cycles={len(selected_rows)} / candidates={len(candidates)}",
    )

    summary = {
        "dataset": "benchmark_bilateral_lower_limb",
        "group": subject_dir.name,
        "basename": basename,
        "n_trials": int(n_trials),
        "candidate_cycles": int(len(candidates)),
        "selected_cycles": int(len(selected_rows)),
        "score_threshold": float(score_threshold),
        "keep_percentile": float(keep_percentile),
        "mode_filter_applied": bool(mode_filter_applied),
        "selection_modes": mode_info,
        "hip_available": bool(any(row["hip_available"] for row in selected_rows)),
        "outputs": {
            "angles_csv": str(angle_csv),
            "cycle_metrics_csv": str(cycle_metrics_csv),
            "plot_png": str(plot_path),
        },
    }
    summary_path = out_dir / f"{basename}_canonical_summary.yaml"
    _write_yaml(summary_path, summary)
    return {
        "dataset": "benchmark_bilateral_lower_limb",
        "group": subject_dir.name,
        "basename": basename,
        "n_trials": int(n_trials),
        "candidate_cycles": int(len(candidates)),
        "selected_cycles": int(len(selected_rows)),
        "score_threshold": float(score_threshold),
        "mode_filter_applied": bool(mode_filter_applied),
        "angles_csv": str(angle_csv),
        "trajectories_csv": "",
        "summary_yaml": str(summary_path),
        "status": "ok",
    }


def _human_gait_groups(eurobench_root: Path) -> dict[tuple[str, str], list[tuple[Path, Path]]]:
    groups: dict[tuple[str, str], list[tuple[Path, Path]]] = defaultdict(list)
    for subject_dir in sorted(p for p in eurobench_root.glob("P*_S*") if p.is_dir()):
        for traj_path in sorted(subject_dir.glob("*_Trajectories.csv")):
            match = TASK_REGEX.match(traj_path.name)
            if match is None:
                continue
            task = match.group("task")
            if task not in WALK_TASKS:
                continue
            events_path = _resolve_events_path(traj_path)
            if events_path is None:
                continue
            groups[(task, subject_dir.name)].append((traj_path, events_path))
    return groups


def _multisensor_groups(eurobench_root: Path) -> dict[str, list[tuple[Path, Path]]]:
    groups: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for subject_dir in sorted(p for p in eurobench_root.glob("user*") if p.is_dir()):
        for traj_path in sorted(subject_dir.glob("*_Trajectories.csv")):
            events_path = _resolve_events_path(traj_path)
            if events_path is None:
                continue
            groups[subject_dir.name].append((traj_path, events_path))
    return groups


def _gait_assessment_groups(eurobench_root: Path) -> dict[str, list[tuple[Path, Path]]]:
    groups: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for subject_dir in sorted(p for p in eurobench_root.glob("Subject*") if p.is_dir()):
        for traj_path in sorted(subject_dir.glob("*_Trajectories.csv")):
            events_path = _resolve_events_path(traj_path)
            if events_path is None:
                continue
            groups[subject_dir.name].append((traj_path, events_path))
    return groups


def run_human_gait(args) -> pd.DataFrame:
    eurobench_root = Path(args.human_gait_root)
    out_root = Path(args.human_gait_out)
    plots_root = Path(args.human_gait_plots)
    rows: list[dict] = []
    for (task, subject), triplets in sorted(_human_gait_groups(eurobench_root).items()):
        out_dir = out_root / task / subject
        plots_dir = plots_root / task / subject
        basename = f"{subject}_{task}"
        try:
            result = _trajectory_process_group(
                dataset_name="human_gait",
                group_label=f"{subject}/{task}",
                basename=basename,
                triplets=triplets,
                out_dir=out_dir,
                plots_dir=plots_dir,
                n_points=args.n_points,
                min_stride_s=0.5,
                max_stride_s=2.0,
                keep_percentile=args.human_gait_keep_percentile,
                require_toe_off=True,
            )
            result["task"] = task
            result["subject"] = subject
            rows.append(result)
        except Exception as exc:
            rows.append(
                {
                    "dataset": "human_gait",
                    "group": f"{subject}/{task}",
                    "basename": basename,
                    "task": task,
                    "subject": subject,
                    "status": "fail",
                    "error": str(exc),
                }
            )
    df = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_root / "human_gait_canonical_groups_summary.csv", index=False)
    return df


def run_multisensor(args) -> pd.DataFrame:
    eurobench_root = Path(args.multisensor_root)
    legacy_processed_root = Path(args.multisensor_legacy_processed_root)
    legacy_summary_csv = Path(args.multisensor_legacy_summary)
    out_root = Path(args.multisensor_out)
    plots_root = Path(args.multisensor_plots)
    rows: list[dict] = []
    for subject, trial_specs in sorted(
        _multisensor_legacy_groups(
            legacy_processed_root=legacy_processed_root,
            eurobench_root=eurobench_root,
            summary_csv=legacy_summary_csv,
        ).items()
    ):
        out_dir = out_root / subject
        plots_dir = plots_root / subject
        try:
            result = _multisensor_fixed_process_group(
                dataset_name="multisensor_gait",
                group_label=subject,
                basename=subject,
                trial_specs=trial_specs,
                out_dir=out_dir,
                plots_dir=plots_dir,
                n_points=args.n_points,
                keep_percentile=args.keep_percentile,
            )
            result["subject"] = subject
            rows.append(result)
        except Exception as exc:
            rows.append(
                {
                    "dataset": "multisensor_gait",
                    "group": subject,
                    "basename": subject,
                    "subject": subject,
                    "status": "fail",
                    "error": str(exc),
                }
            )
    df = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_root / "multisensor_gait_canonical_subjects_summary.csv", index=False)
    return df


def run_gait_assessment(args) -> pd.DataFrame:
    eurobench_root = Path(args.gait_assessment_root)
    out_root = Path(args.gait_assessment_out)
    plots_root = Path(args.gait_assessment_plots)
    rows: list[dict] = []
    for subject, triplets in sorted(_gait_assessment_groups(eurobench_root).items()):
        out_dir = out_root / subject
        plots_dir = plots_root / subject
        try:
            result = _trajectory_process_group(
                dataset_name="gait_analysis_assessment",
                group_label=subject,
                basename=subject,
                triplets=triplets,
                out_dir=out_dir,
                plots_dir=plots_dir,
                n_points=args.n_points,
                min_stride_s=0.8,
                max_stride_s=2.0,
                keep_percentile=args.keep_percentile,
                require_toe_off=True,
            )
            result["subject"] = subject
            rows.append(result)
        except Exception as exc:
            rows.append(
                {
                    "dataset": "gait_analysis_assessment",
                    "group": subject,
                    "basename": subject,
                    "subject": subject,
                    "status": "fail",
                    "error": str(exc),
                }
            )
    df = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_root / "gait_analysis_assessment_canonical_subjects_summary.csv", index=False)
    return df


def run_benchmark(args) -> pd.DataFrame:
    eurobench_root = Path(args.benchmark_root)
    out_root = Path(args.benchmark_out)
    plots_root = Path(args.benchmark_plots)
    rows: list[dict] = []
    for subject_dir in sorted(p for p in eurobench_root.glob("AB*") if p.is_dir()):
        try:
            result = _benchmark_process_subject(
                subject_dir=subject_dir,
                out_root=out_root,
                plots_root=plots_root,
                n_points=args.n_points,
                keep_percentile=args.keep_percentile,
                purity_threshold=args.benchmark_purity_threshold,
                min_duration_s=0.6,
                max_duration_s=2.0,
            )
            result["subject"] = subject_dir.name
            rows.append(result)
        except Exception as exc:
            rows.append(
                {
                    "dataset": "benchmark_bilateral_lower_limb",
                    "group": subject_dir.name,
                    "basename": subject_dir.name,
                    "subject": subject_dir.name,
                    "status": "fail",
                    "error": str(exc),
                }
            )
    df = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_root / "benchmark_bilateral_lower_limb_canonical_subjects_summary.csv", index=False)
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build canonical gait profiles across multiple datasets.")
    parser.add_argument(
        "--datasets",
        default="human_gait,multisensor_gait,gait_analysis_assessment,benchmark_bilateral_lower_limb",
        help="Comma-separated datasets to process.",
    )
    parser.add_argument("--n-points", type=int, default=DEFAULT_N_POINTS)
    parser.add_argument("--keep-percentile", type=float, default=15.0)
    parser.add_argument("--human-gait-keep-percentile", type=float, default=40.0)
    parser.add_argument("--benchmark-purity-threshold", type=float, default=0.95)

    parser.add_argument("--human-gait-root", default="data/human_gait/eurobench")
    parser.add_argument("--human-gait-out", default="data/human_gait/processed_canonical")
    parser.add_argument("--human-gait-plots", default="data/human_gait/plots_canonical")

    parser.add_argument("--multisensor-root", default="data/multisensor_gait/eurobench")
    parser.add_argument("--multisensor-out", default="data/multisensor_gait/processed_canonical")
    parser.add_argument("--multisensor-plots", default="data/multisensor_gait/plots_canonical")
    parser.add_argument(
        "--multisensor-legacy-processed-root",
        default="data/multisensor_gait/processed 11-08-23-207",
    )
    parser.add_argument(
        "--multisensor-legacy-summary",
        default="data/multisensor_gait/analysis/all_trials_summary.csv",
    )

    parser.add_argument("--gait-assessment-root", default="data/gait_analysis_assessment/eurobench")
    parser.add_argument("--gait-assessment-out", default="data/gait_analysis_assessment/processed_canonical")
    parser.add_argument("--gait-assessment-plots", default="data/gait_analysis_assessment/plots_canonical")

    parser.add_argument("--benchmark-root", default="data/benchmark_datasets_for_bilateral_lower_limb/eurobench")
    parser.add_argument("--benchmark-out", default="data/benchmark_datasets_for_bilateral_lower_limb/processed_canonical")
    parser.add_argument("--benchmark-plots", default="data/benchmark_datasets_for_bilateral_lower_limb/plots_canonical")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    wanted = {item.strip() for item in args.datasets.split(",") if item.strip()}
    summaries: list[pd.DataFrame] = []
    if "human_gait" in wanted:
        summaries.append(run_human_gait(args))
    if "multisensor_gait" in wanted:
        summaries.append(run_multisensor(args))
    if "gait_analysis_assessment" in wanted:
        summaries.append(run_gait_assessment(args))
    if "benchmark_bilateral_lower_limb" in wanted:
        summaries.append(run_benchmark(args))

    if summaries:
        combined = pd.concat(summaries, ignore_index=True, sort=False)
        combined.to_csv("data/canonical_gait_all_datasets_summary.csv", index=False)
        print("data/canonical_gait_all_datasets_summary.csv")


if __name__ == "__main__":
    main()
