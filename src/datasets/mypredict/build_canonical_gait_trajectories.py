import argparse
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


REQUIRED_FRAME_MARKERS = ("PELV", "C7", "LASI", "RASI")
JOINT_PREFIXES = {
    "hip": "RHipAngles",
    "knee": "RKneeAngles",
    "ankle": "RAnkleAngles",
}
AXIS_LABELS = ("forward", "lateral_right", "vertical_up")
AXES = ("x", "y", "z")
SIGNS = (1.0, -1.0)
MIN_KEEP_CYCLES = 50
MIN_MODE_KEEP_CYCLES = 40


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML payload in {path}")
    return payload


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def _available_markers(df: pd.DataFrame) -> list[str]:
    markers = []
    for marker in sorted({col[:-2] for col in df.columns if col.endswith("_x")}):
        cols = [f"{marker}_x", f"{marker}_y", f"{marker}_z"]
        if all(col in df.columns for col in cols):
            markers.append(marker)
    return markers


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


def _nanptp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    return np.nanmax(values, axis=axis) - np.nanmin(values, axis=axis)


def _cycle_windows(
    events: dict,
    min_stride_s: float,
    max_stride_s: float,
    require_toe_off: bool,
) -> tuple[list[dict], dict[str, int]]:
    rhs = np.sort(np.asarray(events.get("r_heel_strike", []) or [], dtype=float))
    rto = np.sort(np.asarray(events.get("r_toe_off", []) or [], dtype=float))

    windows: list[dict] = []
    stats = {
        "pairs_seen": 0,
        "pairs_outside_stride_range": 0,
        "pairs_without_toe_off": 0,
        "pairs_kept": 0,
    }

    if rhs.size < 2:
        return windows, stats

    for cycle_idx, (start_t, end_t) in enumerate(zip(rhs[:-1], rhs[1:]), start=1):
        stats["pairs_seen"] += 1
        stride_time_s = float(end_t - start_t)
        if not (min_stride_s <= stride_time_s <= max_stride_s):
            stats["pairs_outside_stride_range"] += 1
            continue

        toe_off = rto[(rto > start_t) & (rto < end_t)]
        if require_toe_off and toe_off.size == 0:
            stats["pairs_without_toe_off"] += 1
            continue

        windows.append(
            {
                "cycle_idx": int(cycle_idx),
                "start_time_s": float(start_t),
                "end_time_s": float(end_t),
                "stride_time_s": stride_time_s,
                "toe_off_count": int(toe_off.size),
            }
        )
        stats["pairs_kept"] += 1

    return windows, stats


def _extract_marker_data(df: pd.DataFrame, markers: list[str]) -> dict[str, np.ndarray]:
    return {marker: _safe_numeric(df, [f"{marker}_x", f"{marker}_y", f"{marker}_z"]) for marker in markers}


def _major_axis_from_pelvis(pelv: np.ndarray) -> tuple[int, int, int]:
    ranges = _nanptp(pelv, axis=0)
    forward_idx = int(np.nanargmax(ranges))
    lateral_idx = int(np.nanargmin(ranges))
    vertical_idx = int(({0, 1, 2}.difference({forward_idx, lateral_idx})).pop())
    return forward_idx, lateral_idx, vertical_idx


def _standardize_cycle_frame(marker_data: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict]:
    missing = [name for name in REQUIRED_FRAME_MARKERS if name not in marker_data]
    if missing:
        raise KeyError(f"Missing markers required for frame alignment: {missing}")

    pelv = marker_data["PELV"]
    c7 = marker_data["C7"]
    lasi = marker_data["LASI"]
    rasi = marker_data["RASI"]

    forward_idx, lateral_idx, vertical_idx = _major_axis_from_pelvis(pelv)
    order = np.array([forward_idx, lateral_idx, vertical_idx], dtype=int)

    forward_step = float(pelv[-1, forward_idx] - pelv[0, forward_idx])
    lateral_span = float(np.nanmedian(rasi[:, lateral_idx] - lasi[:, lateral_idx]))
    vertical_span = float(np.nanmedian(c7[:, vertical_idx] - pelv[:, vertical_idx]))

    signs = np.array(
        [
            1.0 if forward_step >= 0.0 else -1.0,
            1.0 if lateral_span >= 0.0 else -1.0,
            1.0 if vertical_span >= 0.0 else -1.0,
        ],
        dtype=float,
    )

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
        "axis_labels": {
            "x": AXIS_LABELS[0],
            "y": AXIS_LABELS[1],
            "z": AXIS_LABELS[2],
        },
    }
    return standardized, frame_meta


def _score_joint_curve(joint_name: str, curve: np.ndarray, template: np.ndarray) -> tuple[float, dict]:
    dist = _shape_distance(curve, template)
    rom = float(np.ptp(curve))
    peak_idx = int(np.argmax(curve))
    trough_idx = int(np.argmin(curve))

    if joint_name == "hip":
        penalty = 0.0
        penalty += max(0.0, 25.0 - rom) / 8.0
        penalty += max(0.0, rom - 55.0) / 10.0
        penalty += abs(curve[0] - 30.0) / 10.0
        penalty += abs(curve[55] + 12.0) / 10.0
        penalty += abs(curve[-1] - 30.0) / 10.0
        score = dist + 0.28 * penalty
    elif joint_name == "knee":
        penalty = 0.0
        penalty += max(0.0, 35.0 - rom) / 8.0
        penalty += max(0.0, rom - 80.0) / 10.0
        penalty += abs(curve[0] - 5.0) / 8.0
        penalty += abs(curve[10] - 15.0) / 8.0
        penalty += abs(curve[30] - 2.0) / 8.0
        penalty += abs(curve[72] - 55.0) / 10.0
        penalty += abs(curve[-1] - 5.0) / 8.0
        penalty += abs(peak_idx - 72) / 10.0
        score = dist + 0.25 * penalty
    elif joint_name == "ankle":
        penalty = 0.0
        penalty += max(0.0, 20.0 - rom) / 8.0
        penalty += max(0.0, rom - 45.0) / 10.0
        penalty += abs(curve[0] - 0.0) / 8.0
        penalty += abs(curve[10] + 6.0) / 8.0
        penalty += abs(curve[45] - 10.0) / 8.0
        penalty += abs(curve[67] + 18.0) / 10.0
        penalty += abs(curve[-1] - 0.0) / 8.0
        penalty += abs(trough_idx - 67) / 10.0
        score = dist + 0.25 * penalty
    else:
        raise ValueError(f"Unsupported joint '{joint_name}'")

    return score, {
        "shape_distance": float(dist),
        "rom_deg": rom,
        "peak_pct": float(peak_idx),
        "trough_pct": float(trough_idx),
    }


def _pick_best_joint_curve(
    joint_name: str,
    joint_time: np.ndarray,
    joint_df: pd.DataFrame,
    start_t: float,
    end_t: float,
    n_points: int,
    template: np.ndarray,
) -> dict:
    mask = (joint_time >= start_t) & (joint_time <= end_t)
    if int(np.count_nonzero(mask)) < 5:
        raise ValueError("Not enough joint-angle samples inside the selected cycle.")

    time_seg = joint_time[mask]
    best: dict | None = None
    prefix = JOINT_PREFIXES[joint_name]

    for axis in AXES:
        col = f"{prefix}_{axis}"
        if col not in joint_df.columns:
            continue
        values = pd.to_numeric(joint_df[col], errors="coerce").to_numpy(dtype=float)[mask]
        if not np.all(np.isfinite(values)):
            continue
        norm_curve = _normalize_vector(time_seg, values, n_points)
        for sign in SIGNS:
            candidate = sign * norm_curve
            score, qc = _score_joint_curve(joint_name, candidate, template)
            item = {
                "joint": joint_name,
                "axis": axis,
                "sign": int(sign),
                "curve": candidate,
                "score": float(score),
                "rom_deg": float(qc["rom_deg"]),
                "shape_distance": float(qc["shape_distance"]),
                "peak_pct": float(qc["peak_pct"]),
                "trough_pct": float(qc["trough_pct"]),
            }
            if best is None or item["score"] < best["score"]:
                best = item

    if best is None:
        raise ValueError(f"No valid candidate curves found for {joint_name}.")
    return best


def _plot_angle_overlay(
    out_path: Path,
    pct: np.ndarray,
    stacks: dict[str, np.ndarray],
    medians: dict[str, np.ndarray],
    templates: dict[str, np.ndarray],
    subject: str,
    n_selected: int,
    max_curves_draw: int = 200,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 9.5), sharex=True, constrained_layout=True)
    panels = [
        ("ANKLE", "ankle", "ankle_dorsiflexion"),
        ("KNEE", "knee", "knee_flexion"),
        ("HIP", "hip", "hip_flexion"),
    ]

    for ax, (title, key, ylabel) in zip(axes, panels):
        stack = stacks[key]
        if stack.shape[0] > max_curves_draw:
            idx = np.linspace(0, stack.shape[0] - 1, max_curves_draw, dtype=int)
            stack = stack[idx]
        for curve in stack:
            ax.plot(pct, curve, color="gray", alpha=0.14, linewidth=0.8)
        ax.plot(pct, templates[key], color="#2f6f7e", linewidth=1.3, linestyle="--", alpha=0.95)
        ax.plot(pct, medians[key], color="black", linewidth=2.2)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlim(0.0, 100.0)
        ax.grid(alpha=0.18)
        ax.axhline(0.0, color="#6f7e86", linewidth=0.7, alpha=0.5)

    axes[-1].set_xlabel("PERCENT OF GAIT CYCLE (%)")
    fig.suptitle(f"{subject} canonical gait profile (selected cycles={n_selected})")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _subject_trial_triplets(subject_dir: Path) -> list[tuple[Path, Path, Path]]:
    triplets: list[tuple[Path, Path, Path]] = []
    for traj_path in sorted(subject_dir.glob("*_Trajectories.csv")):
        events_path = traj_path.with_name(traj_path.name.replace("_Trajectories.csv", "_gaitEvents.yaml"))
        joint_path = traj_path.with_name(traj_path.name.replace("_Trajectories.csv", "_jointAngles.csv"))
        if events_path.exists() and joint_path.exists():
            triplets.append((traj_path, joint_path, events_path))
    return triplets


def _selection_modes(rows: list[dict]) -> dict[str, dict]:
    out = {}
    for joint_name in JOINT_PREFIXES:
        counter = Counter((row[f"{joint_name}_axis"], row[f"{joint_name}_sign"]) for row in rows)
        if not counter:
            continue
        (axis, sign), count = counter.most_common(1)[0]
        out[joint_name] = {
            "axis": axis,
            "sign": int(sign),
            "count": int(count),
            "ratio": float(count / max(1, len(rows))),
        }
    return out


def process_subject(
    subject_dir: Path,
    out_root: Path,
    plots_root: Path,
    n_points: int,
    min_stride_s: float,
    max_stride_s: float,
    keep_percentile: float,
    require_toe_off: bool,
) -> dict:
    subject = subject_dir.name
    triplets = _subject_trial_triplets(subject_dir)
    if not triplets:
        raise FileNotFoundError(f"No trajectory/joint/events triplets found under {subject_dir}")

    out_dir = out_root / subject
    out_dir.mkdir(parents=True, exist_ok=True)

    pct = np.linspace(0.0, 100.0, n_points)
    norm_time = np.linspace(0.0, 1.0, n_points)
    templates = _canonical_templates(pct)

    markers_ref: list[str] | None = None
    frame_examples: list[dict] = []
    cycle_rows: list[dict] = []
    candidates: list[dict] = []

    n_trials = 0
    n_event_pairs_seen = 0
    n_event_pairs_kept = 0

    for traj_path, joint_path, events_path in triplets:
        n_trials += 1

        traj_df = pd.read_csv(traj_path)
        joint_df = pd.read_csv(joint_path)

        if "time" not in traj_df.columns:
            raise KeyError(f"Missing time column in {traj_path}")
        if "time" not in joint_df.columns:
            raise KeyError(f"Missing time column in {joint_path}")

        traj_time = pd.to_numeric(traj_df["time"], errors="coerce").to_numpy(dtype=float)
        joint_time = pd.to_numeric(joint_df["time"], errors="coerce").to_numpy(dtype=float)

        markers = _available_markers(traj_df)
        if markers_ref is None:
            markers_ref = markers
        elif markers != markers_ref:
            raise ValueError(f"Marker set mismatch in {traj_path.name}")

        marker_data_all = _extract_marker_data(traj_df, markers)
        events = _load_yaml(events_path)
        windows, window_stats = _cycle_windows(
            events,
            min_stride_s=min_stride_s,
            max_stride_s=max_stride_s,
            require_toe_off=require_toe_off,
        )
        n_event_pairs_seen += int(window_stats["pairs_seen"])
        n_event_pairs_kept += int(window_stats["pairs_kept"])

        for window in windows:
            start_t = float(window["start_time_s"])
            end_t = float(window["end_time_s"])
            traj_mask = (traj_time >= start_t) & (traj_time <= end_t)
            traj_samples = int(np.count_nonzero(traj_mask))
            row = {
                "subject": subject,
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

                marker_parts = []
                for marker in markers_ref:
                    marker_parts.append(_normalize_matrix(time_seg, standardized[marker], n_points))
                marker_matrix = np.concatenate(marker_parts, axis=1)

                chosen = {}
                total_score = 0.0
                for joint_name, template in templates.items():
                    joint_pick = _pick_best_joint_curve(
                        joint_name=joint_name,
                        joint_time=joint_time,
                        joint_df=joint_df,
                        start_t=start_t,
                        end_t=end_t,
                        n_points=n_points,
                        template=template,
                    )
                    chosen[joint_name] = joint_pick
                    total_score += joint_pick["score"]

                cycle_rows[row_index].update(
                    {
                        "status": "candidate",
                        "total_score": float(total_score),
                        "frame_forward_axis": int(frame_meta["source_axis_order"]["x"]),
                        "frame_lateral_axis": int(frame_meta["source_axis_order"]["y"]),
                        "frame_vertical_axis": int(frame_meta["source_axis_order"]["z"]),
                        "frame_sign_x": int(frame_meta["sign_flips"]["x"]),
                        "frame_sign_y": int(frame_meta["sign_flips"]["y"]),
                        "frame_sign_z": int(frame_meta["sign_flips"]["z"]),
                    }
                )
                for joint_name in JOINT_PREFIXES:
                    cycle_rows[row_index][f"{joint_name}_axis"] = chosen[joint_name]["axis"]
                    cycle_rows[row_index][f"{joint_name}_sign"] = int(chosen[joint_name]["sign"])
                    cycle_rows[row_index][f"{joint_name}_score"] = float(chosen[joint_name]["score"])
                    cycle_rows[row_index][f"{joint_name}_rom_deg"] = float(chosen[joint_name]["rom_deg"])
                    cycle_rows[row_index][f"{joint_name}_shape_distance"] = float(chosen[joint_name]["shape_distance"])

                candidates.append(
                    {
                        "row_index": row_index,
                        "marker_matrix": marker_matrix,
                        "hip_curve": chosen["hip"]["curve"],
                        "knee_curve": chosen["knee"]["curve"],
                        "ankle_curve": chosen["ankle"]["curve"],
                        "score": float(total_score),
                        "hip_axis": chosen["hip"]["axis"],
                        "hip_sign": int(chosen["hip"]["sign"]),
                        "knee_axis": chosen["knee"]["axis"],
                        "knee_sign": int(chosen["knee"]["sign"]),
                        "ankle_axis": chosen["ankle"]["axis"],
                        "ankle_sign": int(chosen["ankle"]["sign"]),
                    }
                )
                if len(frame_examples) < 3:
                    frame_examples.append(frame_meta)
            except Exception as exc:  # noqa: BLE001
                cycle_rows[row_index]["status"] = "fail"
                cycle_rows[row_index]["error"] = str(exc)

    if not candidates or markers_ref is None:
        raise RuntimeError(f"No valid candidate cycles found for {subject}")

    candidate_scores = np.asarray([candidate["score"] for candidate in candidates], dtype=float)
    score_threshold = float(np.percentile(candidate_scores, keep_percentile))
    keep_idx = np.where(candidate_scores <= score_threshold)[0]

    min_keep = min(len(candidates), max(MIN_KEEP_CYCLES, int(np.ceil(len(candidates) * keep_percentile / 100.0))))
    if keep_idx.size < min_keep:
        keep_idx = np.argsort(candidate_scores)[:min_keep]

    selected_rows = [candidates[idx] for idx in sorted(set(keep_idx.tolist()))]
    mode_info = _selection_modes(selected_rows)

    mode_filtered_rows = [
        row
        for row in selected_rows
        if all(
            row[f"{joint_name}_axis"] == mode_info[joint_name]["axis"]
            and row[f"{joint_name}_sign"] == mode_info[joint_name]["sign"]
            for joint_name in mode_info
        )
    ]
    mode_filter_applied = False
    if len(mode_filtered_rows) >= max(MIN_MODE_KEEP_CYCLES, int(0.6 * len(selected_rows))):
        selected_rows = mode_filtered_rows
        mode_filter_applied = True

    for row in selected_rows:
        cycle_rows[row["row_index"]]["selected"] = True
        cycle_rows[row["row_index"]]["status"] = "selected"

    traj_stack = np.stack([row["marker_matrix"] for row in selected_rows], axis=0)
    hip_stack = np.stack([row["hip_curve"] for row in selected_rows], axis=0)
    knee_stack = np.stack([row["knee_curve"] for row in selected_rows], axis=0)
    ankle_stack = np.stack([row["ankle_curve"] for row in selected_rows], axis=0)

    traj_median = np.nanmedian(traj_stack, axis=0)
    traj_std = np.nanstd(traj_stack, axis=0)
    hip_median = np.nanmedian(hip_stack, axis=0)
    knee_median = np.nanmedian(knee_stack, axis=0)
    ankle_median = np.nanmedian(ankle_stack, axis=0)

    traj_columns = ["time", "pct"]
    for marker in markers_ref:
        traj_columns.extend([f"{marker}_x", f"{marker}_y", f"{marker}_z"])

    traj_csv = out_dir / f"{subject}_canonical_Trajectories.csv"
    pd.DataFrame(
        np.column_stack([norm_time, pct, traj_median]),
        columns=traj_columns,
    ).to_csv(traj_csv, index=False)

    angle_csv = out_dir / f"{subject}_canonical_marker_angles_norm101.csv"
    pd.DataFrame(
        {
            "pct": pct,
            "hip_flexion": hip_median,
            "knee_flexion": knee_median,
            "ankle_dorsiflexion": ankle_median,
        }
    ).to_csv(angle_csv, index=False)

    np.savez(
        out_dir / f"{subject}_canonical_Trajectories.npz",
        time=norm_time,
        pct=pct,
        markers=np.array(markers_ref, dtype=object),
        trajectories_median=traj_median,
        trajectories_std=traj_std,
        hip_flexion=hip_median,
        knee_flexion=knee_median,
        ankle_dorsiflexion=ankle_median,
        hip_stack=hip_stack,
        knee_stack=knee_stack,
        ankle_stack=ankle_stack,
    )

    metrics_df = pd.DataFrame(cycle_rows)
    metrics_csv = out_dir / f"{subject}_canonical_cycle_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    plot_path = plots_root / subject / f"{subject}_canonical_profiles.png"
    _plot_angle_overlay(
        plot_path,
        pct,
        stacks={"hip": hip_stack, "knee": knee_stack, "ankle": ankle_stack},
        medians={"hip": hip_median, "knee": knee_median, "ankle": ankle_median},
        templates=templates,
        subject=subject,
        n_selected=int(len(selected_rows)),
    )

    selected_df = metrics_df[metrics_df["selected"].fillna(False)].copy()
    summary = {
        "subject": subject,
        "trials_seen": int(n_trials),
        "event_pairs_seen": int(n_event_pairs_seen),
        "event_pairs_kept": int(n_event_pairs_kept),
        "candidate_cycles": int(len(candidates)),
        "selected_cycles": int(len(selected_rows)),
        "keep_percentile": float(keep_percentile),
        "score_threshold": float(score_threshold),
        "mode_filter_applied": bool(mode_filter_applied),
        "trajectory_file": traj_csv.name,
        "angles_file": angle_csv.name,
        "metrics_file": metrics_csv.name,
        "plot_file": plot_path.name,
        "coordinate_frame": {
            "translation": "All markers are centered at PELV of the first sample of each cycle.",
            "x": "forward progression",
            "y": "right lateral",
            "z": "vertical up",
        },
        "cycle_definition": {
            "start_event": "r_heel_strike",
            "end_event": "next_r_heel_strike",
            "require_r_toe_off_inside_cycle": bool(require_toe_off),
            "min_stride_s": float(min_stride_s),
            "max_stride_s": float(max_stride_s),
        },
        "selection_templates": {
            "hip": "starts flexed, extends through stance, returns to flexion in swing",
            "knee": "small flexion at load response, near extension in mid-stance, large swing flexion peak",
            "ankle": "plantarflexion after contact, dorsiflexion in mid-stance, plantarflexion in push-off",
        },
        "selection_modes": mode_info,
        "selected_stride_time_s": {
            "min": float(selected_df["stride_time_s"].min()) if not selected_df.empty else None,
            "median": float(selected_df["stride_time_s"].median()) if not selected_df.empty else None,
            "max": float(selected_df["stride_time_s"].max()) if not selected_df.empty else None,
        },
        "selected_median_rom_deg": {
            "hip": float(selected_df["hip_rom_deg"].median()) if "hip_rom_deg" in selected_df.columns and not selected_df.empty else None,
            "knee": float(selected_df["knee_rom_deg"].median()) if "knee_rom_deg" in selected_df.columns and not selected_df.empty else None,
            "ankle": float(selected_df["ankle_rom_deg"].median()) if "ankle_rom_deg" in selected_df.columns and not selected_df.empty else None,
        },
        "frame_examples": frame_examples,
    }
    _write_yaml(out_dir / f"{subject}_canonical_summary.yaml", summary)

    return {
        "subject": subject,
        "trials_seen": int(n_trials),
        "event_pairs_seen": int(n_event_pairs_seen),
        "event_pairs_kept": int(n_event_pairs_kept),
        "candidate_cycles": int(len(candidates)),
        "selected_cycles": int(len(selected_rows)),
        "keep_percentile": float(keep_percentile),
        "score_threshold": float(score_threshold),
        "trajectory_file": str(traj_csv),
        "angles_file": str(angle_csv),
        "metrics_file": str(metrics_csv),
        "plot_file": str(plot_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one canonical gait trajectory per MyPredict subject using template-based cycle selection."
    )
    parser.add_argument(
        "--eurobench-root",
        default="data/mypredict/eurobench",
        help="Root directory with MyPredict Eurobench exports.",
    )
    parser.add_argument(
        "--processed-root",
        default="data/mypredict/processed_canonical",
        help="Output directory for canonical trajectories.",
    )
    parser.add_argument(
        "--plots-root",
        default="data/mypredict/plots_canonical",
        help="Output directory for quick-look plots.",
    )
    parser.add_argument(
        "--subjects",
        default="",
        help="Comma-separated subject IDs to process. Default: all MP* folders.",
    )
    parser.add_argument("--n-points", type=int, default=101, help="Normalized samples per cycle.")
    parser.add_argument("--min-stride-s", type=float, default=0.8)
    parser.add_argument("--max-stride-s", type=float, default=2.0)
    parser.add_argument(
        "--keep-percentile",
        type=float,
        default=15.0,
        help="Keep the best-scoring cycles up to this per-subject percentile.",
    )
    parser.add_argument(
        "--no-require-toe-off",
        action="store_true",
        help="Allow cycles without r_toe_off inside the heel-strike window.",
    )
    args = parser.parse_args()

    eurobench_root = Path(args.eurobench_root)
    processed_root = Path(args.processed_root)
    plots_root = Path(args.plots_root)

    if args.subjects.strip():
        wanted = {token.strip() for token in args.subjects.split(",") if token.strip()}
        subject_dirs = [eurobench_root / subject for subject in sorted(wanted)]
    else:
        subject_dirs = sorted(path for path in eurobench_root.glob("MP*") if path.is_dir())

    if not subject_dirs:
        raise FileNotFoundError(f"No MyPredict subjects found under {eurobench_root}")

    rows = []
    for subject_dir in subject_dirs:
        if not subject_dir.exists():
            print(f"skip_missing_subject={subject_dir}")
            continue
        summary = process_subject(
            subject_dir=subject_dir,
            out_root=processed_root,
            plots_root=plots_root,
            n_points=int(args.n_points),
            min_stride_s=float(args.min_stride_s),
            max_stride_s=float(args.max_stride_s),
            keep_percentile=float(args.keep_percentile),
            require_toe_off=not args.no_require_toe_off,
        )
        rows.append(summary)
        print(
            f"subject={summary['subject']} "
            f"candidate_cycles={summary['candidate_cycles']} "
            f"selected_cycles={summary['selected_cycles']} "
            f"score_threshold={summary['score_threshold']:.3f}"
        )

    processed_root.mkdir(parents=True, exist_ok=True)
    summary_csv = processed_root / "mypredict_canonical_subjects_summary.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    print(summary_csv)


if __name__ == "__main__":
    main()
