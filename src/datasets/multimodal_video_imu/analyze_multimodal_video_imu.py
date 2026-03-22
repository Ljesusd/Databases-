import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


TRIAL_PATTERN = re.compile(
    r"^subject_(?P<subject>S\d+)_cond_(?P<condition>A\d+)_run_(?P<run>\d+)_Trajectories\.csv$"
)
WALK_CONDITIONS = {"A01", "A02", "A03"}
VERTICAL_AXIS = "z"
PROGRESSION_AXES = ("x", "y")
MARKERS_BY_SIDE = {
    "r": {"hip": "right_hip", "knee": "right_knee", "ankle": "right_ankle", "heel": "right_heel", "toe": "right_big_toe"},
    "l": {"hip": "left_hip", "knee": "left_knee", "ankle": "left_ankle", "heel": "left_heel", "toe": "left_big_toe"},
}
MARKER_COLORS = {"hip": "#1f77b4", "knee": "#ff7f0e", "ankle": "#2ca02c"}


@dataclass
class TrialCycle:
    subject: str
    condition: str
    run: str
    side: str
    cycle_source: str
    pct: np.ndarray
    hip: np.ndarray
    knee: np.ndarray
    ankle: np.ndarray
    stride_time_s: float
    stance_time_s: float | None
    swing_time_s: float | None
    stance_pct: float | None
    swing_pct: float | None
    stride_displacement: float
    speed_units_per_s: float


def _load_events(path: Path) -> dict:
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid events YAML: {path}")
    return data


def _load_trajectories(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    if "time" not in df.columns:
        raise ValueError(f"Missing 'time' column in {path.name}")
    return df


def _vector(df: pd.DataFrame, base: str) -> np.ndarray:
    cols = [f"{base}_x", f"{base}_y", f"{base}_z"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for {base}: {', '.join(missing)}")
    return df[cols].to_numpy(dtype=float)


def _find_local_extrema(signal: np.ndarray, kind: str) -> np.ndarray:
    if signal.size < 3:
        return np.array([], dtype=int)
    diff = np.diff(signal)
    if kind == "max":
        return np.where((diff[:-1] > 0) & (diff[1:] <= 0))[0] + 1
    if kind == "min":
        return np.where((diff[:-1] < 0) & (diff[1:] >= 0))[0] + 1
    raise ValueError("kind must be 'max' or 'min'")


def _enforce_min_interval(indices: np.ndarray, time: np.ndarray, min_interval_s: float) -> np.ndarray:
    if indices.size == 0:
        return indices
    sorted_idx = indices[np.argsort(time[indices])]
    kept: list[int] = []
    last_t = -np.inf
    for idx in sorted_idx:
        if time[idx] - last_t >= min_interval_s:
            kept.append(int(idx))
            last_t = time[idx]
    return np.asarray(kept, dtype=int)


def _infer_events_from_markers(df: pd.DataFrame, side: str) -> dict[str, list[float]]:
    side_markers = MARKERS_BY_SIDE[side]
    pelvis = _vector(df, "pelvis")
    heel = _vector(df, side_markers["heel"])
    toe = _vector(df, side_markers["toe"])
    time = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)

    axis_idx = 2
    heel_rel = heel[:, axis_idx] - pelvis[:, axis_idx]
    toe_rel = toe[:, axis_idx] - pelvis[:, axis_idx]

    hs_idx = _find_local_extrema(heel_rel, "min")
    to_idx = _find_local_extrema(toe_rel, "min")

    if hs_idx.size:
        hs_thresh = np.percentile(heel_rel[np.isfinite(heel_rel)], 30)
        hs_idx = hs_idx[heel_rel[hs_idx] <= hs_thresh]
        hs_idx = _enforce_min_interval(hs_idx, time, min_interval_s=0.3)
    if to_idx.size:
        to_thresh = np.percentile(toe_rel[np.isfinite(toe_rel)], 30)
        to_idx = to_idx[toe_rel[to_idx] <= to_thresh]
        to_idx = _enforce_min_interval(to_idx, time, min_interval_s=0.3)

    return {
        f"{side}_heel_strike": time[hs_idx].tolist(),
        f"{side}_toe_off": time[to_idx].tolist(),
    }


def _cycle_candidate(events: dict, side: str) -> tuple[float, float, float | None, str] | None:
    hs = np.sort(np.asarray(events.get(f"{side}_heel_strike", []) or [], dtype=float))
    to = np.sort(np.asarray(events.get(f"{side}_toe_off", []) or [], dtype=float))

    if hs.size >= 2:
        start = float(hs[0])
        end = float(hs[1])
        toe_between = to[(to > start) & (to < end)]
        toe = float(toe_between[0]) if toe_between.size else None
        return start, end, toe, "events_two_hs"

    if hs.size >= 1 and to.size >= 1:
        start = float(hs[0])
        to_after = to[to > start]
        if to_after.size:
            toe = float(to_after[0])
            end = start + 2.0 * (toe - start)
            if end > start:
                return start, end, toe, "events_hs_to_simulated"

    return None


def _pick_cycle(df: pd.DataFrame, yaml_events: dict) -> tuple[str, float, float, float | None, str]:
    candidates: list[tuple[str, float, float, float | None, str]] = []
    for side in ("r", "l"):
        candidate = _cycle_candidate(yaml_events, side)
        if candidate is not None:
            start, end, toe, source = candidate
            candidates.append((side, start, end, toe, source))
        else:
            inferred = _infer_events_from_markers(df, side)
            candidate = _cycle_candidate(inferred, side)
            if candidate is not None:
                start, end, toe, source = candidate
                candidates.append((side, start, end, toe, f"marker_{source}"))

    if not candidates:
        raise ValueError("No valid gait cycle could be defined from events or marker fallback.")

    candidates.sort(key=lambda item: (0 if item[0] == "r" else 1, item[2] - item[1]))
    return candidates[0]


def _normalize_vector_cycle(time_seg: np.ndarray, values: np.ndarray, n_points: int) -> np.ndarray:
    dt = time_seg[-1] - time_seg[0]
    if dt <= 0:
        raise ValueError("Invalid cycle duration.")
    t_norm = (time_seg - time_seg[0]) / dt
    target = np.linspace(0.0, 1.0, n_points)
    out = np.zeros((n_points, values.shape[1]), dtype=float)
    for idx in range(values.shape[1]):
        out[:, idx] = np.interp(target, t_norm, values[:, idx])
    return out


def _infer_progression_axis(pelvis_seg: np.ndarray) -> int:
    ranges = []
    for axis in PROGRESSION_AXES:
        idx = 0 if axis == "x" else 1
        ranges.append((idx, float(np.nanmax(pelvis_seg[:, idx]) - np.nanmin(pelvis_seg[:, idx]))))
    ranges.sort(key=lambda item: item[1], reverse=True)
    return ranges[0][0]


def _extract_cycle(traj_path: Path, events_path: Path, n_points: int) -> TrialCycle:
    match = TRIAL_PATTERN.match(traj_path.name)
    if match is None:
        raise ValueError(f"Unexpected file name: {traj_path.name}")

    df = _load_trajectories(traj_path)
    time = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
    yaml_events = _load_events(events_path)
    side, start_t, end_t, toe_t, cycle_source = _pick_cycle(df, yaml_events)

    if end_t > time[-1]:
        end_t = float(time[-1])
    if start_t >= end_t:
        raise ValueError("Cycle bounds are invalid.")

    mask = np.isfinite(time) & (time >= start_t) & (time <= end_t)
    if np.count_nonzero(mask) < 3:
        raise ValueError("Not enough samples inside the selected gait cycle.")

    side_markers = MARKERS_BY_SIDE[side]
    hip = _vector(df, side_markers["hip"])[mask]
    knee = _vector(df, side_markers["knee"])[mask]
    ankle = _vector(df, side_markers["ankle"])[mask]
    pelvis = _vector(df, "pelvis")[mask]
    time_seg = time[mask]

    progression_idx = _infer_progression_axis(pelvis)

    hip_norm = _normalize_vector_cycle(time_seg, hip, n_points)
    knee_norm = _normalize_vector_cycle(time_seg, knee, n_points)
    ankle_norm = _normalize_vector_cycle(time_seg, ankle, n_points)
    pelvis_norm = _normalize_vector_cycle(time_seg, pelvis, n_points)

    stride_time = float(end_t - start_t)
    stance_time = None if toe_t is None or toe_t <= start_t or toe_t >= end_t else float(toe_t - start_t)
    swing_time = None if stance_time is None else float(stride_time - stance_time)
    stance_pct = None if stance_time is None else float(100.0 * stance_time / stride_time)
    swing_pct = None if swing_time is None else float(100.0 * swing_time / stride_time)
    stride_displacement = float(abs(pelvis_norm[-1, progression_idx] - pelvis_norm[0, progression_idx]))
    speed_units_per_s = float(stride_displacement / stride_time) if stride_time > 0 else float("nan")

    return TrialCycle(
        subject=match.group("subject"),
        condition=match.group("condition"),
        run=match.group("run"),
        side=side,
        cycle_source=cycle_source,
        pct=np.linspace(0.0, 100.0, n_points),
        hip=hip_norm,
        knee=knee_norm,
        ankle=ankle_norm,
        stride_time_s=stride_time,
        stance_time_s=stance_time,
        swing_time_s=swing_time,
        stance_pct=stance_pct,
        swing_pct=swing_pct,
        stride_displacement=stride_displacement,
        speed_units_per_s=speed_units_per_s,
    )


def _trial_pairs(eurobench_root: Path, conditions: set[str]) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for traj in sorted(eurobench_root.glob("*_Trajectories.csv")):
        match = TRIAL_PATTERN.match(traj.name)
        if match is None or match.group("condition") not in conditions:
            continue
        events = traj.with_name(traj.name.replace("_Trajectories.csv", "_gaitEvents.yaml"))
        if events.exists():
            pairs.append((traj, events))
    return pairs


def _marker_stack(cycles: list[TrialCycle], marker: str) -> np.ndarray:
    return np.stack([getattr(cycle, marker) for cycle in cycles], axis=0)


def _relative_progression_and_vertical(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    progression = stack[:, :, 0].copy()
    vertical = stack[:, :, 2].copy()
    progression = progression - progression[:, [0]]
    vertical = vertical - vertical[:, [0]]
    return progression, vertical


def _plot_marker_cycles(cycles: list[TrialCycle], out_png: Path, title: str) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    fig.patch.set_facecolor("white")

    for row_idx, marker in enumerate(("hip", "knee", "ankle")):
        stack = _marker_stack(cycles, marker)
        progression, vertical = _relative_progression_and_vertical(stack)
        pct = cycles[0].pct
        mean_vertical = np.nanmean(vertical, axis=0)
        mean_progression = np.nanmean(progression, axis=0)
        color = MARKER_COLORS[marker]

        ax_profile = axes[row_idx, 0]
        ax_path = axes[row_idx, 1]

        for curve in vertical:
            ax_profile.plot(pct, curve, color="#C9CDD4", alpha=0.25, linewidth=0.9)
        ax_profile.plot(pct, mean_vertical, color=color, linewidth=2.4)
        ax_profile.set_title(f"{marker.upper()} vertical excursion")
        ax_profile.set_xlim(0.0, 100.0)
        ax_profile.set_ylabel("relative z")
        ax_profile.grid(alpha=0.2)

        for idx in range(progression.shape[0]):
            ax_path.plot(progression[idx], vertical[idx], color="#C9CDD4", alpha=0.2, linewidth=0.9)
        ax_path.plot(mean_progression, mean_vertical, color=color, linewidth=2.4)
        ax_path.scatter([mean_progression[0]], [mean_vertical[0]], color=color, s=30, zorder=3)
        ax_path.set_title(f"{marker.upper()} sagittal path")
        ax_path.set_xlabel("relative progression")
        ax_path.set_ylabel("relative z")
        ax_path.grid(alpha=0.2)

    axes[2, 0].set_xlabel("gait cycle (%)")
    fig.suptitle(title, fontsize=15)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _plot_marker_reference_style(cycles: list[TrialCycle], out_png: Path, title: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True, constrained_layout=True)
    bg = "#ebebeb"
    fig.patch.set_facecolor(bg)

    for ax in axes:
        ax.set_facecolor(bg)
        ax.grid(False)
        ax.set_yticks([])
        ax.tick_params(axis="y", length=0)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    pct = cycles[0].pct
    marker_order = ("ankle", "knee", "hip")

    for ax, marker in zip(axes, marker_order):
        stack = _marker_stack(cycles, marker)
        _, vertical = _relative_progression_and_vertical(stack)
        mean_vertical = np.nanmean(vertical, axis=0)
        mean_vertical = pd.Series(mean_vertical).rolling(window=5, center=True, min_periods=1).mean().to_numpy()

        for curve in vertical:
            ax.plot(pct, curve, color="#9e9e9e", alpha=0.32, linewidth=1.0)
        ax.plot(pct, mean_vertical, color="black", linewidth=2.4)
        ax.set_title(marker.upper(), fontsize=18, pad=10)
        ax.set_xlim(0.0, 100.0)

    axes[-1].set_xlabel("PERCENT OF GAIT CYCLE (%)", fontsize=16)
    axes[-1].set_xticks([0, 20, 40, 60, 80, 100])
    axes[-1].tick_params(axis="x", labelsize=14, width=1.0, length=6)
    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)

    fig.suptitle(title, fontsize=20, y=1.01)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_cycle_metrics(cycles: list[TrialCycle], out_csv: Path) -> pd.DataFrame:
    rows = []
    for cycle in cycles:
        rows.append(
            {
                "subject": cycle.subject,
                "condition": cycle.condition,
                "run": cycle.run,
                "side": cycle.side,
                "cycle_source": cycle.cycle_source,
                "stride_time_s": cycle.stride_time_s,
                "stance_time_s": cycle.stance_time_s,
                "swing_time_s": cycle.swing_time_s,
                "stance_pct": cycle.stance_pct,
                "swing_pct": cycle.swing_pct,
                "stride_displacement": cycle.stride_displacement,
                "speed_units_per_s": cycle.speed_units_per_s,
                "cadence_steps_per_min": 120.0 / cycle.stride_time_s if cycle.stride_time_s > 0 else np.nan,
            }
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def _save_condition_summary(metrics_df: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    summary = (
        metrics_df.groupby("condition", dropna=False)
        .agg(
            n_cycles=("subject", "count"),
            stride_time_s_mean=("stride_time_s", "mean"),
            stride_time_s_std=("stride_time_s", "std"),
            stance_pct_mean=("stance_pct", "mean"),
            swing_pct_mean=("swing_pct", "mean"),
            cadence_steps_per_min_mean=("cadence_steps_per_min", "mean"),
            speed_units_per_s_mean=("speed_units_per_s", "mean"),
        )
        .reset_index()
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    return summary


def _save_marker_long_csv(cycles: list[TrialCycle], out_csv: Path) -> None:
    rows = []
    for cycle in cycles:
        for idx, pct in enumerate(cycle.pct):
            row = {
                "subject": cycle.subject,
                "condition": cycle.condition,
                "run": cycle.run,
                "side": cycle.side,
                "pct": float(pct),
            }
            for marker in ("hip", "knee", "ankle"):
                values = getattr(cycle, marker)[idx]
                row[f"{marker}_x"] = float(values[0])
                row[f"{marker}_y"] = float(values[1])
                row[f"{marker}_z"] = float(values[2])
            rows.append(row)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute marker-based gait summaries and plots for Multimodal video and IMU kinematic Eurobench files."
    )
    parser.add_argument(
        "--eurobench-root",
        default="data/Multimodal video and IMU kinematic/eurobench",
        help="Root with flat EUROBENCH files.",
    )
    parser.add_argument(
        "--analysis-root",
        default="data/Multimodal video and IMU kinematic/analysis",
        help="Output folder for CSV summaries.",
    )
    parser.add_argument(
        "--plots-root",
        default="data/Multimodal video and IMU kinematic/plots",
        help="Output folder for plots.",
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=["A01", "A02", "A03"],
        help="Walking conditions to analyze.",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=101,
        help="Number of normalized points per gait cycle.",
    )
    args = parser.parse_args()

    eurobench_root = Path(args.eurobench_root)
    analysis_root = Path(args.analysis_root)
    plots_root = Path(args.plots_root)
    conditions = {cond for cond in args.conditions if cond in WALK_CONDITIONS}
    if not conditions:
        raise ValueError("No valid walking conditions selected.")

    pairs = _trial_pairs(eurobench_root, conditions)
    if not pairs:
        raise FileNotFoundError(f"No trajectory/events pairs found under {eurobench_root}")

    cycles: list[TrialCycle] = []
    skipped_rows: list[dict[str, str]] = []

    for traj_path, events_path in pairs:
        try:
            cycles.append(_extract_cycle(traj_path, events_path, n_points=args.n_points))
        except Exception as exc:
            skipped_rows.append({"trajectory": traj_path.name, "events": events_path.name, "reason": str(exc)})

    if not cycles:
        raise RuntimeError("No valid gait cycles could be extracted.")

    metrics_csv = analysis_root / "trial_cycle_metrics.csv"
    summary_csv = analysis_root / "condition_summary.csv"
    long_csv = analysis_root / "marker_cycles_long.csv"
    skipped_csv = analysis_root / "skipped_trials.csv"
    summary_yaml = analysis_root / "analysis_summary.yaml"

    metrics_df = _save_cycle_metrics(cycles, metrics_csv)
    summary_df = _save_condition_summary(metrics_df, summary_csv)
    _save_marker_long_csv(cycles, long_csv)

    if skipped_rows:
        pd.DataFrame(skipped_rows).to_csv(skipped_csv, index=False)

    for condition in sorted({cycle.condition for cycle in cycles}):
        subset = [cycle for cycle in cycles if cycle.condition == condition]
        if not subset:
            continue
        _plot_marker_cycles(
            subset,
            plots_root / f"{condition}_marker_cycles_hip_knee_ankle.png",
            title=f"Multimodal video and IMU kinematic gait markers ({condition}, n={len(subset)})",
        )
        _plot_marker_reference_style(
            subset,
            plots_root / f"{condition}_marker_cycles_reference_style_hip_knee_ankle.png",
            title=f"Multimodal video and IMU kinematic {condition} (n={len(subset)})",
        )

    _plot_marker_cycles(
        cycles,
        plots_root / "all_marker_cycles_hip_knee_ankle.png",
        title=f"Multimodal video and IMU kinematic gait markers (all conditions, n={len(cycles)})",
    )
    _plot_marker_reference_style(
        cycles,
        plots_root / "all_marker_cycles_reference_style_hip_knee_ankle.png",
        title=f"Multimodal video and IMU kinematic (n={len(cycles)})",
    )

    summary_payload = {
        "eurobench_root": str(eurobench_root),
        "n_trials_found": len(pairs),
        "n_cycles_used": len(cycles),
        "n_trials_skipped": len(skipped_rows),
        "conditions": sorted({cycle.condition for cycle in cycles}),
        "plots": [str(plots_root / "all_marker_cycles_hip_knee_ankle.png")]
        + [str(plots_root / "all_marker_cycles_reference_style_hip_knee_ankle.png")]
        + [str(plots_root / f"{condition}_marker_cycles_hip_knee_ankle.png") for condition in sorted({cycle.condition for cycle in cycles})],
        "analysis_files": {
            "trial_cycle_metrics": str(metrics_csv),
            "condition_summary": str(summary_csv),
            "marker_cycles_long": str(long_csv),
            "skipped_trials": str(skipped_csv) if skipped_rows else None,
        },
        "condition_summary_preview": summary_df.to_dict(orient="records"),
    }
    summary_yaml.parent.mkdir(parents=True, exist_ok=True)
    with summary_yaml.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary_payload, handle, sort_keys=False)

    print(f"pairs_found={len(pairs)}")
    print(f"cycles_used={len(cycles)}")
    print(f"trials_skipped={len(skipped_rows)}")
    print(f"trial_metrics={metrics_csv}")
    print(f"condition_summary={summary_csv}")
    print(f"marker_cycles_long={long_csv}")
    print(f"plot_all={plots_root / 'all_marker_cycles_hip_knee_ankle.png'}")


if __name__ == "__main__":
    main()
