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
    r"^subject_(?P<subject>S\d+)_cond_(?P<condition>A\d+)_run_(?P<run>\d+)_jointAngles\.csv$"
)
WALK_CONDITIONS = {"A01", "A02", "A03"}
MARKERS_BY_SIDE = {
    "r": {"heel": "right_heel", "toe": "right_big_toe"},
    "l": {"heel": "left_heel", "toe": "left_big_toe"},
}
JOINT_COLUMNS = {
    "r": {"hip": "hip_flexion_r", "knee": "knee_angle_r", "ankle": "ankle_angle_r"},
    "l": {"hip": "hip_flexion_l", "knee": "knee_angle_l", "ankle": "ankle_angle_l"},
}


@dataclass
class JointCycle:
    subject: str
    condition: str
    run: str
    side: str
    cycle_source: str
    pct: np.ndarray
    ankle: np.ndarray
    knee: np.ndarray
    hip: np.ndarray
    stride_time_s: float
    canonical_score: float | None = None
    canonical_keep: bool = True


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML content in {path}")
    return data


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=";")


def _vector(df: pd.DataFrame, base: str) -> np.ndarray:
    cols = [f"{base}_x", f"{base}_y", f"{base}_z"]
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for {base}: {', '.join(missing)}")
    return df[cols].to_numpy(dtype=float)


def _find_local_extrema(signal: np.ndarray, kind: str) -> np.ndarray:
    if signal.size < 3:
        return np.array([], dtype=int)
    diff = np.diff(signal)
    if kind == "min":
        return np.where((diff[:-1] < 0) & (diff[1:] >= 0))[0] + 1
    if kind == "max":
        return np.where((diff[:-1] > 0) & (diff[1:] <= 0))[0] + 1
    raise ValueError("kind must be 'min' or 'max'")


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
    pelvis = _vector(df, "pelvis")
    heel = _vector(df, MARKERS_BY_SIDE[side]["heel"])
    toe = _vector(df, MARKERS_BY_SIDE[side]["toe"])
    time = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)

    heel_rel = heel[:, 2] - pelvis[:, 2]
    toe_rel = toe[:, 2] - pelvis[:, 2]
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


def _cycle_candidate(events: dict, side: str) -> tuple[float, float, str] | None:
    hs = np.sort(np.asarray(events.get(f"{side}_heel_strike", []) or [], dtype=float))
    to = np.sort(np.asarray(events.get(f"{side}_toe_off", []) or [], dtype=float))

    if hs.size >= 2:
        return float(hs[0]), float(hs[1]), "events_two_hs"

    if hs.size >= 1 and to.size >= 1:
        start = float(hs[0])
        to_after = to[to > start]
        if to_after.size:
            end = start + 2.0 * (float(to_after[0]) - start)
            if end > start:
                return start, end, "events_hs_to_simulated"

    return None


def _pick_cycle(traj_df: pd.DataFrame, yaml_events: dict) -> tuple[str, float, float, str]:
    candidates: list[tuple[str, float, float, str]] = []
    for side in ("r", "l"):
        candidate = _cycle_candidate(yaml_events, side)
        if candidate is not None:
            start, end, source = candidate
            candidates.append((side, start, end, source))
            continue
        inferred = _infer_events_from_markers(traj_df, side)
        candidate = _cycle_candidate(inferred, side)
        if candidate is not None:
            start, end, source = candidate
            candidates.append((side, start, end, f"marker_{source}"))

    if not candidates:
        raise ValueError("No valid gait cycle could be defined.")

    candidates.sort(key=lambda item: (0 if item[0] == "r" else 1, item[2] - item[1]))
    return candidates[0]


def _normalize_signal(time_seg: np.ndarray, signal_seg: np.ndarray, n_points: int) -> np.ndarray:
    dt = time_seg[-1] - time_seg[0]
    if dt <= 0:
        raise ValueError("Invalid cycle duration.")
    t_norm = (time_seg - time_seg[0]) / dt
    target = np.linspace(0.0, 1.0, n_points)
    return np.interp(target, t_norm, signal_seg)


def _extract_cycle(joint_path: Path, traj_path: Path, events_path: Path, n_points: int) -> JointCycle:
    match = TRIAL_PATTERN.match(joint_path.name)
    if match is None:
        raise ValueError(f"Unexpected filename format: {joint_path.name}")

    joints = _read_csv(joint_path)
    traj = _read_csv(traj_path)
    events = _load_yaml(events_path)

    if "time" not in joints.columns:
        raise ValueError(f"Missing time column in {joint_path.name}")
    if "time" not in traj.columns:
        raise ValueError(f"Missing time column in {traj_path.name}")

    side, start_t, end_t, cycle_source = _pick_cycle(traj, events)

    time = pd.to_numeric(joints["time"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(time)
    time = time[valid]

    side_cols = JOINT_COLUMNS[side]
    hip = pd.to_numeric(joints.loc[valid, side_cols["hip"]], errors="coerce").to_numpy(dtype=float)
    knee = pd.to_numeric(joints.loc[valid, side_cols["knee"]], errors="coerce").to_numpy(dtype=float)
    ankle = pd.to_numeric(joints.loc[valid, side_cols["ankle"]], errors="coerce").to_numpy(dtype=float)

    finite = np.isfinite(time) & np.isfinite(hip) & np.isfinite(knee) & np.isfinite(ankle)
    time = time[finite]
    hip = hip[finite]
    knee = knee[finite]
    ankle = ankle[finite]

    if end_t > time[-1]:
        end_t = float(time[-1])
    mask = (time >= start_t) & (time <= end_t)
    if np.count_nonzero(mask) < 3:
        raise ValueError("Not enough samples inside the selected gait cycle.")

    time_seg = time[mask]

    return JointCycle(
        subject=match.group("subject"),
        condition=match.group("condition"),
        run=match.group("run"),
        side=side,
        cycle_source=cycle_source,
        pct=np.linspace(0.0, 100.0, n_points),
        ankle=_normalize_signal(time_seg, ankle[mask], n_points),
        knee=_normalize_signal(time_seg, knee[mask], n_points),
        hip=_normalize_signal(time_seg, hip[mask], n_points),
        stride_time_s=float(end_t - start_t),
    )


def _collect_pairs(eurobench_root: Path, conditions: set[str]) -> list[tuple[Path, Path, Path]]:
    pairs: list[tuple[Path, Path, Path]] = []
    for joint_path in sorted(eurobench_root.glob("*_jointAngles.csv")):
        match = TRIAL_PATTERN.match(joint_path.name)
        if match is None or match.group("condition") not in conditions:
            continue
        traj_path = joint_path.with_name(joint_path.name.replace("_jointAngles.csv", "_Trajectories.csv"))
        if not traj_path.exists():
            continue
        events_path = joint_path.with_name(joint_path.name.replace("_jointAngles.csv", "_gaitEvents.yaml"))
        pairs.append((joint_path, traj_path, events_path))
    return pairs


def _plot_reference_style(cycles: list[JointCycle], out_png: Path, title: str) -> None:
    pct = cycles[0].pct
    ankle = np.stack([cycle.ankle for cycle in cycles], axis=0)
    knee = np.stack([cycle.knee for cycle in cycles], axis=0)
    hip = np.stack([cycle.hip for cycle in cycles], axis=0)

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

    for ax, label, stack in zip(axes, ("ANKLE", "KNEE", "HIP"), (ankle, knee, hip)):
        mean = np.nanmean(stack, axis=0)
        mean = pd.Series(mean).rolling(window=5, center=True, min_periods=1).mean().to_numpy()
        for curve in stack:
            ax.plot(pct, curve, color="#9e9e9e", alpha=0.32, linewidth=1.0)
        ax.plot(pct, mean, color="black", linewidth=2.4)
        ax.set_title(label, fontsize=18, pad=10)
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


def _smoothed(curve: np.ndarray, window: int = 5) -> np.ndarray:
    return pd.Series(curve).rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def _interp_template(pct: np.ndarray, knots: list[tuple[float, float]]) -> np.ndarray:
    x = np.asarray([k[0] for k in knots], dtype=float)
    y = np.asarray([k[1] for k in knots], dtype=float)
    return _smoothed(np.interp(pct, x, y), window=5)


def _canonical_templates(pct: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "ankle": _interp_template(
            pct,
            [(0, 0), (5, -6), (12, -2), (30, 6), (48, 10), (58, 6), (66, -16), (78, -2), (90, 2), (100, 0)],
        ),
        "knee": _interp_template(
            pct,
            [(0, 8), (8, 12), (15, 8), (30, 3), (45, 2), (55, 8), (62, 20), (68, 40), (73, 60), (80, 45), (90, 12), (100, 8)],
        ),
        "hip": _interp_template(
            pct,
            [(0, 30), (10, 28), (20, 25), (35, 18), (50, 10), (60, 7), (70, 10), (80, 18), (90, 26), (100, 30)],
        ),
    }


def _zscore(curve: np.ndarray) -> np.ndarray:
    curve = np.asarray(curve, dtype=float)
    std = float(np.nanstd(curve))
    if not np.isfinite(std) or std < 1e-6:
        return np.zeros_like(curve)
    return (curve - float(np.nanmean(curve))) / std


def _shape_distance(curve: np.ndarray, template: np.ndarray) -> float:
    curve_s = _zscore(_smoothed(curve))
    templ_s = _zscore(template)
    rmse = float(np.sqrt(np.nanmean((curve_s - templ_s) ** 2)))
    corr = np.corrcoef(curve_s, templ_s)[0, 1]
    if not np.isfinite(corr):
        corr = 0.0
    return 0.65 * rmse + 0.35 * (1.0 - corr)


def _range_penalty(value_range: float, lower: float, upper: float, scale: float) -> float:
    if value_range < lower:
        return float((lower - value_range) / scale)
    if value_range > upper:
        return float((value_range - upper) / scale)
    return 0.0


def _annotate_canonical_similarity(
    cycles: list[JointCycle],
    keep_percentile: float,
    min_cycles_per_condition: int,
) -> None:
    by_condition: dict[str, list[JointCycle]] = {}
    for cycle in cycles:
        by_condition.setdefault(cycle.condition, []).append(cycle)

    for subset in by_condition.values():
        templates = _canonical_templates(subset[0].pct)
        ankle_ranges = [float(np.ptp(cycle.ankle)) for cycle in subset]
        use_ankle = float(np.nanmedian(ankle_ranges)) > 2.0
        scores = []
        for cycle in subset:
            knee_score = _shape_distance(cycle.knee, templates["knee"])
            hip_score = _shape_distance(cycle.hip, templates["hip"])
            ankle_score = _shape_distance(cycle.ankle, templates["ankle"]) if use_ankle else 0.0

            knee_range = float(np.ptp(cycle.knee))
            hip_range = float(np.ptp(cycle.hip))
            ankle_range = float(np.ptp(cycle.ankle))

            range_pen = 0.0
            range_pen += _range_penalty(knee_range, lower=18.0, upper=75.0, scale=12.0)
            range_pen += _range_penalty(hip_range, lower=8.0, upper=40.0, scale=8.0)
            if use_ankle:
                range_pen += 0.5 * _range_penalty(ankle_range, lower=6.0, upper=35.0, scale=8.0)

            knee_peak_idx = int(np.argmax(_smoothed(cycle.knee)))
            peak_pen = abs(knee_peak_idx - 72) / 18.0

            score = 0.7 * knee_score + 0.3 * hip_score + range_pen + peak_pen
            if use_ankle:
                score += 0.15 * ankle_score
            scores.append(score)

        scores = np.asarray(scores, dtype=float)
        threshold = float(np.nanpercentile(scores, keep_percentile))
        keep_mask = scores <= threshold

        min_keep = min(len(subset), max(3, min_cycles_per_condition))
        if int(np.count_nonzero(keep_mask)) < min_keep:
            keep_mask = np.zeros(len(subset), dtype=bool)
            keep_mask[np.argsort(scores)[:min_keep]] = True

        for cycle, score, keep in zip(subset, scores, keep_mask):
            cycle.canonical_score = float(score)
            cycle.canonical_keep = bool(keep)


def _save_metrics(cycles: list[JointCycle], out_csv: Path) -> pd.DataFrame:
    rows = [
        {
            "subject": cycle.subject,
            "condition": cycle.condition,
            "run": cycle.run,
            "side": cycle.side,
            "cycle_source": cycle.cycle_source,
            "stride_time_s": cycle.stride_time_s,
            "canonical_score": cycle.canonical_score,
            "canonical_keep": cycle.canonical_keep,
        }
        for cycle in cycles
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def _save_long_csv(cycles: list[JointCycle], out_csv: Path) -> None:
    rows = []
    for cycle in cycles:
        for idx, pct in enumerate(cycle.pct):
            rows.append(
                {
                    "subject": cycle.subject,
                    "condition": cycle.condition,
                    "run": cycle.run,
                    "side": cycle.side,
                    "pct": float(pct),
                    "ankle": float(cycle.ankle[idx]),
                    "knee": float(cycle.knee[idx]),
                    "hip": float(cycle.hip[idx]),
                }
            )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot reference-style gait joint-angle profiles for Multimodal video and IMU kinematic."
    )
    parser.add_argument(
        "--eurobench-root",
        default="data/Multimodal video and IMU kinematic/eurobench",
        help="Root with flat EUROBENCH files.",
    )
    parser.add_argument(
        "--analysis-root",
        default="data/Multimodal video and IMU kinematic/analysis/joint_angles",
        help="Output folder for CSV summaries.",
    )
    parser.add_argument(
        "--plots-root",
        default="data/Multimodal video and IMU kinematic/plots/joint_angles",
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
    parser.add_argument(
        "--shape-filter-percentile",
        type=float,
        default=50.0,
        help="Keep cycles with canonical score up to this within-condition percentile.",
    )
    parser.add_argument(
        "--min-cycles-per-condition",
        type=int,
        default=6,
        help="Minimum number of kept cycles per condition after filtering.",
    )
    args = parser.parse_args()

    eurobench_root = Path(args.eurobench_root)
    analysis_root = Path(args.analysis_root)
    plots_root = Path(args.plots_root)
    conditions = {cond for cond in args.conditions if cond in WALK_CONDITIONS}
    if not conditions:
        raise ValueError("No valid walking conditions selected.")

    pairs = _collect_pairs(eurobench_root, conditions)
    if not pairs:
        raise FileNotFoundError(f"No jointAngles/Trajectories pairs found under {eurobench_root}")

    cycles: list[JointCycle] = []
    skipped: list[dict[str, str]] = []

    for joint_path, traj_path, events_path in pairs:
        try:
            cycles.append(_extract_cycle(joint_path, traj_path, events_path, args.n_points))
        except Exception as exc:
            skipped.append(
                {
                    "jointAngles": joint_path.name,
                    "trajectories": traj_path.name,
                    "events": events_path.name,
                    "reason": str(exc),
                }
            )

    if not cycles:
        raise RuntimeError("No valid joint-angle gait cycles could be extracted.")

    _annotate_canonical_similarity(
        cycles,
        keep_percentile=args.shape_filter_percentile,
        min_cycles_per_condition=args.min_cycles_per_condition,
    )
    filtered_cycles = [cycle for cycle in cycles if cycle.canonical_keep]
    if not filtered_cycles:
        raise RuntimeError("Shape filter rejected all cycles.")

    metrics_csv = analysis_root / "trial_cycle_metrics.csv"
    long_csv = analysis_root / "joint_cycles_long.csv"
    skipped_csv = analysis_root / "skipped_trials.csv"
    summary_yaml = analysis_root / "analysis_summary.yaml"

    metrics_df = _save_metrics(cycles, metrics_csv)
    _save_long_csv(cycles, long_csv)
    if skipped:
        pd.DataFrame(skipped).to_csv(skipped_csv, index=False)

    for condition in sorted({cycle.condition for cycle in cycles}):
        subset = [cycle for cycle in cycles if cycle.condition == condition]
        _plot_reference_style(
            subset,
            plots_root / f"{condition}_joint_angles_reference_style_hip_knee_ankle.png",
            title=f"Multimodal video and IMU kinematic {condition} (n={len(subset)})",
        )
        filtered_subset = [cycle for cycle in subset if cycle.canonical_keep]
        if filtered_subset:
            _plot_reference_style(
                filtered_subset,
                plots_root / f"{condition}_joint_angles_reference_style_hip_knee_ankle_filtered.png",
                title=f"Multimodal video and IMU kinematic {condition} filtered (n={len(filtered_subset)})",
            )

    _plot_reference_style(
        cycles,
        plots_root / "all_joint_angles_reference_style_hip_knee_ankle.png",
        title=f"Multimodal video and IMU kinematic joint angles (n={len(cycles)})",
    )
    _plot_reference_style(
        filtered_cycles,
        plots_root / "all_joint_angles_reference_style_hip_knee_ankle_filtered.png",
        title=f"Multimodal video and IMU kinematic joint angles filtered (n={len(filtered_cycles)})",
    )

    summary = {
        "eurobench_root": str(eurobench_root),
        "n_pairs_found": len(pairs),
        "n_cycles_used": len(cycles),
        "n_cycles_filtered_kept": len(filtered_cycles),
        "n_trials_skipped": len(skipped),
        "shape_filter_percentile": args.shape_filter_percentile,
        "min_cycles_per_condition": args.min_cycles_per_condition,
        "conditions": sorted({cycle.condition for cycle in cycles}),
        "plots": [str(plots_root / "all_joint_angles_reference_style_hip_knee_ankle.png")]
        + [str(plots_root / "all_joint_angles_reference_style_hip_knee_ankle_filtered.png")]
        + [str(plots_root / f"{condition}_joint_angles_reference_style_hip_knee_ankle.png") for condition in sorted({cycle.condition for cycle in cycles})],
        "analysis_files": {
            "trial_cycle_metrics": str(metrics_csv),
            "joint_cycles_long": str(long_csv),
            "skipped_trials": str(skipped_csv) if skipped else None,
        },
        "cycle_source_counts": metrics_df["cycle_source"].value_counts(dropna=False).to_dict(),
        "canonical_keep_counts": {
            "kept": int(metrics_df["canonical_keep"].fillna(False).sum()),
            "rejected": int((~metrics_df["canonical_keep"].fillna(False)).sum()),
        },
    }
    summary_yaml.parent.mkdir(parents=True, exist_ok=True)
    with summary_yaml.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)

    print(f"pairs_found={len(pairs)}")
    print(f"cycles_used={len(cycles)}")
    print(f"cycles_kept_after_shape_filter={len(filtered_cycles)}")
    print(f"trials_skipped={len(skipped)}")
    print(f"plot_all={plots_root / 'all_joint_angles_reference_style_hip_knee_ankle.png'}")
    print(f"plot_all_filtered={plots_root / 'all_joint_angles_reference_style_hip_knee_ankle_filtered.png'}")
    print(f"metrics={metrics_csv}")


if __name__ == "__main__":
    main()
