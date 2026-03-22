import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRIAL_RE = re.compile(
    r"^Subject(?P<subject>\d+)_(?P<speed>V\d+)_(?P<run>\d+)_Trajectories\.csv$",
    re.IGNORECASE,
)


@dataclass
class CycleSignal:
    subject: str
    speed: str
    run: str
    trial: str
    side: str
    cycle_id: int
    pct: np.ndarray
    hip: np.ndarray
    knee: np.ndarray
    ankle: np.ndarray
    ankle_z: np.ndarray


def _subject_tag(value: str) -> str:
    value = value.strip()
    m = re.fullmatch(r"Subject(?P<id>\d+)", value, flags=re.IGNORECASE)
    if m:
        return f"Subject{m.group('id').zfill(2)}"
    if re.fullmatch(r"\d+", value):
        return f"Subject{value.zfill(2)}"
    raise ValueError("Subject must look like '02' or 'Subject02'.")


def _speed_tag(value: str) -> str:
    value = value.strip().upper()
    if not re.fullmatch(r"V\d+", value):
        raise ValueError("Speed must look like V1, V15, V25, V35, etc.")
    return value


def _speed_to_kmh(speed: str) -> float:
    digits = speed[1:]
    if len(digits) == 1:
        return float(int(digits))
    return float(int(digits)) / 10.0


def _vector(df: pd.DataFrame, base: str) -> np.ndarray | None:
    cols = [f"{base}_x", f"{base}_y", f"{base}_z"]
    if all(c in df.columns for c in cols):
        return df[cols].to_numpy(dtype=float)
    return None


def _mean_vectors(*vectors: np.ndarray | None) -> np.ndarray | None:
    valid = [v for v in vectors if v is not None]
    if not valid:
        return None
    return np.nanmean(np.stack(valid, axis=0), axis=0)


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


def _normalize_cycle(time_seg: np.ndarray, values: np.ndarray, n_points: int) -> np.ndarray:
    if time_seg.size < 2:
        raise ValueError("Not enough samples to normalize cycle.")
    dt = float(time_seg[-1] - time_seg[0])
    if dt <= 0.0:
        raise ValueError("Invalid cycle duration (<= 0).")
    t_norm = (time_seg - time_seg[0]) / dt
    t_target = np.linspace(0.0, 1.0, n_points)
    return np.interp(t_target, t_norm, values)


def _smooth_signal(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < 3:
        return values.copy()
    window = int(max(3, window))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(values, kernel, mode="same")


def _local_minima(values: np.ndarray) -> np.ndarray:
    if values.size < 3:
        return np.array([], dtype=int)
    return np.where((values[1:-1] < values[:-2]) & (values[1:-1] <= values[2:]))[0] + 1


def _pick_cycle_boundaries_from_knee(
    time: np.ndarray,
    knee: np.ndarray,
    min_stride_s: float,
    max_stride_s: float,
    smooth_window_samples: int,
) -> list[tuple[float, float]]:
    if time.size < 5 or knee.size < 5:
        return []

    k_smooth = _smooth_signal(knee, window=smooth_window_samples)
    cand = _local_minima(k_smooth)
    if cand.size < 2:
        return []

    # Keep only lower minima to reduce false splits within a stride.
    thr = np.nanpercentile(k_smooth, 60.0)
    cand = cand[k_smooth[cand] <= thr]
    if cand.size < 2:
        return []

    # Greedy non-maximum suppression for minima: keep the deepest one in close neighborhoods.
    min_sep = max(0.25, 0.5 * min_stride_s)
    kept: list[int] = []
    for idx in cand:
        if not kept:
            kept.append(int(idx))
            continue
        if time[idx] - time[kept[-1]] < min_sep:
            if k_smooth[idx] < k_smooth[kept[-1]]:
                kept[-1] = int(idx)
        else:
            kept.append(int(idx))

    if len(kept) < 2:
        return []

    cycles: list[tuple[float, float]] = []
    for i in range(len(kept) - 1):
        i0 = kept[i]
        i1 = kept[i + 1]
        dt = float(time[i1] - time[i0])
        if min_stride_s <= dt <= max_stride_s:
            cycles.append((float(time[i0]), float(time[i1])))
    return cycles


def _infer_progression_axis(pelvis_center: np.ndarray) -> tuple[int, float]:
    dx = float(pelvis_center[-1, 0] - pelvis_center[0, 0])
    dy = float(pelvis_center[-1, 1] - pelvis_center[0, 1])
    if not np.isfinite(dx):
        dx = 0.0
    if not np.isfinite(dy):
        dy = 0.0
    if abs(dx) >= abs(dy):
        forward_idx = 0
        net = dx
    else:
        forward_idx = 1
        net = dy
    if net == 0.0:
        step = np.nanmedian(np.diff(pelvis_center[:, forward_idx]))
        sign = 1.0 if (np.isfinite(step) and step >= 0.0) else -1.0
    else:
        sign = 1.0 if net >= 0.0 else -1.0
    return forward_idx, sign


def _compute_side_signals(
    hip_marker: np.ndarray,
    knee_center: np.ndarray,
    ankle_center: np.ndarray,
    toe_marker: np.ndarray,
    forward_idx: int,
    vertical_idx: int,
    progression_sign: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    thigh = knee_center - hip_marker
    shank = ankle_center - knee_center
    foot = toe_marker - ankle_center

    knee_deg = _angle_between_3d(thigh, shank)
    ankle_deg = _angle_between_3d(shank, foot) - 90.0

    thigh_2d = np.stack(
        [
            progression_sign * thigh[:, forward_idx],
            thigh[:, vertical_idx],
        ],
        axis=1,
    )
    vertical_down = np.tile([0.0, -1.0], (thigh_2d.shape[0], 1))
    hip_signed = _angle_between_2d(vertical_down, thigh_2d)
    hip_unwrapped = np.degrees(np.unwrap(np.radians(hip_signed)))
    if hip_unwrapped.size and hip_unwrapped[0] > 90.0:
        hip_unwrapped = hip_unwrapped - 180.0
    elif hip_unwrapped.size and hip_unwrapped[0] < -90.0:
        hip_unwrapped = hip_unwrapped + 180.0
    hip_deg = -hip_unwrapped

    ankle_vertical = ankle_center[:, vertical_idx]
    return hip_deg, knee_deg, ankle_deg, ankle_vertical


def _analyze_trial(
    trajectories_path: Path,
    n_points: int,
    min_stride_s: float,
    max_stride_s: float,
    smooth_window_samples: int,
) -> tuple[list[CycleSignal], list[dict]]:
    df = pd.read_csv(trajectories_path)
    if "time" not in df.columns:
        raise ValueError(f"Missing 'time' in {trajectories_path.name}")

    m = TRIAL_RE.match(trajectories_path.name)
    if m is None:
        raise ValueError(f"Unexpected filename format: {trajectories_path.name}")
    subject = f"Subject{m.group('subject').zfill(2)}"
    speed = m.group("speed").upper()
    run = m.group("run").zfill(2)
    trial = trajectories_path.name.replace("_Trajectories.csv", "")
    speed_kmh = _speed_to_kmh(speed)

    time = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)

    rasis = _vector(df, "RASIS")
    lasis = _vector(df, "LASIS")
    rpsis = _vector(df, "RPSIS")
    lpsis = _vector(df, "LPSIS")
    rhip = _vector(df, "RTROC")
    lhip = _vector(df, "LTROC")

    rknee = _mean_vectors(_vector(df, "RLK"), _vector(df, "RMK"))
    lknee = _mean_vectors(_vector(df, "LLK"), _vector(df, "LMK"))
    rank = _mean_vectors(_vector(df, "RLA"), _vector(df, "RMA"))
    lank = _mean_vectors(_vector(df, "LLA"), _vector(df, "LMA"))

    rtoe = _mean_vectors(_vector(df, "RFM1"), _vector(df, "RFM2"), _vector(df, "RFM5"))
    ltoe = _mean_vectors(_vector(df, "LFM1"), _vector(df, "LFM2"), _vector(df, "LFM5"))

    required_vectors = [
        rasis,
        lasis,
        rpsis,
        lpsis,
        rhip,
        lhip,
        rknee,
        lknee,
        rank,
        lank,
        rtoe,
        ltoe,
    ]
    if any(v is None for v in required_vectors):
        raise ValueError(f"Missing required markers in {trajectories_path.name}")

    pelvis_center = np.nanmean(np.stack([rasis, lasis, rpsis, lpsis], axis=0), axis=0)
    forward_idx, progression_sign = _infer_progression_axis(pelvis_center)
    vertical_idx = 2

    side_cfg = [
        ("left", lhip, lknee, lank, ltoe),
        ("right", rhip, rknee, rank, rtoe),
    ]

    cycle_signals: list[CycleSignal] = []
    metric_rows: list[dict] = []

    for side_name, hip_marker, knee_center, ankle_center, toe_marker in side_cfg:
        hip_deg, knee_deg, ankle_deg, ankle_z = _compute_side_signals(
            hip_marker=hip_marker,
            knee_center=knee_center,
            ankle_center=ankle_center,
            toe_marker=toe_marker,
            forward_idx=forward_idx,
            vertical_idx=vertical_idx,
            progression_sign=progression_sign,
        )

        valid = (
            np.isfinite(time)
            & np.isfinite(hip_deg)
            & np.isfinite(knee_deg)
            & np.isfinite(ankle_deg)
            & np.isfinite(ankle_z)
        )
        time_v = time[valid]
        hip_v = hip_deg[valid]
        knee_v = knee_deg[valid]
        ankle_v = ankle_deg[valid]
        ankle_z_v = ankle_z[valid]
        if time_v.size < 8:
            continue

        cycle_bounds = _pick_cycle_boundaries_from_knee(
            time=time_v,
            knee=knee_v,
            min_stride_s=min_stride_s,
            max_stride_s=max_stride_s,
            smooth_window_samples=smooth_window_samples,
        )

        for cycle_idx, (t0, t1) in enumerate(cycle_bounds, start=1):
            seg = (time_v >= t0) & (time_v <= t1)
            if int(np.sum(seg)) < 12:
                continue

            t_seg = time_v[seg]
            hip_norm = _normalize_cycle(t_seg, hip_v[seg], n_points=n_points)
            knee_norm = _normalize_cycle(t_seg, knee_v[seg], n_points=n_points)
            ankle_norm = _normalize_cycle(t_seg, ankle_v[seg], n_points=n_points)
            ankle_z_norm = _normalize_cycle(t_seg, ankle_z_v[seg], n_points=n_points)
            pct = np.linspace(0.0, 100.0, n_points)

            cycle_signals.append(
                CycleSignal(
                    subject=subject,
                    speed=speed,
                    run=run,
                    trial=trial,
                    side=side_name,
                    cycle_id=cycle_idx,
                    pct=pct,
                    hip=hip_norm,
                    knee=knee_norm,
                    ankle=ankle_norm,
                    ankle_z=ankle_z_norm,
                )
            )

            stride_time = float(t1 - t0)
            cadence_spm = 120.0 / stride_time if stride_time > 0 else np.nan
            metric_rows.append(
                {
                    "subject": subject,
                    "speed": speed,
                    "speed_kmh": speed_kmh,
                    "speed_mps_nominal": speed_kmh / 3.6,
                    "run": run,
                    "trial": trial,
                    "side": side_name,
                    "cycle_id": cycle_idx,
                    "start_s": t0,
                    "end_s": t1,
                    "stride_time_s": stride_time,
                    "cadence_spm": cadence_spm,
                }
            )

    return cycle_signals, metric_rows


def _cycles_to_long_df(cycles: list[CycleSignal]) -> pd.DataFrame:
    rows = []
    for cycle in cycles:
        for i, pct in enumerate(cycle.pct):
            rows.append(
                {
                    "subject": cycle.subject,
                    "speed": cycle.speed,
                    "run": cycle.run,
                    "trial": cycle.trial,
                    "side": cycle.side,
                    "cycle_id": cycle.cycle_id,
                    "pct": float(pct),
                    "hip_deg": float(cycle.hip[i]),
                    "knee_deg": float(cycle.knee[i]),
                    "ankle_deg": float(cycle.ankle[i]),
                    "ankle_z_mm": float(cycle.ankle_z[i]),
                }
            )
    return pd.DataFrame(rows)


def _plot_joint_cycles(
    cycles: list[CycleSignal],
    out_png: Path,
    side: str = "both",
) -> None:
    sides = ["left", "right"] if side == "both" else [side]
    joints = [("hip", "Hip [deg]"), ("knee", "Knee [deg]"), ("ankle", "Ankle [deg]")]
    colors = {"hip": "#1f77b4", "knee": "#2ca02c", "ankle": "#d62728"}

    n_cols = len(sides)
    fig_width = 11 if n_cols == 2 else 5.8
    fig, axes = plt.subplots(3, n_cols, figsize=(fig_width, 11), sharex=True, constrained_layout=True)
    fig.patch.set_facecolor("white")
    if n_cols == 1:
        axes = np.asarray(axes).reshape(3, 1)

    for col, side_name in enumerate(sides):
        side_cycles = [c for c in cycles if c.side == side_name]
        if not side_cycles:
            for row in range(3):
                ax = axes[row, col]
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
            continue

        pct = side_cycles[0].pct
        for row, (joint_key, ylab) in enumerate(joints):
            ax = axes[row, col]
            stack = np.array([getattr(c, joint_key) for c in side_cycles])
            n_curves = stack.shape[0]

            sample_n = min(180, n_curves)
            sample_idx = np.linspace(0, n_curves - 1, sample_n, dtype=int)
            for idx in sample_idx:
                ax.plot(pct, stack[idx], color="#C4C7CE", alpha=0.2, linewidth=0.8)

            mean = np.nanmean(stack, axis=0)
            std = np.nanstd(stack, axis=0)
            color = colors[joint_key]
            ax.fill_between(pct, mean - std, mean + std, color=color, alpha=0.22, linewidth=0.0)
            ax.plot(pct, mean, color=color, linewidth=2.2)
            ax.set_xlim(0.0, 100.0)
            ax.grid(alpha=0.22)
            ax.set_ylabel(ylab)
            if row == 0:
                ax.set_title(f"{side_name.capitalize()} (n={n_curves} cycles)")

    for col in range(n_cols):
        axes[2, col].set_xlabel("Gait cycle [%]")
    title_side = f" ({side})" if side != "both" else ""
    fig.suptitle(f"Lower limb kinematic: joint cycles (mean ± SD){title_side}", fontsize=15)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _plot_reference_style(cycles: list[CycleSignal], out_png: Path, side: str = "both") -> None:
    if side == "both":
        selected = cycles
    else:
        selected = [c for c in cycles if c.side == side]
    if not selected:
        raise ValueError(f"No cycles available for side='{side}'.")

    pct = selected[0].pct
    ankle = np.array([c.ankle for c in selected])
    knee = np.array([c.knee for c in selected])
    hip = np.array([c.hip for c in selected])

    mean_ankle = np.nanmean(ankle, axis=0)
    mean_knee = np.nanmean(knee, axis=0)
    mean_hip = np.nanmean(hip, axis=0)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True, constrained_layout=True)
    fig.patch.set_facecolor("#d3d3d3")
    for ax in axes:
        ax.set_facecolor("#d3d3d3")
        ax.grid(alpha=0.22)

    for curve in ankle:
        axes[0].plot(pct, curve, color="gray", alpha=0.14, linewidth=1.0)
    axes[0].plot(pct, mean_ankle, color="black", linewidth=2.2)
    axes[0].set_title("ANKLE", fontsize=16, pad=8)

    for curve in knee:
        axes[1].plot(pct, curve, color="gray", alpha=0.14, linewidth=1.0)
    axes[1].plot(pct, mean_knee, color="black", linewidth=2.2)
    axes[1].set_title("KNEE", fontsize=16, pad=8)

    for curve in hip:
        axes[2].plot(pct, curve, color="gray", alpha=0.14, linewidth=1.0)
    axes[2].plot(pct, mean_hip, color="black", linewidth=2.2)
    axes[2].set_title("HIP", fontsize=16, pad=8)

    axes[2].set_xlabel("PERCENT OF GAIT CYCLE (%)", fontsize=14)
    axes[2].set_xlim(0.0, 100.0)
    fig.suptitle(f"lower_limb_kinematic gait ({side}, n={len(selected)})", fontsize=17, y=1.01)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _plot_stride_time_by_speed(metrics_df: pd.DataFrame, out_png: Path) -> None:
    if metrics_df.empty:
        return

    order = sorted(metrics_df["speed"].dropna().unique(), key=lambda s: (len(s), s))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, constrained_layout=True)

    for i, side in enumerate(["left", "right"]):
        ax = axes[i]
        sub = metrics_df[metrics_df["side"] == side]
        data = [sub.loc[sub["speed"] == s, "stride_time_s"].dropna().to_numpy() for s in order]
        valid = [arr for arr in data if arr.size > 0]
        if not valid:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        ax.boxplot(data, tick_labels=order, patch_artist=True)
        ax.set_title(f"{side.capitalize()} stride time")
        ax.set_xlabel("Condition speed")
        if i == 0:
            ax.set_ylabel("Stride time [s]")
        ax.grid(alpha=0.22, axis="y")

    fig.suptitle("Lower limb kinematic: stride time by speed", fontsize=14)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _zscore(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x))
    if not np.isfinite(sd) or sd <= 1e-8:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sd


def _best_corr_shift(
    signal: np.ndarray,
    template: np.ndarray,
    max_shift_pts: int,
) -> tuple[float, int]:
    s = _zscore(signal)
    t = _zscore(template)
    norm_t = float(np.linalg.norm(t))
    if norm_t <= 1e-8:
        return 0.0, 0

    best_corr = -2.0
    best_shift = 0
    for shift in range(-max_shift_pts, max_shift_pts + 1):
        sr = np.roll(s, shift)
        den = float(np.linalg.norm(sr)) * norm_t
        if den <= 1e-8:
            corr = 0.0
        else:
            corr = float(np.dot(sr, t) / den)
        if corr > best_corr:
            best_corr = corr
            best_shift = int(shift)
    return best_corr, best_shift


def _reference_templates(n_points: int) -> dict[str, np.ndarray]:
    xp = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
    yp = np.linspace(0.0, 100.0, n_points)

    # Approximate normative sagittal curves (shape only).
    hip = np.array([30, 27, 18, 5, -8, -18, -10, 2, 15, 24, 28], dtype=float)
    knee = np.array([5, 15, 10, 3, 0, 10, 25, 48, 38, 15, 5], dtype=float)
    ankle = np.array([0, -5, -1, 2, 5, 7, 0, -22, -10, 0, 2], dtype=float)

    return {
        "hip": np.interp(yp, xp, hip),
        "knee": np.interp(yp, xp, knee),
        "ankle": np.interp(yp, xp, ankle),
    }


def _filter_cycles_by_reference_shape(
    cycles: list[CycleSignal],
    min_corr_hip: float,
    min_corr_knee: float,
    min_corr_ankle: float,
    min_score: float,
    max_shift_pct: float,
) -> tuple[list[CycleSignal], pd.DataFrame]:
    if not cycles:
        cols = [
            "trial",
            "side",
            "cycle_id",
            "corr_hip",
            "corr_knee",
            "corr_ankle",
            "score",
            "shift_hip_pts",
            "shift_knee_pts",
            "shift_ankle_pts",
            "keep",
        ]
        return [], pd.DataFrame(columns=cols)

    n_points = int(cycles[0].pct.size)
    max_shift_pts = int(round(max(0.0, max_shift_pct) * (n_points - 1) / 100.0))
    templates = _reference_templates(n_points=n_points)

    rows: list[dict] = []
    kept: list[CycleSignal] = []

    for side in sorted({c.side for c in cycles}):
        side_cycles = [c for c in cycles if c.side == side]
        if not side_cycles:
            continue

        side_mean = {
            "hip": np.nanmean(np.stack([c.hip for c in side_cycles], axis=0), axis=0),
            "knee": np.nanmean(np.stack([c.knee for c in side_cycles], axis=0), axis=0),
            "ankle": np.nanmean(np.stack([c.ankle for c in side_cycles], axis=0), axis=0),
        }

        oriented: dict[str, np.ndarray] = {}
        for joint in ["hip", "knee", "ankle"]:
            r = templates[joint]
            cpos = float(np.corrcoef(_zscore(side_mean[joint]), _zscore(r))[0, 1])
            cneg = float(np.corrcoef(_zscore(side_mean[joint]), _zscore(-r))[0, 1])
            oriented[joint] = r if cpos >= cneg else -r

        for cyc in side_cycles:
            hip_s = _smooth_signal(cyc.hip, window=5)
            knee_s = _smooth_signal(cyc.knee, window=5)
            ankle_s = _smooth_signal(cyc.ankle, window=5)

            corr_hip, shift_hip = _best_corr_shift(
                signal=hip_s,
                template=oriented["hip"],
                max_shift_pts=max_shift_pts,
            )
            corr_knee, shift_knee = _best_corr_shift(
                signal=knee_s,
                template=oriented["knee"],
                max_shift_pts=max_shift_pts,
            )
            corr_ankle, shift_ankle = _best_corr_shift(
                signal=ankle_s,
                template=oriented["ankle"],
                max_shift_pts=max_shift_pts,
            )

            score = 0.25 * corr_hip + 0.5 * corr_knee + 0.25 * corr_ankle
            keep = (
                corr_hip >= min_corr_hip
                and corr_knee >= min_corr_knee
                and corr_ankle >= min_corr_ankle
                and score >= min_score
            )
            if keep:
                kept.append(cyc)

            rows.append(
                {
                    "trial": cyc.trial,
                    "side": cyc.side,
                    "cycle_id": int(cyc.cycle_id),
                    "corr_hip": float(corr_hip),
                    "corr_knee": float(corr_knee),
                    "corr_ankle": float(corr_ankle),
                    "score": float(score),
                    "shift_hip_pts": int(shift_hip),
                    "shift_knee_pts": int(shift_knee),
                    "shift_ankle_pts": int(shift_ankle),
                    "keep": bool(keep),
                }
            )

    return kept, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute gait-cycle analysis and plots for lower_limb_kinematic."
    )
    parser.add_argument(
        "--eurobench-root",
        default="data/lower_limb_kinematic/eurobench",
        help="Root containing converted Eurobench files.",
    )
    parser.add_argument(
        "--analysis-root",
        default="data/lower_limb_kinematic/analysis",
        help="Output folder for analysis CSV tables.",
    )
    parser.add_argument(
        "--plots-root",
        default="data/lower_limb_kinematic/plots",
        help="Output folder for plots.",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Optional single subject (e.g. 02 or Subject02).",
    )
    parser.add_argument(
        "--speed",
        default=None,
        help="Optional speed filter (e.g. V3).",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=101,
        help="Number of normalized points per cycle.",
    )
    parser.add_argument(
        "--min-stride-s",
        type=float,
        default=0.45,
        help="Minimum cycle duration in seconds.",
    )
    parser.add_argument(
        "--max-stride-s",
        type=float,
        default=2.2,
        help="Maximum cycle duration in seconds.",
    )
    parser.add_argument(
        "--smooth-window-samples",
        type=int,
        default=9,
        help="Smoothing window (samples) for knee-based cycle detection.",
    )
    parser.add_argument(
        "--joint-cycles-side",
        choices=["left", "right", "both"],
        default="both",
        help="Side selection for mean±SD joint plot.",
    )
    parser.add_argument(
        "--reference-style-side",
        choices=["left", "right", "both"],
        default="both",
        help="Side selection for reference-style plot.",
    )
    parser.add_argument(
        "--no-reference-shape-filter",
        action="store_false",
        dest="apply_reference_shape_filter",
        help="Disable shape-based cycle filtering against a normative gait template.",
    )
    parser.set_defaults(apply_reference_shape_filter=True)
    parser.add_argument(
        "--ref-corr-min-hip",
        type=float,
        default=0.25,
        help="Minimum hip template correlation to keep a cycle.",
    )
    parser.add_argument(
        "--ref-corr-min-knee",
        type=float,
        default=0.55,
        help="Minimum knee template correlation to keep a cycle.",
    )
    parser.add_argument(
        "--ref-corr-min-ankle",
        type=float,
        default=0.20,
        help="Minimum ankle template correlation to keep a cycle.",
    )
    parser.add_argument(
        "--ref-score-min",
        type=float,
        default=0.45,
        help="Minimum weighted morphology score to keep a cycle.",
    )
    parser.add_argument(
        "--ref-max-shift-pct",
        type=float,
        default=12.0,
        help="Max circular phase-shift [%% gait cycle] allowed while scoring correlation.",
    )
    args = parser.parse_args()

    eurobench_root = Path(args.eurobench_root)
    analysis_root = Path(args.analysis_root)
    plots_root = Path(args.plots_root)
    analysis_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    trajectories = sorted(eurobench_root.rglob("*_Trajectories.csv"))
    if args.subject:
        subject = _subject_tag(args.subject)
        trajectories = [p for p in trajectories if p.name.lower().startswith(subject.lower() + "_")]
    if args.speed:
        speed = _speed_tag(args.speed)
        trajectories = [p for p in trajectories if f"_{speed.upper()}_" in p.name.upper()]

    if not trajectories:
        raise FileNotFoundError(f"No *_Trajectories.csv found under {eurobench_root}")

    all_cycles: list[CycleSignal] = []
    all_metrics: list[dict] = []
    trial_log_rows: list[dict] = []

    for traj in trajectories:
        try:
            cycles, metrics = _analyze_trial(
                trajectories_path=traj,
                n_points=args.n_points,
                min_stride_s=args.min_stride_s,
                max_stride_s=args.max_stride_s,
                smooth_window_samples=args.smooth_window_samples,
            )
            all_cycles.extend(cycles)
            all_metrics.extend(metrics)
            trial_log_rows.append(
                {
                    "trial": traj.name,
                    "status": "ok",
                    "n_cycles": len(cycles),
                    "n_metric_rows": len(metrics),
                    "error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            trial_log_rows.append(
                {
                    "trial": traj.name,
                    "status": "error",
                    "n_cycles": 0,
                    "n_metric_rows": 0,
                    "error": str(exc),
                }
            )

    filter_log_csv = analysis_root / "reference_shape_filter_log.csv"
    if args.apply_reference_shape_filter:
        cycles_before = len(all_cycles)
        all_cycles, filter_df = _filter_cycles_by_reference_shape(
            cycles=all_cycles,
            min_corr_hip=args.ref_corr_min_hip,
            min_corr_knee=args.ref_corr_min_knee,
            min_corr_ankle=args.ref_corr_min_ankle,
            min_score=args.ref_score_min,
            max_shift_pct=args.ref_max_shift_pct,
        )
        keep_keys = {(c.trial, c.side, int(c.cycle_id)) for c in all_cycles}
        all_metrics = [
            m
            for m in all_metrics
            if (m.get("trial"), m.get("side"), int(m.get("cycle_id", -1))) in keep_keys
        ]
        filter_df.to_csv(filter_log_csv, index=False)
        print(f"reference_filter_enabled=True")
        print(f"cycles_before_filter={cycles_before}")
        print(f"cycles_after_filter={len(all_cycles)}")
        print(f"reference_filter_log={filter_log_csv}")
    else:
        cols = [
            "trial",
            "side",
            "cycle_id",
            "corr_hip",
            "corr_knee",
            "corr_ankle",
            "score",
            "shift_hip_pts",
            "shift_knee_pts",
            "shift_ankle_pts",
            "keep",
        ]
        pd.DataFrame(columns=cols).to_csv(filter_log_csv, index=False)
        print(f"reference_filter_enabled=False")
        print(f"reference_filter_log={filter_log_csv}")

    log_df = pd.DataFrame(trial_log_rows)
    metrics_df = pd.DataFrame(all_metrics)
    cycles_long_df = _cycles_to_long_df(all_cycles)

    log_csv = analysis_root / "trial_analysis_log.csv"
    metrics_csv = analysis_root / "cycle_metrics.csv"
    cycles_long_csv = analysis_root / "cycles_long.csv"
    summary_csv = analysis_root / "subject_speed_summary.csv"

    log_df.to_csv(log_csv, index=False)
    metrics_df.to_csv(metrics_csv, index=False)
    cycles_long_df.to_csv(cycles_long_csv, index=False)

    if metrics_df.empty:
        raise RuntimeError("No valid gait cycles were extracted from the dataset.")

    summary = (
        metrics_df.groupby(["subject", "speed", "side"], dropna=False)[
            ["speed_kmh", "speed_mps_nominal", "stride_time_s", "cadence_spm"]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = ["_".join(c).strip("_") for c in summary.columns.to_flat_index()]
    summary.to_csv(summary_csv, index=False)

    if args.joint_cycles_side == "both":
        joints_png = plots_root / "joint_cycles_mean_sd.png"
    else:
        joints_png = plots_root / f"joint_cycles_mean_sd_{args.joint_cycles_side}.png"
    reference_png = plots_root / f"joint_cycles_reference_style_{args.reference_style_side}.png"
    stride_speed_png = plots_root / "stride_time_by_speed.png"

    _plot_joint_cycles(all_cycles, joints_png, side=args.joint_cycles_side)
    _plot_reference_style(all_cycles, reference_png, side=args.reference_style_side)
    _plot_stride_time_by_speed(metrics_df, stride_speed_png)

    n_ok = int((log_df["status"] == "ok").sum()) if not log_df.empty else 0
    print(f"trials_total={len(trajectories)}")
    print(f"trials_ok={n_ok}")
    print(f"cycles={len(all_cycles)}")
    print(f"log={log_csv}")
    print(f"metrics={metrics_csv}")
    print(f"cycles_long={cycles_long_csv}")
    print(f"summary={summary_csv}")
    print(f"plot_joint={joints_png}")
    print(f"plot_reference_style={reference_png}")
    print(f"plot_stride_time_speed={stride_speed_png}")


if __name__ == "__main__":
    main()
