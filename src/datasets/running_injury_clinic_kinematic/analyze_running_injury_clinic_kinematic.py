import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


TRIAL_RE = re.compile(
    r"^Subject(?P<subject>\d+)_(?P<condition>WALK|RUN)_(?P<run>\d+)_Trajectories\.csv$",
    re.IGNORECASE,
)


@dataclass
class CycleSignal:
    subject: str
    condition: str
    run: str
    trial: str
    side: str
    cycle_id: int
    pct: np.ndarray
    hip: np.ndarray
    knee: np.ndarray
    ankle: np.ndarray


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


def _normalize_cycle(time_seg: np.ndarray, values: np.ndarray, n_points: int) -> np.ndarray:
    if time_seg.size < 2:
        raise ValueError("Not enough samples to normalize cycle.")
    dt = float(time_seg[-1] - time_seg[0])
    if dt <= 0.0:
        raise ValueError("Invalid cycle duration (<= 0).")
    t_norm = (time_seg - time_seg[0]) / dt
    t_target = np.linspace(0.0, 1.0, n_points)
    return np.interp(t_target, t_norm, values)


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


def _zscore(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x))
    if not np.isfinite(sd) or sd <= 1e-9:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sd


def _best_corr_shift(signal: np.ndarray, template: np.ndarray, max_shift_pts: int) -> tuple[float, int]:
    s = _zscore(signal)
    t = _zscore(template)
    norm_t = float(np.linalg.norm(t))
    if norm_t <= 1e-9:
        return 0.0, 0

    best_corr = -2.0
    best_shift = 0
    for shift in range(-max_shift_pts, max_shift_pts + 1):
        sr = np.roll(s, shift)
        den = float(np.linalg.norm(sr)) * norm_t
        corr = 0.0 if den <= 1e-9 else float(np.dot(sr, t) / den)
        if corr > best_corr:
            best_corr = corr
            best_shift = int(shift)
    return best_corr, best_shift


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

    thr = np.nanpercentile(k_smooth, 70.0)
    cand = cand[k_smooth[cand] <= thr]
    if cand.size < 2:
        return []

    min_sep = max(0.2, 0.5 * min_stride_s)
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


def _infer_ap_axis(
    pelvis_center: np.ndarray,
    left_toe: np.ndarray | None,
    right_toe: np.ndarray | None,
    left_foot: np.ndarray,
    right_foot: np.ndarray,
    left_shank: np.ndarray,
    right_shank: np.ndarray,
) -> tuple[int, float]:
    candidates = [pelvis_center, left_foot, right_foot, left_shank, right_shank]
    if left_toe is not None:
        candidates.append(left_toe)
    if right_toe is not None:
        candidates.append(right_toe)
    pool = np.vstack(candidates)
    var_x = float(np.nanvar(pool[:, 0]))
    var_y = float(np.nanvar(pool[:, 1]))
    ap_idx = 0 if var_x >= var_y else 1

    refs = []
    if left_toe is not None:
        refs.append(np.nanmean((left_toe - left_foot)[:, ap_idx]))
    if right_toe is not None:
        refs.append(np.nanmean((right_toe - right_foot)[:, ap_idx]))
    if refs:
        s = float(np.nanmean(refs))
        ap_sign = 1.0 if (np.isfinite(s) and s >= 0.0) else -1.0
    else:
        step = float(np.nanmedian(np.diff(pelvis_center[:, ap_idx])))
        ap_sign = 1.0 if (np.isfinite(step) and step >= 0.0) else -1.0

    return ap_idx, ap_sign


def _compute_side_signals(
    pelvis_center: np.ndarray,
    thigh_center: np.ndarray,
    shank_center: np.ndarray,
    foot_center: np.ndarray,
    toe_marker: np.ndarray | None,
    ap_idx: int,
    vertical_idx: int,
    ap_sign: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    thigh = thigh_center - pelvis_center
    shank = shank_center - thigh_center
    if toe_marker is not None:
        foot = toe_marker - foot_center
    else:
        # Fallback when toe marker is missing: use distal direction from shank to foot cluster.
        foot = foot_center - shank_center

    knee_deg = _angle_between_3d(thigh, shank)
    ankle_deg = _angle_between_3d(shank, foot) - 90.0

    thigh_2d = np.stack([ap_sign * thigh[:, ap_idx], thigh[:, vertical_idx]], axis=1)
    vertical_down = np.tile([0.0, -1.0], (thigh_2d.shape[0], 1))
    hip_signed = _angle_between_2d(vertical_down, thigh_2d)
    hip_unwrapped = np.degrees(np.unwrap(np.radians(hip_signed)))
    if hip_unwrapped.size and hip_unwrapped[0] > 90.0:
        hip_unwrapped = hip_unwrapped - 180.0
    elif hip_unwrapped.size and hip_unwrapped[0] < -90.0:
        hip_unwrapped = hip_unwrapped + 180.0
    hip_deg = -hip_unwrapped

    return hip_deg, knee_deg, ankle_deg


def _trial_info(path_csv: Path) -> dict:
    info_yaml = path_csv.with_name(path_csv.name.replace("_Trajectories.csv", "_info.yaml"))
    if not info_yaml.exists():
        return {}
    try:
        payload = yaml.safe_load(info_yaml.read_text(encoding="utf-8", errors="ignore")) or {}
    except Exception:  # noqa: BLE001
        return {}
    return payload if isinstance(payload, dict) else {}


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
    subject = f"Subject{m.group('subject')}"
    condition = m.group("condition").upper()
    run = m.group("run")
    trial = trajectories_path.name.replace("_Trajectories.csv", "")
    info = _trial_info(trajectories_path)

    time = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
    pelvis = _mean_vectors(_vector(df, "pelvis_1"), _vector(df, "pelvis_2"), _vector(df, "pelvis_3"), _vector(df, "pelvis_4"))
    lth = _mean_vectors(_vector(df, "L_thigh_1"), _vector(df, "L_thigh_2"), _vector(df, "L_thigh_3"), _vector(df, "L_thigh_4"))
    rth = _mean_vectors(_vector(df, "R_thigh_1"), _vector(df, "R_thigh_2"), _vector(df, "R_thigh_3"), _vector(df, "R_thigh_4"))
    lsh = _mean_vectors(_vector(df, "L_shank_1"), _vector(df, "L_shank_2"), _vector(df, "L_shank_3"), _vector(df, "L_shank_4"))
    rsh = _mean_vectors(_vector(df, "R_shank_1"), _vector(df, "R_shank_2"), _vector(df, "R_shank_3"), _vector(df, "R_shank_4"))
    lft = _mean_vectors(_vector(df, "L_foot_1"), _vector(df, "L_foot_2"), _vector(df, "L_foot_3"), _vector(df, "L_foot_4"))
    rft = _mean_vectors(_vector(df, "R_foot_1"), _vector(df, "R_foot_2"), _vector(df, "R_foot_3"), _vector(df, "R_foot_4"))
    ltoe = _vector(df, "L_toe")
    rtoe = _vector(df, "R_toe")

    required = [pelvis, lth, rth, lsh, rsh, lft, rft]
    if any(v is None for v in required):
        raise ValueError(f"Missing required cluster markers in {trajectories_path.name}")

    ap_idx, ap_sign = _infer_ap_axis(
        pelvis_center=pelvis,
        left_toe=ltoe,
        right_toe=rtoe,
        left_foot=lft,
        right_foot=rft,
        left_shank=lsh,
        right_shank=rsh,
    )
    vertical_idx = 2

    speed_mps = info.get("speed_mps", np.nan)
    try:
        speed_mps = float(speed_mps)
    except Exception:  # noqa: BLE001
        speed_mps = np.nan

    side_cfg = [
        ("left", lth, lsh, lft, ltoe),
        ("right", rth, rsh, rft, rtoe),
    ]
    cycle_signals: list[CycleSignal] = []
    metric_rows: list[dict] = []

    for side_name, th, sh, ft, toe in side_cfg:
        hip, knee, ankle = _compute_side_signals(
            pelvis_center=pelvis,
            thigh_center=th,
            shank_center=sh,
            foot_center=ft,
            toe_marker=toe,
            ap_idx=ap_idx,
            vertical_idx=vertical_idx,
            ap_sign=ap_sign,
        )

        valid = np.isfinite(time) & np.isfinite(hip) & np.isfinite(knee) & np.isfinite(ankle)
        time_v = time[valid]
        hip_v = hip[valid]
        knee_v = knee[valid]
        ankle_v = ankle[valid]
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
            pct = np.linspace(0.0, 100.0, n_points)

            hip_n = _normalize_cycle(t_seg, hip_v[seg], n_points=n_points)
            knee_n = _normalize_cycle(t_seg, knee_v[seg], n_points=n_points)
            ankle_n = _normalize_cycle(t_seg, ankle_v[seg], n_points=n_points)

            cycle_signals.append(
                CycleSignal(
                    subject=subject,
                    condition=condition,
                    run=run,
                    trial=trial,
                    side=side_name,
                    cycle_id=cycle_idx,
                    pct=pct,
                    hip=hip_n,
                    knee=knee_n,
                    ankle=ankle_n,
                )
            )

            stride_time = float(t1 - t0)
            cadence_spm = 120.0 / stride_time if stride_time > 0 else np.nan
            metric_rows.append(
                {
                    "subject": subject,
                    "condition": condition,
                    "run": run,
                    "trial": trial,
                    "side": side_name,
                    "cycle_id": cycle_idx,
                    "start_s": t0,
                    "end_s": t1,
                    "stride_time_s": stride_time,
                    "cadence_spm": cadence_spm,
                    "speed_mps": speed_mps,
                }
            )

    return cycle_signals, metric_rows


def _filter_cycles_by_population_shape(
    cycles: list[CycleSignal],
    max_shift_pct: float,
    min_corr_hip: float,
    min_corr_knee: float,
    min_corr_ankle: float,
    min_score: float,
    hip_sign_margin_deg: float,
) -> tuple[list[CycleSignal], pd.DataFrame]:
    cols = [
        "trial",
        "condition",
        "side",
        "cycle_id",
        "corr_hip",
        "corr_knee",
        "corr_ankle",
        "score",
        "shift_hip_pts",
        "shift_knee_pts",
        "shift_ankle_pts",
        "hip_mean_deg",
        "dominant_hip_sign",
        "hip_sign_ok",
        "keep",
    ]
    if not cycles:
        return [], pd.DataFrame(columns=cols)

    n_points = int(cycles[0].pct.size)
    max_shift_pts = int(round(max(0.0, max_shift_pct) * (n_points - 1) / 100.0))
    rows = []
    kept = []

    key_pairs = sorted({(c.condition, c.side) for c in cycles})
    for condition, side in key_pairs:
        curr = [c for c in cycles if c.condition == condition and c.side == side]
        if not curr:
            continue
        template_hip = np.nanmedian(np.stack([c.hip for c in curr], axis=0), axis=0)
        template_knee = np.nanmedian(np.stack([c.knee for c in curr], axis=0), axis=0)
        template_ankle = np.nanmedian(np.stack([c.ankle for c in curr], axis=0), axis=0)
        hip_means = np.array([float(np.nanmean(c.hip)) for c in curr], dtype=float)
        dominant_hip_sign = 1 if float(np.nanmedian(hip_means)) >= 0.0 else -1

        for cyc in curr:
            hip_s = _smooth_signal(cyc.hip, window=5)
            knee_s = _smooth_signal(cyc.knee, window=5)
            ankle_s = _smooth_signal(cyc.ankle, window=5)

            corr_hip, shift_hip = _best_corr_shift(hip_s, template_hip, max_shift_pts=max_shift_pts)
            corr_knee, shift_knee = _best_corr_shift(knee_s, template_knee, max_shift_pts=max_shift_pts)
            corr_ankle, shift_ankle = _best_corr_shift(ankle_s, template_ankle, max_shift_pts=max_shift_pts)

            score = 0.25 * corr_hip + 0.50 * corr_knee + 0.25 * corr_ankle
            hip_mean_deg = float(np.nanmean(cyc.hip))
            hip_sign_ok = (hip_mean_deg * float(dominant_hip_sign)) >= -abs(float(hip_sign_margin_deg))
            keep = (
                corr_hip >= min_corr_hip
                and corr_knee >= min_corr_knee
                and corr_ankle >= min_corr_ankle
                and score >= min_score
                and hip_sign_ok
            )
            if keep:
                kept.append(cyc)
            rows.append(
                {
                    "trial": cyc.trial,
                    "condition": cyc.condition,
                    "side": cyc.side,
                    "cycle_id": int(cyc.cycle_id),
                    "corr_hip": float(corr_hip),
                    "corr_knee": float(corr_knee),
                    "corr_ankle": float(corr_ankle),
                    "score": float(score),
                    "shift_hip_pts": int(shift_hip),
                    "shift_knee_pts": int(shift_knee),
                    "shift_ankle_pts": int(shift_ankle),
                    "hip_mean_deg": hip_mean_deg,
                    "dominant_hip_sign": int(dominant_hip_sign),
                    "hip_sign_ok": bool(hip_sign_ok),
                    "keep": bool(keep),
                }
            )

    return kept, pd.DataFrame(rows, columns=cols)


def _cycles_to_long_df(cycles: list[CycleSignal]) -> pd.DataFrame:
    rows = []
    for c in cycles:
        for i, pct in enumerate(c.pct):
            rows.append(
                {
                    "subject": c.subject,
                    "condition": c.condition,
                    "run": c.run,
                    "trial": c.trial,
                    "side": c.side,
                    "cycle_id": c.cycle_id,
                    "pct": float(pct),
                    "hip_deg": float(c.hip[i]),
                    "knee_deg": float(c.knee[i]),
                    "ankle_deg": float(c.ankle[i]),
                }
            )
    return pd.DataFrame(rows)


def _plot_joint_cycles_mean_sd(cycles: list[CycleSignal], out_png: Path) -> None:
    sides = ["left", "right"]
    joints = [("hip", "Hip [deg]"), ("knee", "Knee [deg]"), ("ankle", "Ankle [deg]")]
    colors = {"hip": "#1f77b4", "knee": "#2ca02c", "ankle": "#d62728"}

    fig, axes = plt.subplots(3, 2, figsize=(11, 11), sharex=True, constrained_layout=True)
    fig.patch.set_facecolor("white")

    for col, side in enumerate(sides):
        side_cycles = [c for c in cycles if c.side == side]
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
            sample_n = min(220, n_curves)
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
                ax.set_title(f"{side.capitalize()} (n={n_curves} cycles)")

    axes[2, 0].set_xlabel("Gait cycle [%]")
    axes[2, 1].set_xlabel("Gait cycle [%]")
    fig.suptitle("Running Injury Clinic: joint cycles (mean ± SD)", fontsize=15)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _plot_reference_style(cycles: list[CycleSignal], out_png: Path, side: str = "both") -> None:
    if side == "both":
        selected = cycles
    else:
        selected = [c for c in cycles if c.side == side]
    if not selected:
        return

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
    fig.suptitle(f"running_injury_clinic gait ({side}, n={len(selected)})", fontsize=17, y=1.01)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _plot_stride_time_vs_speed(metrics_df: pd.DataFrame, out_png: Path) -> None:
    if metrics_df.empty or "speed_mps" not in metrics_df.columns:
        return
    df = metrics_df[pd.to_numeric(metrics_df["speed_mps"], errors="coerce").notna()].copy()
    if df.empty:
        return
    df["speed_mps"] = pd.to_numeric(df["speed_mps"], errors="coerce")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, constrained_layout=True)
    for i, side in enumerate(["left", "right"]):
        ax = axes[i]
        sub = df[df["side"] == side]
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        ax.scatter(
            sub["speed_mps"].to_numpy(),
            sub["stride_time_s"].to_numpy(),
            s=16,
            alpha=0.25,
            c="#1f77b4",
            edgecolors="none",
        )
        ax.set_title(f"{side.capitalize()} stride time vs speed")
        ax.set_xlabel("Treadmill speed [m/s]")
        if i == 0:
            ax.set_ylabel("Stride time [s]")
        ax.grid(alpha=0.22)
    fig.suptitle("Running Injury Clinic: stride time vs speed", fontsize=14)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute gait-cycle analysis and plots for running_injury_clinic_kinematic."
    )
    parser.add_argument(
        "--eurobench-root",
        default="data/running_injury_clinic_kinematic/healthy/eurobench",
        help="Root containing Eurobench trajectories and info files.",
    )
    parser.add_argument(
        "--analysis-root",
        default="data/running_injury_clinic_kinematic/healthy/analysis",
        help="Output folder for analysis CSV tables.",
    )
    parser.add_argument(
        "--plots-root",
        default="data/running_injury_clinic_kinematic/healthy/plots",
        help="Output folder for plots.",
    )
    parser.add_argument(
        "--condition",
        choices=["WALK", "RUN", "BOTH"],
        default="WALK",
        help="Condition to analyze.",
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
        default=0.55,
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
        "--no-shape-filter",
        action="store_false",
        dest="apply_shape_filter",
        help="Disable population-shape filter.",
    )
    parser.set_defaults(apply_shape_filter=True)
    parser.add_argument(
        "--shape-min-corr-hip",
        type=float,
        default=0.30,
        help="Minimum hip correlation against population template.",
    )
    parser.add_argument(
        "--shape-min-corr-knee",
        type=float,
        default=0.45,
        help="Minimum knee correlation against population template.",
    )
    parser.add_argument(
        "--shape-min-corr-ankle",
        type=float,
        default=0.20,
        help="Minimum ankle correlation against population template.",
    )
    parser.add_argument(
        "--shape-min-score",
        type=float,
        default=0.42,
        help="Minimum weighted morphology score.",
    )
    parser.add_argument(
        "--shape-max-shift-pct",
        type=float,
        default=12.0,
        help="Max circular phase-shift [%% gait cycle] in correlation scoring.",
    )
    parser.add_argument(
        "--shape-hip-sign-margin-deg",
        type=float,
        default=10.0,
        help="Allowed margin [deg] around the dominant hip-sign cluster.",
    )
    parser.add_argument(
        "--reference-style-side",
        choices=["left", "right", "both"],
        default="both",
        help="Side selection for reference-style plot.",
    )
    args = parser.parse_args()

    eurobench_root = Path(args.eurobench_root)
    analysis_root = Path(args.analysis_root)
    plots_root = Path(args.plots_root)
    analysis_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)

    trajectories = sorted(eurobench_root.rglob("*_Trajectories.csv"))
    if args.condition != "BOTH":
        trajectories = [p for p in trajectories if f"_{args.condition}_" in p.name.upper()]
    if not trajectories:
        raise FileNotFoundError(f"No trajectories found for condition={args.condition}")

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

    filter_log_csv = analysis_root / f"reference_shape_filter_log_{args.condition.lower()}.csv"
    if args.apply_shape_filter:
        cycles_before = len(all_cycles)
        all_cycles, filter_df = _filter_cycles_by_population_shape(
            cycles=all_cycles,
            max_shift_pct=args.shape_max_shift_pct,
            min_corr_hip=args.shape_min_corr_hip,
            min_corr_knee=args.shape_min_corr_knee,
            min_corr_ankle=args.shape_min_corr_ankle,
            min_score=args.shape_min_score,
            hip_sign_margin_deg=args.shape_hip_sign_margin_deg,
        )
        keep_keys = {(c.trial, c.side, int(c.cycle_id)) for c in all_cycles}
        all_metrics = [
            m
            for m in all_metrics
            if (m.get("trial"), m.get("side"), int(m.get("cycle_id", -1))) in keep_keys
        ]
        filter_df.to_csv(filter_log_csv, index=False)
        print("shape_filter_enabled=True")
        print(f"cycles_before_filter={cycles_before}")
        print(f"cycles_after_filter={len(all_cycles)}")
        print(f"shape_filter_log={filter_log_csv}")
    else:
        pd.DataFrame(columns=["trial", "condition", "side", "cycle_id", "keep"]).to_csv(
            filter_log_csv, index=False
        )
        print("shape_filter_enabled=False")
        print(f"shape_filter_log={filter_log_csv}")

    log_df = pd.DataFrame(trial_log_rows)
    metrics_df = pd.DataFrame(all_metrics)
    cycles_long_df = _cycles_to_long_df(all_cycles)

    if metrics_df.empty:
        raise RuntimeError("No valid gait cycles were extracted from the selected condition.")

    log_csv = analysis_root / f"trial_analysis_log_{args.condition.lower()}.csv"
    metrics_csv = analysis_root / f"cycle_metrics_{args.condition.lower()}.csv"
    cycles_long_csv = analysis_root / f"cycles_long_{args.condition.lower()}.csv"
    summary_csv = analysis_root / f"subject_summary_{args.condition.lower()}.csv"

    log_df.to_csv(log_csv, index=False)
    metrics_df.to_csv(metrics_csv, index=False)
    cycles_long_df.to_csv(cycles_long_csv, index=False)

    summary = (
        metrics_df.groupby(["subject", "condition", "side"], dropna=False)[
            ["stride_time_s", "cadence_spm", "speed_mps"]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = ["_".join(c).strip("_") for c in summary.columns.to_flat_index()]
    summary.to_csv(summary_csv, index=False)

    cond_tag = args.condition.lower()
    joints_png = plots_root / f"{cond_tag}_joint_cycles_mean_sd.png"
    reference_png = plots_root / f"{cond_tag}_joint_cycles_reference_style_{args.reference_style_side}.png"
    stride_speed_png = plots_root / f"{cond_tag}_stride_time_vs_speed.png"

    _plot_joint_cycles_mean_sd(all_cycles, joints_png)
    _plot_reference_style(all_cycles, reference_png, side=args.reference_style_side)
    _plot_stride_time_vs_speed(metrics_df, stride_speed_png)

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
