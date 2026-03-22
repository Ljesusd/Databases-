import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


TRIAL_RE = re.compile(r"^Subject(?P<subject>\d+)_(?P<run>\d+)_Trajectories\.csv$", re.IGNORECASE)


@dataclass
class CycleSignal:
    subject: str
    run: str
    trial: str
    side: str
    cycle_id: int
    pct: np.ndarray
    hip: np.ndarray
    knee: np.ndarray
    ankle: np.ndarray
    heel_z: np.ndarray


def _as_float_list(value: object) -> list[float]:
    if value is None:
        return []
    if isinstance(value, list):
        out = [float(v) for v in value]
    else:
        out = [float(value)]
    return sorted(out)


def _vector(df: pd.DataFrame, base: str) -> np.ndarray | None:
    cols = [f"{base}_x", f"{base}_y", f"{base}_z"]
    if all(c in df.columns for c in cols):
        return df[cols].to_numpy(dtype=float)
    return None


def _mean_vector(df: pd.DataFrame, base_a: str, base_b: str) -> np.ndarray | None:
    a = _vector(df, base_a)
    b = _vector(df, base_b)
    if a is not None and b is not None:
        stacked = np.stack([a, b], axis=0)
        return np.nanmean(stacked, axis=0)
    return a if a is not None else b


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


def _pair_cycles(
    heel_strikes: list[float],
    toe_offs: list[float],
    min_stride_s: float,
    max_stride_s: float,
) -> list[tuple[float, float, float | None]]:
    out: list[tuple[float, float, float | None]] = []
    if len(heel_strikes) < 2:
        return out
    for i in range(len(heel_strikes) - 1):
        hs1 = heel_strikes[i]
        hs2 = heel_strikes[i + 1]
        stride_time = hs2 - hs1
        if stride_time < min_stride_s or stride_time > max_stride_s:
            continue
        to_inside = [t for t in toe_offs if hs1 < t < hs2]
        toe = to_inside[0] if to_inside else None
        out.append((hs1, hs2, toe))
    return out


def _interp_in_range(time: np.ndarray, values: np.ndarray, t: float) -> float | None:
    if time.size == 0 or values.size == 0:
        return None
    if t < time[0] or t > time[-1]:
        return None
    return float(np.interp(t, time, values))


def _subject_tag(value: str) -> str:
    m = re.fullmatch(r"Subject(?P<id>\d+)", value, flags=re.IGNORECASE)
    if m:
        return f"Subject{m.group('id').zfill(2)}"
    if re.fullmatch(r"\d+", value):
        return f"Subject{value.zfill(2)}"
    raise ValueError("Subject must look like '01' or 'Subject01'.")


def _align_events_to_trial_time(
    time: np.ndarray,
    events: dict[str, list[float]],
) -> tuple[dict[str, list[float]], float]:
    finite_time = time[np.isfinite(time)]
    if finite_time.size == 0:
        return events, 0.0

    t_min = float(finite_time.min())
    t_max = float(finite_time.max())
    all_events = [v for vals in events.values() for v in vals if np.isfinite(v)]
    if not all_events:
        return events, 0.0

    def in_range_ratio(values: list[float]) -> float:
        if not values:
            return 0.0
        count = sum(1 for x in values if t_min <= x <= t_max)
        return float(count) / float(len(values))

    orig_ratio = in_range_ratio(all_events)
    offset = min(all_events) - t_min
    shifted = {k: [float(v - offset) for v in vals] for k, vals in events.items()}
    shifted_values = [v for vals in shifted.values() for v in vals]
    shifted_ratio = in_range_ratio(shifted_values)

    if shifted_ratio > orig_ratio + 0.2:
        return shifted, float(offset)
    return events, 0.0


def _compute_side_signals(
    hip_marker: np.ndarray,
    knee_center: np.ndarray,
    ankle_center: np.ndarray,
    toe_marker: np.ndarray,
    heel_marker: np.ndarray,
    forward_idx: int,
    vertical_idx: int,
    progression_sign: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    thigh = knee_center - hip_marker
    shank = ankle_center - knee_center
    foot = toe_marker - heel_marker

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

    heel_vertical = heel_marker[:, vertical_idx]
    return hip_deg, knee_deg, ankle_deg, heel_vertical


def _analyze_trial(
    trajectories_path: Path,
    events_path: Path,
    n_points: int,
    min_stride_s: float,
    max_stride_s: float,
) -> tuple[list[CycleSignal], list[dict]]:
    df = pd.read_csv(trajectories_path)
    if "time" not in df.columns:
        raise ValueError(f"Missing 'time' in {trajectories_path.name}")
    events = yaml.safe_load(events_path.read_text()) or {}

    match = TRIAL_RE.match(trajectories_path.name)
    if match is None:
        raise ValueError(f"Unexpected filename format: {trajectories_path.name}")

    subject = f"Subject{match.group('subject').zfill(2)}"
    run = match.group("run").zfill(2)
    trial = trajectories_path.name.replace("_Trajectories.csv", "")
    time = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)

    lasi = _vector(df, "LASI")
    rasi = _vector(df, "RASI")
    lpsi = _vector(df, "LPSI")
    rpsi = _vector(df, "RPSI")

    l_hip = _mean_vector(df, "LASI", "LPSI")
    r_hip = _mean_vector(df, "RASI", "RPSI")
    l_knee = _mean_vector(df, "LKNE", "LKNM")
    r_knee = _mean_vector(df, "RKNE", "RKNM")
    l_ankle = _mean_vector(df, "LANK", "LMED")
    r_ankle = _mean_vector(df, "RANK", "RMED")
    l_toe = _vector(df, "LTOE")
    r_toe = _vector(df, "RTOE")
    l_heel = _vector(df, "LHEE")
    r_heel = _vector(df, "RHEE")

    required_vectors = [
        lasi,
        rasi,
        lpsi,
        rpsi,
        l_hip,
        r_hip,
        l_knee,
        r_knee,
        l_ankle,
        r_ankle,
        l_toe,
        r_toe,
        l_heel,
        r_heel,
    ]
    if any(v is None for v in required_vectors):
        raise ValueError(f"Missing required markers in {trajectories_path.name}")

    pelvis_center = np.nanmean(np.stack([lasi, rasi, lpsi, rpsi], axis=0), axis=0)
    forward_idx, progression_sign = _infer_progression_axis(pelvis_center)
    vertical_idx = 2
    pelvis_forward = progression_sign * pelvis_center[:, forward_idx]
    valid_progression = np.isfinite(time) & np.isfinite(pelvis_forward)
    time_progression = time[valid_progression]
    pelvis_progression = pelvis_forward[valid_progression]

    side_cfg = [
        (
            "left",
            "l",
            l_hip,
            l_knee,
            l_ankle,
            l_toe,
            l_heel,
            _as_float_list(events.get("l_heel_strike")),
            _as_float_list(events.get("l_toe_off")),
            _as_float_list(events.get("r_heel_strike")),
        ),
        (
            "right",
            "r",
            r_hip,
            r_knee,
            r_ankle,
            r_toe,
            r_heel,
            _as_float_list(events.get("r_heel_strike")),
            _as_float_list(events.get("r_toe_off")),
            _as_float_list(events.get("l_heel_strike")),
        ),
    ]

    event_lists = {
        "l_heel_strike": side_cfg[0][7],
        "l_toe_off": side_cfg[0][8],
        "r_heel_strike": side_cfg[1][7],
        "r_toe_off": side_cfg[1][8],
    }
    aligned_events, _ = _align_events_to_trial_time(time=time, events=event_lists)
    side_cfg = [
        (
            side_cfg[0][0],
            side_cfg[0][1],
            side_cfg[0][2],
            side_cfg[0][3],
            side_cfg[0][4],
            side_cfg[0][5],
            side_cfg[0][6],
            aligned_events["l_heel_strike"],
            aligned_events["l_toe_off"],
            aligned_events["r_heel_strike"],
        ),
        (
            side_cfg[1][0],
            side_cfg[1][1],
            side_cfg[1][2],
            side_cfg[1][3],
            side_cfg[1][4],
            side_cfg[1][5],
            side_cfg[1][6],
            aligned_events["r_heel_strike"],
            aligned_events["r_toe_off"],
            aligned_events["l_heel_strike"],
        ),
    ]

    cycle_signals: list[CycleSignal] = []
    metric_rows: list[dict] = []

    for side_name, side_key, hip_marker, knee_center, ankle_center, toe_marker, heel_marker, hs, to, hs_opp in side_cfg:
        hip_deg, knee_deg, ankle_deg, heel_z = _compute_side_signals(
            hip_marker=hip_marker,
            knee_center=knee_center,
            ankle_center=ankle_center,
            toe_marker=toe_marker,
            heel_marker=heel_marker,
            forward_idx=forward_idx,
            vertical_idx=vertical_idx,
            progression_sign=progression_sign,
        )
        paired_cycles = _pair_cycles(hs, to, min_stride_s=min_stride_s, max_stride_s=max_stride_s)

        valid = (
            np.isfinite(time)
            & np.isfinite(hip_deg)
            & np.isfinite(knee_deg)
            & np.isfinite(ankle_deg)
            & np.isfinite(heel_z)
        )
        time_v = time[valid]
        hip_v = hip_deg[valid]
        knee_v = knee_deg[valid]
        ankle_v = ankle_deg[valid]
        heel_v = heel_z[valid]
        if time_v.size < 2:
            continue

        for cycle_idx, (hs1, hs2, toe_off) in enumerate(paired_cycles, start=1):
            seg = (time_v >= hs1) & (time_v <= hs2)
            if int(np.sum(seg)) < 5:
                continue

            t_seg = time_v[seg]
            hip_norm = _normalize_cycle(t_seg, hip_v[seg], n_points=n_points)
            knee_norm = _normalize_cycle(t_seg, knee_v[seg], n_points=n_points)
            ankle_norm = _normalize_cycle(t_seg, ankle_v[seg], n_points=n_points)
            heel_norm = _normalize_cycle(t_seg, heel_v[seg], n_points=n_points)
            pct = np.linspace(0.0, 100.0, n_points)

            cycle_signals.append(
                CycleSignal(
                    subject=subject,
                    run=run,
                    trial=trial,
                    side=side_name,
                    cycle_id=cycle_idx,
                    pct=pct,
                    hip=hip_norm,
                    knee=knee_norm,
                    ankle=ankle_norm,
                    heel_z=heel_norm,
                )
            )

            stride_time = float(hs2 - hs1)
            cadence_spm = 120.0 / stride_time if stride_time > 0 else np.nan

            step_candidates = [x for x in hs_opp if hs1 < x < hs2]
            step_time = float(step_candidates[0] - hs1) if step_candidates else np.nan

            stance_time = np.nan
            swing_time = np.nan
            stance_pct = np.nan
            swing_pct = np.nan
            toe_off_pct = np.nan
            if toe_off is not None:
                stance_time = float(toe_off - hs1)
                swing_time = float(hs2 - toe_off)
                if stance_time >= 0.0 and swing_time >= 0.0:
                    stance_pct = 100.0 * stance_time / stride_time
                    swing_pct = 100.0 * swing_time / stride_time
                    toe_off_pct = 100.0 * (toe_off - hs1) / stride_time

            p1 = _interp_in_range(time_progression, pelvis_progression, hs1)
            p2 = _interp_in_range(time_progression, pelvis_progression, hs2)
            stride_length_m = np.nan
            speed_mps = np.nan
            if p1 is not None and p2 is not None:
                stride_length_m = abs(p2 - p1) / 1000.0
                speed_mps = stride_length_m / stride_time if stride_time > 0 else np.nan

            metric_rows.append(
                {
                    "subject": subject,
                    "run": run,
                    "trial": trial,
                    "side": side_name,
                    "cycle_id": cycle_idx,
                    "hs1_s": hs1,
                    "hs2_s": hs2,
                    "toe_off_s": np.nan if toe_off is None else float(toe_off),
                    "stride_time_s": stride_time,
                    "step_time_s": step_time,
                    "stance_time_s": stance_time,
                    "swing_time_s": swing_time,
                    "stance_pct": stance_pct,
                    "swing_pct": swing_pct,
                    "toe_off_pct": toe_off_pct,
                    "cadence_spm": cadence_spm,
                    "stride_length_m": stride_length_m,
                    "speed_mps": speed_mps,
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
                    "run": cycle.run,
                    "trial": cycle.trial,
                    "side": cycle.side,
                    "cycle_id": cycle.cycle_id,
                    "pct": float(pct),
                    "hip_deg": float(cycle.hip[i]),
                    "knee_deg": float(cycle.knee[i]),
                    "ankle_deg": float(cycle.ankle[i]),
                    "heel_z_mm": float(cycle.heel_z[i]),
                }
            )
    return pd.DataFrame(rows)


def _plot_joint_cycles(cycles: list[CycleSignal], out_png: Path) -> None:
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

            sample_n = min(150, n_curves)
            sample_idx = np.linspace(0, n_curves - 1, sample_n, dtype=int)
            for idx in sample_idx:
                ax.plot(pct, stack[idx], color="#C4C7CE", alpha=0.25, linewidth=0.8)

            mean = np.nanmean(stack, axis=0)
            std = np.nanstd(stack, axis=0)
            color = colors[joint_key]
            ax.fill_between(pct, mean - std, mean + std, color=color, alpha=0.25, linewidth=0.0)
            ax.plot(pct, mean, color=color, linewidth=2.2)
            ax.set_xlim(0.0, 100.0)
            ax.grid(alpha=0.22)
            ax.set_ylabel(ylab)
            if row == 0:
                ax.set_title(f"{side.capitalize()} (n={n_curves} cycles)")

    axes[2, 0].set_xlabel("Gait cycle [%]")
    axes[2, 1].set_xlabel("Gait cycle [%]")
    fig.suptitle("Gait Analysis Assessment: joint cycles (mean ± SD)", fontsize=15)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _plot_heel_vertical_cycles(cycles: list[CycleSignal], out_png: Path) -> None:
    sides = ["left", "right"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, constrained_layout=True)

    for col, side in enumerate(sides):
        ax = axes[col]
        side_cycles = [c for c in cycles if c.side == side]
        if not side_cycles:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        pct = side_cycles[0].pct
        stack = np.array([c.heel_z for c in side_cycles])
        n_curves = stack.shape[0]

        sample_n = min(150, n_curves)
        sample_idx = np.linspace(0, n_curves - 1, sample_n, dtype=int)
        for idx in sample_idx:
            ax.plot(pct, stack[idx], color="#C4C7CE", alpha=0.25, linewidth=0.8)

        mean = np.nanmean(stack, axis=0)
        std = np.nanstd(stack, axis=0)
        ax.fill_between(pct, mean - std, mean + std, color="#9467bd", alpha=0.25, linewidth=0.0)
        ax.plot(pct, mean, color="#9467bd", linewidth=2.2)
        ax.set_xlim(0.0, 100.0)
        ax.grid(alpha=0.22)
        ax.set_title(f"{side.capitalize()} heel vertical (n={n_curves})")
        ax.set_xlabel("Gait cycle [%]")
        ax.set_ylabel("Heel vertical [mm]")

    fig.suptitle("Gait Analysis Assessment: heel vertical cycles (mean ± SD)", fontsize=14)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _plot_spatiotemporal_boxplots(metrics_df: pd.DataFrame, out_png: Path) -> None:
    metric_info = [
        ("stride_time_s", "Stride time [s]"),
        ("step_time_s", "Step time [s]"),
        ("cadence_spm", "Cadence [steps/min]"),
        ("stance_pct", "Stance [%]"),
        ("swing_pct", "Swing [%]"),
        ("speed_mps", "Speed [m/s]"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    axes_flat = axes.flatten()

    for ax, (col, title) in zip(axes_flat, metric_info):
        left_vals = metrics_df.loc[metrics_df["side"] == "left", col].dropna().to_numpy()
        right_vals = metrics_df.loc[metrics_df["side"] == "right", col].dropna().to_numpy()
        if left_vals.size == 0 and right_vals.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            ax.set_axis_off()
            continue

        box = ax.boxplot(
            [left_vals, right_vals],
            tick_labels=["Left", "Right"],
            patch_artist=True,
            widths=0.6,
        )
        colors = ["#1f77b4", "#ff7f0e"]
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
        for med in box["medians"]:
            med.set_color("black")
            med.set_linewidth(1.3)
        ax.set_title(title)
        ax.grid(alpha=0.22, axis="y")

    fig.suptitle("Gait Analysis Assessment: spatiotemporal summary", fontsize=14)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _best_circular_shift(curve: np.ndarray, template: np.ndarray) -> tuple[int, float]:
    x = np.asarray(curve, dtype=float)
    y = np.asarray(template, dtype=float)
    if x.size != y.size or x.size < 3:
        return 0, float("nan")

    x_fill = np.where(np.isfinite(x), x, np.nanmedian(x))
    y_fill = np.where(np.isfinite(y), y, np.nanmedian(y))
    x0 = x_fill - np.mean(x_fill)
    y0 = y_fill - np.mean(y_fill)
    y_norm = float(np.linalg.norm(y0))
    if y_norm == 0.0:
        return 0, float("nan")

    max_shift = x.size // 2
    best_shift = 0
    best_corr = -np.inf
    for shift in range(-max_shift, max_shift + 1):
        x_shift = np.roll(x0, shift)
        denom = float(np.linalg.norm(x_shift) * y_norm)
        if denom == 0.0:
            continue
        corr = float(np.dot(x_shift, y0) / denom)
        if corr > best_corr:
            best_corr = corr
            best_shift = shift

    return best_shift, best_corr


def _filter_cycles_by_phase(
    cycles: list[CycleSignal],
    max_shift_pct: float,
    min_corr: float,
) -> tuple[list[CycleSignal], pd.DataFrame]:
    if not cycles:
        return [], pd.DataFrame(columns=["trial", "side", "cycle_id", "best_shift_samples", "best_shift_pct", "corr", "keep"])

    n_points = cycles[0].pct.size
    max_shift_samples = max(1, int(round((max_shift_pct / 100.0) * n_points)))
    template = np.nanmedian(np.array([c.knee for c in cycles]), axis=0)

    kept: list[CycleSignal] = []
    rows: list[dict] = []
    for cycle in cycles:
        shift, corr = _best_circular_shift(cycle.knee, template)
        keep = bool(np.isfinite(corr) and abs(shift) <= max_shift_samples and corr >= min_corr)
        rows.append(
            {
                "trial": cycle.trial,
                "side": cycle.side,
                "cycle_id": cycle.cycle_id,
                "best_shift_samples": int(shift),
                "best_shift_pct": 100.0 * float(shift) / float(n_points),
                "corr": float(corr) if np.isfinite(corr) else np.nan,
                "keep": int(keep),
            }
        )
        if keep:
            kept.append(cycle)

    phase_df = pd.DataFrame(rows)
    return kept, phase_df


def _filter_reference_cycles(
    cycles: list[CycleSignal],
    max_shift_pct: float,
    min_corr: float,
) -> tuple[list[CycleSignal], pd.DataFrame]:
    by_side: dict[str, list[CycleSignal]] = {
        "left": [c for c in cycles if c.side == "left"],
        "right": [c for c in cycles if c.side == "right"],
    }

    kept_all: list[CycleSignal] = []
    logs: list[pd.DataFrame] = []
    for side in ("left", "right"):
        side_cycles = by_side[side]
        if not side_cycles:
            continue
        kept_side, log_side = _filter_cycles_by_phase(
            cycles=side_cycles,
            max_shift_pct=max_shift_pct,
            min_corr=min_corr,
        )
        logs.append(log_side)
        kept_all.extend(kept_side)

    if logs:
        phase_df = pd.concat(logs, ignore_index=True)
    else:
        phase_df = pd.DataFrame(columns=["trial", "side", "cycle_id", "best_shift_samples", "best_shift_pct", "corr", "keep"])
    return kept_all, phase_df


def _plot_reference_style(
    cycles: list[CycleSignal],
    out_png: Path,
    side: str = "both",
    apply_phase_filter: bool = False,
    max_shift_pct: float = 12.0,
    min_corr: float = 0.25,
    phase_log_csv: Path | None = None,
) -> tuple[int, int]:
    if side == "both":
        selected = cycles
    else:
        selected = [c for c in cycles if c.side == side]
    if not selected:
        raise ValueError(f"No cycles available for side='{side}'.")

    n_before = len(selected)
    if apply_phase_filter:
        selected, phase_df = _filter_reference_cycles(
            cycles=selected,
            max_shift_pct=max_shift_pct,
            min_corr=min_corr,
        )
        if phase_log_csv is not None:
            phase_log_csv.parent.mkdir(parents=True, exist_ok=True)
            phase_df.to_csv(phase_log_csv, index=False)
        if not selected:
            raise RuntimeError(
                "Phase filter removed all cycles. Relax --reference-max-shift-pct "
                "or --reference-min-corr."
            )

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
        axes[0].plot(pct, curve, color="gray", alpha=0.16, linewidth=1.0)
    axes[0].plot(pct, mean_ankle, color="black", linewidth=2.2)
    axes[0].set_title("ANKLE", fontsize=16, pad=8)

    for curve in knee:
        axes[1].plot(pct, curve, color="gray", alpha=0.16, linewidth=1.0)
    axes[1].plot(pct, mean_knee, color="black", linewidth=2.2)
    axes[1].set_title("KNEE", fontsize=16, pad=8)

    for curve in hip:
        axes[2].plot(pct, curve, color="gray", alpha=0.16, linewidth=1.0)
    axes[2].plot(pct, mean_hip, color="black", linewidth=2.2)
    axes[2].set_title("HIP", fontsize=16, pad=8)

    axes[2].set_xlabel("PERCENT OF GAIT CYCLE (%)", fontsize=14)
    axes[2].set_xlim(0.0, 100.0)
    fig.suptitle(
        f"gait_analysis_assessment gait ({side}, n={len(selected)})",
        fontsize=17,
        y=1.01,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    return n_before, len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute gait analysis summaries and plots for gait_analysis_assessment."
    )
    parser.add_argument(
        "--eurobench-root",
        default="data/gait_analysis_assessment/eurobench",
        help="Root containing converted Eurobench files.",
    )
    parser.add_argument(
        "--analysis-root",
        default="data/gait_analysis_assessment/analysis",
        help="Output folder for analysis CSV tables.",
    )
    parser.add_argument(
        "--plots-root",
        default="data/gait_analysis_assessment/plots",
        help="Output folder for plots.",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Optional single subject (e.g. 01 or Subject01).",
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
        default=0.5,
        help="Minimum HS-HS cycle duration in seconds.",
    )
    parser.add_argument(
        "--max-stride-s",
        type=float,
        default=2.2,
        help="Maximum HS-HS cycle duration in seconds.",
    )
    parser.add_argument(
        "--reference-style-side",
        choices=["left", "right", "both"],
        default="both",
        help="Side selection for the reference-style joint plot.",
    )
    parser.add_argument(
        "--reference-phase-filter",
        action="store_true",
        help="Filter cycles with large phase offset before the reference-style plot.",
    )
    parser.add_argument(
        "--reference-max-shift-pct",
        type=float,
        default=12.0,
        help="Maximum allowed circular shift (%% of cycle) when phase filter is enabled.",
    )
    parser.add_argument(
        "--reference-min-corr",
        type=float,
        default=0.25,
        help="Minimum knee-template correlation when phase filter is enabled.",
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

    if not trajectories:
        raise FileNotFoundError(f"No *_Trajectories.csv found under {eurobench_root}")

    all_cycles: list[CycleSignal] = []
    all_metrics: list[dict] = []
    trial_log_rows: list[dict] = []

    for traj in trajectories:
        events = traj.with_name(traj.name.replace("_Trajectories.csv", "_point_gaitEvents.yaml"))
        if not events.exists():
            trial_log_rows.append(
                {
                    "trial": traj.name,
                    "status": "missing_events",
                    "error": f"Missing {events.name}",
                }
            )
            continue
        try:
            cycles, metrics = _analyze_trial(
                trajectories_path=traj,
                events_path=events,
                n_points=args.n_points,
                min_stride_s=args.min_stride_s,
                max_stride_s=args.max_stride_s,
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

    log_df = pd.DataFrame(trial_log_rows)
    metrics_df = pd.DataFrame(all_metrics)
    cycles_long_df = _cycles_to_long_df(all_cycles)

    log_csv = analysis_root / "trial_analysis_log.csv"
    metrics_csv = analysis_root / "cycle_metrics.csv"
    cycles_long_csv = analysis_root / "cycles_long.csv"
    subject_summary_csv = analysis_root / "subject_summary.csv"

    log_df.to_csv(log_csv, index=False)
    metrics_df.to_csv(metrics_csv, index=False)
    cycles_long_df.to_csv(cycles_long_csv, index=False)

    if metrics_df.empty:
        raise RuntimeError("No valid gait cycles were extracted from the dataset.")

    summary = (
        metrics_df.groupby(["subject", "side"], dropna=False)[
            [
                "stride_time_s",
                "step_time_s",
                "stance_pct",
                "swing_pct",
                "cadence_spm",
                "stride_length_m",
                "speed_mps",
            ]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = ["_".join(c).strip("_") for c in summary.columns.to_flat_index()]
    summary.to_csv(subject_summary_csv, index=False)

    joints_png = plots_root / "joint_cycles_mean_sd.png"
    heel_png = plots_root / "heel_vertical_cycles_mean_sd.png"
    spatiotemporal_png = plots_root / "spatiotemporal_boxplots.png"
    if args.reference_phase_filter:
        reference_style_png = plots_root / f"joint_cycles_reference_style_{args.reference_style_side}_phase_filtered.png"
    else:
        reference_style_png = plots_root / f"joint_cycles_reference_style_{args.reference_style_side}.png"
    reference_phase_log = analysis_root / f"reference_style_phase_filter_{args.reference_style_side}.csv"

    _plot_joint_cycles(all_cycles, joints_png)
    _plot_heel_vertical_cycles(all_cycles, heel_png)
    _plot_spatiotemporal_boxplots(metrics_df, spatiotemporal_png)
    ref_n_before, ref_n_after = _plot_reference_style(
        all_cycles,
        reference_style_png,
        side=args.reference_style_side,
        apply_phase_filter=args.reference_phase_filter,
        max_shift_pct=args.reference_max_shift_pct,
        min_corr=args.reference_min_corr,
        phase_log_csv=reference_phase_log if args.reference_phase_filter else None,
    )

    n_ok = int((log_df["status"] == "ok").sum()) if not log_df.empty else 0
    print(f"trials_total={len(trajectories)}")
    print(f"trials_ok={n_ok}")
    print(f"cycles={len(all_cycles)}")
    print(f"log={log_csv}")
    print(f"metrics={metrics_csv}")
    print(f"cycles_long={cycles_long_csv}")
    print(f"summary={subject_summary_csv}")
    print(f"plot_joint={joints_png}")
    print(f"plot_heel={heel_png}")
    print(f"plot_spatiotemporal={spatiotemporal_png}")
    print(f"plot_reference_style={reference_style_png}")
    print(f"reference_cycles_before={ref_n_before}")
    print(f"reference_cycles_after={ref_n_after}")
    if args.reference_phase_filter:
        print(f"reference_phase_log={reference_phase_log}")


if __name__ == "__main__":
    main()
