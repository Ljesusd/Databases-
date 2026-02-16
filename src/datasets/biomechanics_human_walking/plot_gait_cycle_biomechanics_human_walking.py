import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


TRIAL_PATTERN = re.compile(
    r"^(?P<subject>\d+)_(?P<condition>C\d+)_(?P<run>\d+)_Trajectories\.csv$"
)


@dataclass
class TrialCycle:
    subject: str
    condition: str
    run: str
    pct: np.ndarray
    ankle: np.ndarray
    knee: np.ndarray
    hip: np.ndarray


def _vector(df: pd.DataFrame, base: str) -> np.ndarray | None:
    cols = [f"{base}_x", f"{base}_y", f"{base}_z"]
    if all(c in df.columns for c in cols):
        return df[cols].to_numpy(dtype=float)
    return None


def _mean_vector(df: pd.DataFrame, base_a: str, base_b: str) -> np.ndarray | None:
    a = _vector(df, base_a)
    b = _vector(df, base_b)
    if a is not None and b is not None:
        return 0.5 * (a + b)
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
    dt = time_seg[-1] - time_seg[0]
    if dt <= 0:
        raise ValueError("Invalid cycle duration (<= 0).")
    t_norm = (time_seg - time_seg[0]) / dt
    t_target = np.linspace(0.0, 1.0, n_points)
    return np.interp(t_target, t_norm, values)


def _extract_cycle_from_trial(
    trajectories_csv: Path,
    events_yaml: Path,
    n_points: int,
    forward_idx: int,
    vertical_idx: int,
) -> TrialCycle:
    df = pd.read_csv(trajectories_csv)
    if "time" not in df.columns:
        raise ValueError(f"Missing 'time' in {trajectories_csv.name}")

    events = yaml.safe_load(events_yaml.read_text()) or {}
    hs1 = events.get("r_heel_strike1")
    hs2 = events.get("r_heel_strike2")
    if hs1 is None or hs2 is None:
        raise ValueError(f"Missing right heel strike pair in {events_yaml.name}")

    hs1_val = float(hs1[0] if isinstance(hs1, list) else hs1)
    hs2_val = float(hs2[0] if isinstance(hs2, list) else hs2)
    if hs2_val <= hs1_val:
        raise ValueError(f"Invalid heel strike order in {events_yaml.name}")

    hip = _vector(df, "RGTR")
    knee = _mean_vector(df, "RLEP", "RMEP")
    ankle = _mean_vector(df, "RLML", "RMML")
    toe = _vector(df, "R5TH")
    heel = _vector(df, "RCAL")

    if any(v is None for v in [hip, knee, ankle, toe, heel]):
        raise ValueError(f"Missing required right-leg markers in {trajectories_csv.name}")

    lasi = _vector(df, "LASI")
    rasi = _vector(df, "RASI")
    sac = _vector(df, "SACR")
    if any(v is None for v in [lasi, rasi, sac]):
        raise ValueError(f"Missing pelvis markers (LASI/RASI/SACR) in {trajectories_csv.name}")

    thigh = knee - hip
    shank = ankle - knee
    foot = toe - heel

    knee_deg = _angle_between_3d(thigh, shank)
    ankle_deg = _angle_between_3d(shank, foot) - 90.0

    pelvis_center = (lasi + rasi + sac) / 3.0
    time = pd.to_numeric(df["time"], errors="coerce").to_numpy()
    progression_mask = (time >= hs1_val) & (time <= hs2_val)
    progression_sign = 1.0
    if np.any(progression_mask):
        step = np.nanmedian(np.diff(pelvis_center[progression_mask, forward_idx]))
        progression_sign = 1.0 if (np.isfinite(step) and step >= 0.0) else -1.0

    thigh_2d = np.stack([progression_sign * thigh[:, forward_idx], thigh[:, vertical_idx]], axis=1)
    vertical_down = np.tile([0.0, -1.0], (thigh_2d.shape[0], 1))
    hip_signed = _angle_between_2d(vertical_down, thigh_2d)
    hip_unwrapped = np.degrees(np.unwrap(np.radians(hip_signed)))
    if hip_unwrapped[0] > 90.0:
        hip_unwrapped = hip_unwrapped - 180.0
    elif hip_unwrapped[0] < -90.0:
        hip_unwrapped = hip_unwrapped + 180.0
    hip_deg = -hip_unwrapped

    valid = np.isfinite(time) & np.isfinite(knee_deg) & np.isfinite(ankle_deg) & np.isfinite(hip_deg)
    time = time[valid]
    knee_deg = knee_deg[valid]
    ankle_deg = ankle_deg[valid]
    hip_deg = hip_deg[valid]

    mask = (time >= hs1_val) & (time <= hs2_val)
    time_seg = time[mask]
    if time_seg.size < 2:
        raise ValueError(f"Not enough samples in gait cycle for {trajectories_csv.name}")

    knee_norm = _normalize_cycle(time_seg, knee_deg[mask], n_points)
    ankle_norm = _normalize_cycle(time_seg, ankle_deg[mask], n_points)
    hip_norm = _normalize_cycle(time_seg, hip_deg[mask], n_points)

    pct = np.linspace(0.0, 100.0, n_points)
    match = TRIAL_PATTERN.match(trajectories_csv.name)
    if match is None:
        raise ValueError(f"Unexpected filename format: {trajectories_csv.name}")

    return TrialCycle(
        subject=match.group("subject"),
        condition=match.group("condition"),
        run=match.group("run"),
        pct=pct,
        ankle=ankle_norm,
        knee=knee_norm,
        hip=hip_norm,
    )


def _collect_trial_pairs(
    eurobench_root: Path,
    include_conditions: set[str] | None = None,
) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for traj in sorted(eurobench_root.rglob("*_Trajectories.csv")):
        match = TRIAL_PATTERN.match(traj.name)
        if match is None:
            continue
        cond = match.group("condition")
        if include_conditions is not None and cond not in include_conditions:
            continue
        events = traj.with_name(traj.name.replace("_Trajectories.csv", "_point_gaitEvents.yaml"))
        if events.exists():
            pairs.append((traj, events))
    return pairs


def _save_cycles_long_csv(cycles: list[TrialCycle], out_csv: Path) -> None:
    rows = []
    for c in cycles:
        for i, pct in enumerate(c.pct):
            rows.append(
                {
                    "subject": c.subject,
                    "condition": c.condition,
                    "run": c.run,
                    "pct": float(pct),
                    "ankle": float(c.ankle[i]),
                    "knee": float(c.knee[i]),
                    "hip": float(c.hip[i]),
                }
            )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def _plot_cycles(cycles: list[TrialCycle], out_png: Path, title: str) -> None:
    pct = cycles[0].pct
    ankle = np.array([c.ankle for c in cycles])
    knee = np.array([c.knee for c in cycles])
    hip = np.array([c.hip for c in cycles])

    mean_ankle = np.nanmean(ankle, axis=0)
    mean_knee = np.nanmean(knee, axis=0)
    mean_hip = np.nanmean(hip, axis=0)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True, constrained_layout=True)
    fig.patch.set_facecolor("#ebebeb")
    for ax in axes:
        ax.set_facecolor("#ebebeb")
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
    fig.suptitle(title, fontsize=17, y=1.01)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ankle, knee and hip gait cycles for biomechanics_human_walking."
    )
    parser.add_argument(
        "--eurobench-root",
        default="data/biomechanics_human_walking/eurobench",
        help="Root directory with Eurobench trajectories/events files.",
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help="Optional conditions to include (e.g., C01 C02 C03). Default: all Cxx.",
    )
    parser.add_argument(
        "--out-dir",
        default="data/biomechanics_human_walking/plots",
        help="Output directory for PNG and CSV files.",
    )
    parser.add_argument(
        "--name",
        default="all",
        help="Output name suffix (e.g., all, C01_C05).",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=101,
        help="Number of normalized points per gait cycle.",
    )
    parser.add_argument(
        "--forward-idx",
        type=int,
        default=1,
        help="Forward axis index in XYZ (default 1 for this dataset).",
    )
    parser.add_argument(
        "--vertical-idx",
        type=int,
        default=2,
        help="Vertical axis index in XYZ (default 2).",
    )
    args = parser.parse_args()

    include_conditions = None
    if args.conditions:
        include_conditions = {c.strip() for c in args.conditions if c.strip()}

    trial_pairs = _collect_trial_pairs(
        eurobench_root=Path(args.eurobench_root),
        include_conditions=include_conditions,
    )
    if not trial_pairs:
        print("status,no_trials")
        return

    cycles: list[TrialCycle] = []
    failures: list[str] = []
    for trajectories_csv, events_yaml in trial_pairs:
        try:
            cycle = _extract_cycle_from_trial(
                trajectories_csv=trajectories_csv,
                events_yaml=events_yaml,
                n_points=args.n_points,
                forward_idx=args.forward_idx,
                vertical_idx=args.vertical_idx,
            )
            cycles.append(cycle)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{trajectories_csv.name}: {exc}")

    if not cycles:
        print("status,no_valid_cycles")
        return

    out_dir = Path(args.out_dir)
    n_subjects = len({c.subject for c in cycles})
    cond_label = "all" if include_conditions is None else ",".join(sorted(include_conditions))
    title = f"biomechanics_human_walking Gait ({cond_label}, subjects={n_subjects}, n={len(cycles)})"

    out_png = out_dir / f"gait_cycle_{args.name}_ankle_knee_hip.png"
    out_csv = out_dir / f"gait_cycle_{args.name}_ankle_knee_hip_long.csv"
    _save_cycles_long_csv(cycles, out_csv)
    _plot_cycles(cycles, out_png, title=title)

    print(f"saved_png,{out_png}")
    print(f"saved_csv,{out_csv}")
    print(f"n_subjects,{n_subjects}")
    print(f"n_cycles,{len(cycles)}")
    print(f"n_failures,{len(failures)}")
    if failures:
        print("first_failures")
        for line in failures[:10]:
            print(line)


if __name__ == "__main__":
    main()

