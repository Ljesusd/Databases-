import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")


EVENT_STYLES = {
    "r_heel_strike": {"label": "IC", "color": "#0b6e8e"},
    "l_toe_off": {"label": "OT", "color": "#3b8ea5"},
    "l_heel_strike": {"label": "OI", "color": "#3b8ea5"},
    "r_toe_off": {"label": "TO", "color": "#0b6e8e"},
}

PHASE_COLORS = ["#deebef", "#e8f1f4", "#deebef", "#e8f1f4", "#deebef"]


def _safe_events(path: Path) -> dict[str, np.ndarray]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, np.ndarray] = {}
    for k, v in payload.items():
        if not isinstance(v, list):
            continue
        vals = []
        for x in v:
            try:
                vals.append(float(x))
            except (TypeError, ValueError):
                continue
        if vals:
            out[str(k)] = np.array(sorted(set(vals)), dtype=float)
    return out


def _pick_cycle(events: dict[str, np.ndarray]) -> tuple[float, float]:
    hs = events.get("r_heel_strike", np.array([], dtype=float))
    to = events.get("r_toe_off", np.array([], dtype=float))

    if hs.size >= 2:
        return float(hs[0]), float(hs[1])
    if hs.size == 1 and to.size > 0:
        hs0 = float(hs[0])
        to_after = to[to > hs0]
        if to_after.size == 0:
            raise ValueError("r_toe_off does not occur after r_heel_strike")
        stride = 2.0 * (float(to_after[0]) - hs0)
        if stride <= 0:
            raise ValueError("Invalid synthetic stride duration")
        return hs0, hs0 + stride
    raise ValueError("Not enough right gait events to define one cycle")


def _first_between(arr: np.ndarray, start: float, end: float) -> float | None:
    if arr.size == 0:
        return None
    vals = arr[(arr > start) & (arr < end)]
    if vals.size == 0:
        return None
    return float(vals[0])


def _normalize_signal(
    time: np.ndarray,
    signal: np.ndarray,
    start: float,
    end: float,
    n_points: int,
) -> np.ndarray:
    mask = (time >= start) & (time <= end)
    t = time[mask]
    y = signal[mask]
    if t.size < 2:
        raise ValueError("Not enough samples inside cycle window")
    dt = t[-1] - t[0]
    if dt <= 0:
        raise ValueError("Cycle duration is zero")
    t_norm = (t - t[0]) / dt
    x = np.linspace(0.0, 1.0, n_points)
    return np.interp(x, t_norm, y)


def _event_pct(t_event: float | None, start: float, end: float) -> float | None:
    if t_event is None:
        return None
    if end <= start:
        return None
    pct = 100.0 * (t_event - start) / (end - start)
    if pct <= 0.0 or pct >= 100.0:
        return None
    return float(pct)


def _trial_pairs(root: Path, subject: str | None, trial: str | None) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for subject_dir in sorted(p for p in root.glob("AB*") if p.is_dir()):
        if subject and subject_dir.name.lower() != subject.lower():
            continue
        for joint_csv in sorted(subject_dir.glob("*_jointAngles.csv")):
            base = joint_csv.name.replace("_jointAngles.csv", "")
            if trial and base.lower() != trial.lower():
                continue
            events_yaml = subject_dir / f"{base}_gaitEvents.yaml"
            if events_yaml.exists():
                pairs.append((joint_csv, events_yaml))
    return pairs


def _build_phase_bounds(event_marks: list[tuple[str, float]]) -> list[float]:
    bounds = [0.0]
    for key, pct in event_marks:
        if key in {"l_toe_off", "l_heel_strike", "r_toe_off"}:
            if pct > bounds[-1] and pct < 100.0:
                bounds.append(pct)
    if bounds[-1] != 100.0:
        bounds.append(100.0)
    return bounds


def _style_axis(ax, ylim_pad_ratio: float = 0.08) -> None:
    y0, y1 = ax.get_ylim()
    if np.isfinite(y0) and np.isfinite(y1) and y1 > y0:
        pad = (y1 - y0) * ylim_pad_ratio
        ax.set_ylim(y0 - pad, y1 + pad)
    ax.grid(True, alpha=0.18, linestyle="-", linewidth=0.7)
    ax.axhline(0.0, color="#6f7e86", linewidth=0.8, alpha=0.6)


def _plot_one_trial(
    joint_csv: Path,
    events_yaml: Path,
    out_dir: Path,
    n_points: int,
    dpi: int,
    overwrite: bool,
) -> tuple[Path, Path]:
    base = joint_csv.name.replace("_jointAngles.csv", "")
    out_png = out_dir / f"{base}_gaitCycle_profile.png"
    out_csv = out_dir / f"{base}_gaitCycle_norm101.csv"
    if not overwrite and out_png.exists() and out_csv.exists():
        return out_png, out_csv

    df = pd.read_csv(joint_csv)
    required = ["time", "RKneeAngles_x", "RAnkleAngles_x", "RHipAngles_x"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    time = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
    knee = pd.to_numeric(df["RKneeAngles_x"], errors="coerce").to_numpy(dtype=float)
    ankle = pd.to_numeric(df["RAnkleAngles_x"], errors="coerce").to_numpy(dtype=float)
    hip = pd.to_numeric(df["RHipAngles_x"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(time) & np.isfinite(knee) & np.isfinite(ankle) & np.isfinite(hip)
    time, knee, ankle, hip = time[valid], knee[valid], ankle[valid], hip[valid]
    if time.size < 3:
        raise ValueError("Not enough valid time-angle samples")

    events = _safe_events(events_yaml)
    start, end = _pick_cycle(events)
    if end > time[-1]:
        end = float(time[-1])
    if end <= start:
        raise ValueError("Invalid cycle after clamping to time range")

    knee_norm = _normalize_signal(time, knee, start, end, n_points=n_points)
    ankle_norm = _normalize_signal(time, ankle, start, end, n_points=n_points)
    hip_norm = _normalize_signal(time, hip, start, end, n_points=n_points)
    pct = np.linspace(0.0, 100.0, n_points)

    l_to = _first_between(events.get("l_toe_off", np.array([], dtype=float)), start, end)
    l_hs = _first_between(events.get("l_heel_strike", np.array([], dtype=float)), start, end)
    r_to = _first_between(events.get("r_toe_off", np.array([], dtype=float)), start, end)

    event_marks: list[tuple[str, float]] = [("r_heel_strike", 0.0)]
    for key, tv in [("l_toe_off", l_to), ("l_heel_strike", l_hs), ("r_toe_off", r_to)]:
        p = _event_pct(tv, start, end)
        if p is not None:
            event_marks.append((key, p))
    event_marks.append(("r_heel_strike", 100.0))
    event_marks = sorted(event_marks, key=lambda x: x[1])

    phase_bounds = _build_phase_bounds(event_marks)

    hip_missing = float(np.nanstd(hip_norm)) < 1e-8
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 10.0), sharex=True)
    fig.patch.set_facecolor("#eef4f7")
    for ax in axes:
        ax.set_facecolor("#eef4f7")
        for i in range(len(phase_bounds) - 1):
            a = phase_bounds[i]
            b = phase_bounds[i + 1]
            ax.axvspan(a, b, color=PHASE_COLORS[i % len(PHASE_COLORS)], alpha=0.8, zorder=0)

    axes[0].plot(pct, hip_norm, color="#121212", linewidth=2.0)
    axes[0].set_ylabel("Hip (deg)")
    axes[0].set_title("Hip (N/A in source)" if hip_missing else "Hip")
    _style_axis(axes[0])

    axes[1].plot(pct, knee_norm, color="#121212", linewidth=2.0)
    axes[1].set_ylabel("Knee (deg)")
    axes[1].set_title("Knee")
    _style_axis(axes[1])

    axes[2].plot(pct, ankle_norm, color="#121212", linewidth=2.0)
    axes[2].set_ylabel("Ankle (deg)")
    axes[2].set_title("Ankle")
    axes[2].set_xlabel("Gait cycle (%)")
    _style_axis(axes[2])

    for ax in axes:
        for k, p in event_marks:
            color = EVENT_STYLES.get(k, {}).get("color", "#4f636e")
            ax.axvline(p, color=color, linewidth=1.0, linestyle=":", alpha=0.95)
        ax.set_xlim(0.0, 100.0)

    # Event labels on top panel.
    used = {}
    for k, p in event_marks:
        label = EVENT_STYLES.get(k, {}).get("label", k)
        if label == "IC" and p >= 99.5:
            label = "IC"
        if label in used and abs(p - used[label]) < 1e-6:
            continue
        used[label] = p
        axes[0].text(
            p,
            1.06,
            label,
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=10,
            color="#183642",
        )

    phase_labels: list[tuple[float, str]] = []
    if len(phase_bounds) >= 2:
        phase_labels.append(((phase_bounds[0] + phase_bounds[1]) / 2.0, "Loading\nresponse"))
    if len(phase_bounds) >= 3:
        phase_labels.append(((phase_bounds[1] + phase_bounds[2]) / 2.0, "Mid-stance"))
    if len(phase_bounds) >= 4:
        phase_labels.append(((phase_bounds[2] + phase_bounds[3]) / 2.0, "Terminal\nstance"))
    if len(phase_bounds) >= 5:
        phase_labels.append(((phase_bounds[3] + phase_bounds[4]) / 2.0, "Swing"))

    for x, txt in phase_labels:
        axes[0].text(
            x,
            0.93,
            txt,
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8.5,
            color="#274754",
        )

    fig.suptitle(base, y=0.995, fontsize=12.5)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.985])
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)

    norm_df = pd.DataFrame(
        {
            "pct": pct,
            "RHipAngles_x": hip_norm,
            "RKneeAngles_x": knee_norm,
            "RAnkleAngles_x": ankle_norm,
            "cycle_start_s": np.full_like(pct, fill_value=start, dtype=float),
            "cycle_end_s": np.full_like(pct, fill_value=end, dtype=float),
        }
    )
    norm_df.to_csv(out_csv, index=False)
    return out_png, out_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create gait-cycle profile plots (0-100%) for benchmark bilateral lower limb Eurobench files."
    )
    parser.add_argument(
        "--eurobench-root",
        default="data/benchmark_datasets_for_bilateral_lower_limb/eurobench",
        help="Root with *_jointAngles.csv and *_gaitEvents.yaml files.",
    )
    parser.add_argument(
        "--plots-root",
        default="data/benchmark_datasets_for_bilateral_lower_limb/plots",
        help="Output root for plots and normalized cycle CSV files.",
    )
    parser.add_argument("--subject", default=None, help="Optional subject filter (e.g., AB156).")
    parser.add_argument("--trial", default=None, help="Optional trial filter (e.g., AB156_Circuit_001).")
    parser.add_argument("--n-points", type=int, default=101, help="Normalized cycle points.")
    parser.add_argument("--dpi", type=int, default=220, help="Figure DPI.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    args = parser.parse_args()

    eurobench_root = Path(args.eurobench_root)
    plots_root = Path(args.plots_root)
    plots_root.mkdir(parents=True, exist_ok=True)

    pairs = _trial_pairs(
        root=eurobench_root,
        subject=args.subject,
        trial=args.trial,
    )
    if not pairs:
        raise FileNotFoundError(f"No trial pairs found in {eurobench_root}")

    rows = []
    total = len(pairs)
    for i, (joint_csv, events_yaml) in enumerate(pairs, start=1):
        subject_dir = plots_root / joint_csv.parent.name
        subject_dir.mkdir(parents=True, exist_ok=True)
        trial_base = joint_csv.name.replace("_jointAngles.csv", "")
        row = {
            "subject": joint_csv.parent.name,
            "trial": trial_base,
            "status": "ok",
            "error": "",
            "joint_csv": str(joint_csv),
            "events_yaml": str(events_yaml),
            "out_profile_png": "",
            "out_norm_csv": "",
        }
        try:
            out_png, out_csv = _plot_one_trial(
                joint_csv=joint_csv,
                events_yaml=events_yaml,
                out_dir=subject_dir,
                n_points=int(args.n_points),
                dpi=int(args.dpi),
                overwrite=bool(args.overwrite),
            )
            row["out_profile_png"] = str(out_png)
            row["out_norm_csv"] = str(out_csv)
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            row["error"] = str(exc)

        rows.append(row)
        if i == 1 or i % 25 == 0 or i == total:
            print(f"[{i}/{total}] {trial_base} -> {row['status']}")

    log_df = pd.DataFrame(rows)
    log_path = plots_root / "gait_cycle_profile_log.csv"
    log_df.to_csv(log_path, index=False)

    summary = {
        "trials_total": int(len(log_df)),
        "trials_ok": int((log_df["status"] == "ok").sum()),
        "trials_error": int((log_df["status"] == "error").sum()),
        "log_file": str(log_path),
    }
    summary_path = plots_root / "gait_cycle_profile_summary.yaml"
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False, allow_unicode=False), encoding="utf-8")

    print(log_path)
    print(summary_path)


if __name__ == "__main__":
    main()
