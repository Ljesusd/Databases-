import argparse
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import yaml

matplotlib.use("Agg")


EVENT_COLORS = {
    "r_heel_strike": "#d62728",
    "r_toe_off": "#ff9896",
    "l_heel_strike": "#1f77b4",
    "l_toe_off": "#9ecae1",
}


def _safe_load_events(events_path: Path) -> dict[str, list[float]]:
    payload = yaml.safe_load(events_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, list[float]] = {}
    for k, vals in payload.items():
        if not isinstance(vals, list):
            continue
        clean = []
        for v in vals:
            try:
                clean.append(float(v))
            except (TypeError, ValueError):
                continue
        if clean:
            out[str(k)] = clean
    return out


def _trial_pairs(
    eurobench_root: Path,
    subject_filter: str | None = None,
    trial_filter: str | None = None,
) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for subject_dir in sorted(p for p in eurobench_root.glob("AB*") if p.is_dir()):
        if subject_filter and subject_dir.name.lower() != subject_filter.lower():
            continue
        for joint_csv in sorted(subject_dir.glob("*_jointAngles.csv")):
            base = joint_csv.name.replace("_jointAngles.csv", "")
            if trial_filter and base.lower() != trial_filter.lower():
                continue
            events_yaml = subject_dir / f"{base}_gaitEvents.yaml"
            if events_yaml.exists():
                pairs.append((joint_csv, events_yaml))
    return pairs


def _plot_trial(
    joint_csv: Path,
    events_yaml: Path,
    out_dir: Path,
    dpi: int,
    overwrite: bool,
    max_line_points: int,
) -> tuple[Path, Path]:
    base = joint_csv.name.replace("_jointAngles.csv", "")
    out_timeseries = out_dir / f"{base}_jointAngles_timeseries.png"
    out_raster = out_dir / f"{base}_gaitEvents_raster.png"
    if not overwrite and out_timeseries.exists() and out_raster.exists():
        return out_timeseries, out_raster

    df = pd.read_csv(joint_csv)
    needed = ["time", "RKneeAngles_x", "LKneeAngles_x", "RAnkleAngles_x", "LAnkleAngles_x"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {joint_csv.name}: {missing}")

    events = _safe_load_events(events_yaml)

    n = len(df)
    step = max(1, math.ceil(n / float(max_line_points))) if max_line_points > 0 else 1
    d = df.iloc[::step, :]

    t = d["time"].to_numpy()
    r_knee = d["RKneeAngles_x"].to_numpy()
    l_knee = d["LKneeAngles_x"].to_numpy()
    r_ankle = d["RAnkleAngles_x"].to_numpy()
    l_ankle = d["LAnkleAngles_x"].to_numpy()

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    axes[0].plot(t, r_knee, color="#d62728", linewidth=1.0, label="RKneeAngles_x")
    axes[0].plot(t, l_knee, color="#1f77b4", linewidth=1.0, label="LKneeAngles_x")
    axes[0].set_ylabel("Angle (deg)")
    axes[0].set_title(f"{base}: Knee angles")
    axes[0].grid(True, alpha=0.2)
    axes[0].legend(loc="upper right")

    axes[1].plot(t, r_ankle, color="#2ca02c", linewidth=1.0, label="RAnkleAngles_x")
    axes[1].plot(t, l_ankle, color="#9467bd", linewidth=1.0, label="LAnkleAngles_x")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Angle (deg)")
    axes[1].set_title(f"{base}: Ankle angles")
    axes[1].grid(True, alpha=0.2)
    axes[1].legend(loc="upper right")

    for key, ev_t in events.items():
        color = EVENT_COLORS.get(key, "#7f7f7f")
        for x in ev_t:
            axes[0].axvline(x, color=color, alpha=0.12, linewidth=0.8)
            axes[1].axvline(x, color=color, alpha=0.12, linewidth=0.8)

    handles = [plt.Line2D([0], [0], color=EVENT_COLORS[k], lw=3) for k in EVENT_COLORS]
    labels = list(EVENT_COLORS.keys())
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_timeseries, dpi=dpi)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 4.5))
    rows = ["r_heel_strike", "r_toe_off", "l_heel_strike", "l_toe_off"]
    ypos = {k: i for i, k in enumerate(rows, start=1)}
    for k in rows:
        ev = events.get(k, []) or []
        y = ypos[k]
        ax.scatter(ev, [y] * len(ev), s=28, color=EVENT_COLORS[k], label=k)

    ax.set_yticks(list(ypos.values()), labels=rows)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{base}: Gait event timeline")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_raster, dpi=dpi)
    plt.close(fig)

    return out_timeseries, out_raster


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch plot joint angles and gait events for benchmark bilateral dataset.")
    parser.add_argument(
        "--eurobench-root",
        default="data/benchmark_datasets_for_bilateral_lower_limb/eurobench",
        help="Root with converted Eurobench files.",
    )
    parser.add_argument(
        "--plots-root",
        default="data/benchmark_datasets_for_bilateral_lower_limb/plots",
        help="Output root for PNG files.",
    )
    parser.add_argument("--subject", default=None, help="Optional subject filter, e.g. AB156.")
    parser.add_argument("--trial", default=None, help="Optional trial filter, e.g. AB156_Circuit_001.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing PNG files.")
    parser.add_argument("--dpi", type=int, default=180, help="PNG DPI.")
    parser.add_argument(
        "--max-line-points",
        type=int,
        default=12000,
        help="Max number of points plotted per line (for performance).",
    )
    args = parser.parse_args()

    eurobench_root = Path(args.eurobench_root)
    plots_root = Path(args.plots_root)
    plots_root.mkdir(parents=True, exist_ok=True)

    pairs = _trial_pairs(
        eurobench_root=eurobench_root,
        subject_filter=args.subject,
        trial_filter=args.trial,
    )
    if not pairs:
        raise FileNotFoundError(f"No trials found in {eurobench_root}")

    rows = []
    total = len(pairs)
    for i, (joint_csv, events_yaml) in enumerate(pairs, start=1):
        subject = joint_csv.parent.name
        out_dir = plots_root / subject
        out_dir.mkdir(parents=True, exist_ok=True)

        row = {
            "subject": subject,
            "trial": joint_csv.name.replace("_jointAngles.csv", ""),
            "status": "ok",
            "error": "",
            "joint_csv": str(joint_csv),
            "events_yaml": str(events_yaml),
            "timeseries_png": "",
            "raster_png": "",
        }
        try:
            out_ts, out_rs = _plot_trial(
                joint_csv=joint_csv,
                events_yaml=events_yaml,
                out_dir=out_dir,
                dpi=int(args.dpi),
                overwrite=bool(args.overwrite),
                max_line_points=int(args.max_line_points),
            )
            row["timeseries_png"] = str(out_ts)
            row["raster_png"] = str(out_rs)
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            row["error"] = str(exc)
        rows.append(row)

        if i == 1 or i % 25 == 0 or i == total:
            print(f"[{i}/{total}] {row['trial']} -> {row['status']}")

    log_df = pd.DataFrame(rows)
    log_path = plots_root / "plot_log.csv"
    log_df.to_csv(log_path, index=False)

    summary = {
        "trials_total": int(len(log_df)),
        "trials_ok": int((log_df["status"] == "ok").sum()),
        "trials_error": int((log_df["status"] == "error").sum()),
        "log_file": str(log_path),
    }
    summary_path = plots_root / "plot_summary.yaml"
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False, allow_unicode=False), encoding="utf-8")

    print(log_path)
    print(summary_path)


if __name__ == "__main__":
    main()
