import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


TRIAL_PATTERN = re.compile(
    r"^subject_(?P<subject>PDFE\d+)_cond_(?P<condition>[a-zA-Z0-9]+)_run_(?P<run>\d+)_imu\.csv$"
)


def _trial_metadata(path: Path) -> dict[str, str]:
    match = TRIAL_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected IMU filename: {path.name}")
    return match.groupdict()


def _episode_lengths(mask: np.ndarray, dt: float) -> list[float]:
    if mask.size == 0:
        return []
    changes = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    return [float((end - start) * dt) for start, end in zip(starts, ends)]


def _analyze_trial(path: Path) -> dict[str, object]:
    meta = _trial_metadata(path)
    df = pd.read_csv(path, sep=";")
    if "time" not in df.columns or "freezing_event_flag" not in df.columns:
        raise ValueError(f"Missing time/freezing_event_flag columns in {path.name}")

    time = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
    flag = pd.to_numeric(df["freezing_event_flag"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    valid = np.isfinite(time)
    time = time[valid]
    flag = flag[valid]
    if time.size < 2:
        raise ValueError(f"Not enough valid samples in {path.name}")

    dt_series = np.diff(time)
    dt_series = dt_series[dt_series > 0]
    dt = float(np.median(dt_series)) if dt_series.size else np.nan
    sample_rate = float(1.0 / dt) if np.isfinite(dt) and dt > 0 else np.nan
    duration = float(time[-1] - time[0]) if time.size else np.nan

    freezing_mask = flag > 0.5
    episode_lengths = _episode_lengths(freezing_mask, dt if np.isfinite(dt) else 0.0)
    total_freezing_s = float(np.sum(episode_lengths))
    freezing_pct = float(100.0 * total_freezing_s / duration) if np.isfinite(duration) and duration > 0 else np.nan

    acc_mag = np.sqrt(
        pd.to_numeric(df["acc_ml_g"], errors="coerce").fillna(0.0).to_numpy() ** 2
        + pd.to_numeric(df["acc_ap_g"], errors="coerce").fillna(0.0).to_numpy() ** 2
        + pd.to_numeric(df["acc_si_g"], errors="coerce").fillna(0.0).to_numpy() ** 2
    )
    gyr_mag = np.sqrt(
        pd.to_numeric(df["gyr_ml_deg_s"], errors="coerce").fillna(0.0).to_numpy() ** 2
        + pd.to_numeric(df["gyr_ap_deg_s"], errors="coerce").fillna(0.0).to_numpy() ** 2
        + pd.to_numeric(df["gyr_si_deg_s"], errors="coerce").fillna(0.0).to_numpy() ** 2
    )

    return {
        "subject": meta["subject"],
        "condition": meta["condition"],
        "run": meta["run"],
        "file": path.name,
        "n_samples": int(time.size),
        "sample_rate_hz": sample_rate,
        "duration_s": duration,
        "freezing_samples": int(np.count_nonzero(freezing_mask)),
        "freezing_total_s": total_freezing_s,
        "freezing_pct": freezing_pct,
        "freezing_episode_count": int(len(episode_lengths)),
        "freezing_episode_max_s": float(max(episode_lengths)) if episode_lengths else 0.0,
        "acc_magnitude_mean_g": float(np.nanmean(acc_mag)),
        "acc_magnitude_std_g": float(np.nanstd(acc_mag)),
        "gyr_magnitude_mean_deg_s": float(np.nanmean(gyr_mag)),
        "gyr_magnitude_std_deg_s": float(np.nanstd(gyr_mag)),
    }


def _plot_subject_freezing(subject_df: pd.DataFrame, out_png: Path, title: str) -> None:
    ordered = subject_df.sort_values("freezing_pct_mean", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 5.5), constrained_layout=True)
    bars = ax.bar(ordered["subject"], ordered["freezing_pct_mean"], color="#3B6FB6", alpha=0.9)
    ax.errorbar(
        ordered["subject"],
        ordered["freezing_pct_mean"],
        yerr=ordered["freezing_pct_std"].fillna(0.0),
        fmt="none",
        ecolor="#1f1f1f",
        elinewidth=1.0,
        capsize=3,
    )
    ax.set_title(title, fontsize=15)
    ax.set_ylabel("freezing time (%)")
    ax.set_xlabel("subject")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(bottom=0.0)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
        label.set_fontsize(8)
    for bar, value in zip(bars, ordered["freezing_pct_mean"]):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{value:.1f}", ha="center", va="bottom", fontsize=7)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize freezing-event labels from the public video/acceleration/angular velocity EUROBENCH IMU files."
    )
    parser.add_argument(
        "--eurobench-root",
        default="data/A public dataset of video, acceleration, angular velocity/eurobench",
        help="Root folder with EUROBENCH IMU files.",
    )
    parser.add_argument(
        "--analysis-root",
        default="data/A public dataset of video, acceleration, angular velocity/analysis",
        help="Output folder for CSV/YAML summaries.",
    )
    parser.add_argument(
        "--plots-root",
        default="data/A public dataset of video, acceleration, angular velocity/plots",
        help="Output folder for plots.",
    )
    args = parser.parse_args()

    eurobench_root = Path(args.eurobench_root)
    analysis_root = Path(args.analysis_root)
    plots_root = Path(args.plots_root)

    imu_files = sorted(eurobench_root.glob("*_imu.csv"))
    if not imu_files:
        raise FileNotFoundError(f"No *_imu.csv files found under {eurobench_root}")

    rows = [_analyze_trial(path) for path in imu_files]
    trials_df = pd.DataFrame(rows).sort_values(["subject", "condition", "run"]).reset_index(drop=True)

    by_subject = (
        trials_df.groupby(["subject", "condition"], dropna=False)
        .agg(
            n_trials=("file", "count"),
            freezing_pct_mean=("freezing_pct", "mean"),
            freezing_pct_std=("freezing_pct", "std"),
            freezing_total_s_mean=("freezing_total_s", "mean"),
            freezing_episode_count_mean=("freezing_episode_count", "mean"),
            duration_s_mean=("duration_s", "mean"),
        )
        .reset_index()
    )

    overall = (
        trials_df.groupby("condition", dropna=False)
        .agg(
            n_trials=("file", "count"),
            freezing_pct_mean=("freezing_pct", "mean"),
            freezing_pct_std=("freezing_pct", "std"),
            freezing_total_s_mean=("freezing_total_s", "mean"),
            freezing_episode_count_mean=("freezing_episode_count", "mean"),
            duration_s_mean=("duration_s", "mean"),
        )
        .reset_index()
    )

    analysis_root.mkdir(parents=True, exist_ok=True)
    trials_csv = analysis_root / "freezing_trial_metrics.csv"
    subject_csv = analysis_root / "freezing_subject_summary.csv"
    overall_csv = analysis_root / "freezing_condition_summary.csv"
    summary_yaml = analysis_root / "freezing_analysis_summary.yaml"

    trials_df.to_csv(trials_csv, index=False)
    by_subject.to_csv(subject_csv, index=False)
    overall.to_csv(overall_csv, index=False)

    walk_subjects = by_subject[by_subject["condition"] == "walk"].copy()
    standing_subjects = by_subject[by_subject["condition"] == "standing"].copy()
    if not walk_subjects.empty:
        _plot_subject_freezing(
            walk_subjects,
            plots_root / "freezing_pct_by_subject_walk.png",
            title=f"Freezing proportion by subject during walk (n={len(walk_subjects)})",
        )
    if not standing_subjects.empty:
        _plot_subject_freezing(
            standing_subjects,
            plots_root / "freezing_pct_by_subject_standing.png",
            title=f"Freezing proportion by subject during standing (n={len(standing_subjects)})",
        )

    summary = {
        "eurobench_root": str(eurobench_root),
        "n_imu_trials": int(len(trials_df)),
        "conditions": sorted(trials_df["condition"].dropna().unique().tolist()),
        "analysis_files": {
            "trial_metrics": str(trials_csv),
            "subject_summary": str(subject_csv),
            "condition_summary": str(overall_csv),
        },
        "plots": {
            "walk": str(plots_root / "freezing_pct_by_subject_walk.png") if not walk_subjects.empty else None,
            "standing": str(plots_root / "freezing_pct_by_subject_standing.png") if not standing_subjects.empty else None,
        },
        "condition_preview": overall.to_dict(orient="records"),
    }
    with summary_yaml.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(summary, fh, sort_keys=False)

    print(f"imu_trials={len(trials_df)}")
    print(f"trial_metrics={trials_csv}")
    print(f"subject_summary={subject_csv}")
    print(f"condition_summary={overall_csv}")


if __name__ == "__main__":
    main()
