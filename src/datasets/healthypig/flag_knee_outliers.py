import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_knee_curve(csv_path: Path, column: str):
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise KeyError(f"Missing '{column}' in {csv_path}")
    return df["pct"].values, df[column].values


def _align_by_peak(
    curve: np.ndarray,
    peak_min: float,
    peak_max: float,
    peak_target: float,
    mode: str,
):
    if mode == "min":
        peak_idx = int(np.argmin(curve))
    else:
        peak_idx = int(np.argmax(curve))
    peak_frac = peak_idx / float(curve.size - 1)
    if peak_frac < peak_min or peak_frac > peak_max:
        target_idx = int(round(peak_target * (curve.size - 1)))
        shift = target_idx - peak_idx
        curve = np.roll(curve, shift)
    return curve, peak_frac


def _label_base(column: str) -> str:
    if column == "knee_flexion":
        return "knee"
    if column == "ankle_dorsiflexion":
        return "ankle"
    if column == "hip_flexion":
        return "hip"
    return column


def flag_knee_outliers(
    processed_root: Path,
    out_dir: Path,
    min_range: float,
    max_rmse: float | None,
    peak_min: float,
    peak_max: float,
    peak_target: float,
    column: str,
    align_mode: str,
):
    subject_dirs = sorted(p for p in processed_root.glob("SUBJ*") if p.is_dir())
    subject_ids = []
    curves = []
    peak_fracs = []
    ranges = []

    for subj_dir in subject_dirs:
        subj = subj_dir.name
        csv_path = subj_dir / f"{subj}_flexion_norm101.csv"
        if not csv_path.exists():
            continue
        pct, knee = _load_knee_curve(csv_path, column)
        if np.any(np.isnan(knee)):
            continue
        knee_aligned, peak_frac = _align_by_peak(
            knee,
            peak_min=peak_min,
            peak_max=peak_max,
            peak_target=peak_target,
            mode=align_mode,
        )
        subject_ids.append(subj)
        curves.append(knee_aligned)
        peak_fracs.append(peak_frac)
        ranges.append(float(knee.max() - knee.min()))

    if not curves:
        raise FileNotFoundError("No valid knee flexion curves found.")

    curves = np.vstack(curves)
    ranges = np.array(ranges)
    peak_fracs = np.array(peak_fracs)

    mean_curve = curves[ranges >= min_range].mean(axis=0)
    rmse = np.sqrt(np.mean((curves - mean_curve) ** 2, axis=1))

    rows = []
    outliers = []
    inliers = []
    for subj, rng, peak_frac, r in zip(subject_ids, ranges, peak_fracs, rmse):
        reason = []
        if rng < min_range:
            reason.append(f"range<{min_range:g}")
        if max_rmse is not None and r > max_rmse:
            reason.append(f"rmse>{max_rmse:g}")
        row = {
            "subject_id": subj,
            "range": rng,
            "peak_frac": peak_frac,
            "rmse": r,
            "reason": ";".join(reason) if reason else "",
        }
        rows.append(row)
        if reason:
            outliers.append(row)
        else:
            inliers.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    base_label = _label_base(column)
    pd.DataFrame(rows).to_csv(
        out_dir / f"{base_label}_outlier_metrics.csv", index=False
    )
    pd.DataFrame(outliers).to_csv(out_dir / f"outliers_{base_label}.csv", index=False)
    pd.DataFrame(inliers).to_csv(out_dir / f"inliers_{base_label}.csv", index=False)

    return len(subject_ids), len(outliers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-root",
        default="data/HealthyPiG/138_HealthyPiG/processed",
        help="Root directory with subject-level normalized outputs",
    )
    parser.add_argument(
        "--out-dir",
        default="data/HealthyPiG/138_HealthyPiG/processed/outliers_knee",
        help="Output directory for outlier reports",
    )
    parser.add_argument("--min-range", type=float, default=5.0)
    parser.add_argument("--max-rmse", type=float, default=20.0)
    parser.add_argument("--peak-min", type=float, default=0.45)
    parser.add_argument("--peak-max", type=float, default=0.9)
    parser.add_argument("--peak-target", type=float, default=0.75)
    parser.add_argument(
        "--align-mode",
        choices=["max", "min"],
        default="max",
        help="Align by maximum or minimum of the curve",
    )
    parser.add_argument(
        "--column",
        default="knee_flexion",
        help="Column name to analyze (e.g. knee_flexion, ankle_dorsiflexion)",
    )
    args = parser.parse_args()

    total, outliers = flag_knee_outliers(
        Path(args.processed_root),
        Path(args.out_dir),
        min_range=args.min_range,
        max_rmse=args.max_rmse,
        peak_min=args.peak_min,
        peak_max=args.peak_max,
        peak_target=args.peak_target,
        column=args.column,
        align_mode=args.align_mode,
    )
    print(f"subjects: {total} outliers: {outliers}")


if __name__ == "__main__":
    main()
