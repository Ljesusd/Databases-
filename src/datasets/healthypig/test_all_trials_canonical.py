import argparse
import os
from pathlib import Path
import site
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_ezc3d_dylib_or_reexec():
    if "DYLD_LIBRARY_PATH" in os.environ and "ezc3d" in os.environ["DYLD_LIBRARY_PATH"]:
        return
    if os.environ.get("EZC3D_DYLD_REEXEC") == "1":
        return
    candidates = [site.getusersitepackages()] + site.getsitepackages()
    for base in candidates:
        ez_dir = Path(base) / "ezc3d"
        if (ez_dir / "libezc3d.dylib").exists():
            current = os.environ.get("DYLD_LIBRARY_PATH", "")
            os.environ["DYLD_LIBRARY_PATH"] = f"{ez_dir}{os.pathsep}{current}" if current else str(ez_dir)
            os.environ["EZC3D_DYLD_REEXEC"] = "1"
            os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_ezc3d_dylib_or_reexec()

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(SRC_ROOT))

from mh_toolbox.conversion.c3d.c3d_eurobench import (
    convert_dir_c3d_to_eurobench_using_predefined_types,
)
try:
    from src.datasets.marker_standardization import (
        TRAJECTORY_MARKER_STANDARDIZATION,
    )
except ModuleNotFoundError:
    from datasets.marker_standardization import (
        TRAJECTORY_MARKER_STANDARDIZATION,
    )
from detect_gait_events_markers import detect_gait_events_markers


def _sanitize_basename(stem: str) -> str:
    return stem.replace(" (", "_").replace(")", "")


def _load_canonical_means(npz_path: Path) -> dict[str, np.ndarray]:
    data = np.load(npz_path)
    return {
        "hip": data["hip_mean"][:, 0],
        "knee": data["knee_mean"][:, 0],
        "ankle": data["ankle_mean"][:, 0],
    }


def _resample_segment(segment: np.ndarray, n_points: int) -> np.ndarray:
    x_old = np.linspace(0.0, 1.0, segment.size)
    x_new = np.linspace(0.0, 1.0, n_points)
    return np.interp(x_new, x_old, segment)


def _best_match_cycle(
    time: np.ndarray,
    knee: np.ndarray,
    canonical: np.ndarray,
    n_points: int,
    min_s: float,
    max_s: float,
    length_step: int,
):
    if time.size < 10:
        raise ValueError("Not enough samples to search for a cycle.")

    dt = float(np.median(np.diff(time)))
    if dt <= 0:
        raise ValueError("Invalid time vector.")

    min_len = max(5, int(round(min_s / dt)))
    max_len = max(min_len + 1, int(round(max_s / dt)))
    max_len = min(max_len, time.size - 1)

    canonical = canonical.astype(float)
    canonical_norm = (canonical - canonical.mean()) / (canonical.std() + 1e-8)

    best = {
        "score": -np.inf,
        "start_idx": None,
        "end_idx": None,
        "cycle": None,
    }

    last_start = time.size - min_len
    for start in range(0, last_start):
        max_len_here = min(max_len, time.size - start)
        for length in range(min_len, max_len_here + 1, length_step):
            seg = knee[start : start + length]
            seg_resamp = _resample_segment(seg, n_points)
            seg_norm = (seg_resamp - seg_resamp.mean()) / (seg_resamp.std() + 1e-8)
            score = float(np.dot(seg_norm, canonical_norm) / canonical_norm.size)
            if score > best["score"]:
                best.update(
                    {
                        "score": score,
                        "start_idx": start,
                        "end_idx": start + length - 1,
                        "cycle": seg_resamp,
                    }
                )

    if best["start_idx"] is None:
        raise ValueError("No candidate cycle found.")

    return best


def _find_col(df: pd.DataFrame, suffix: str, fallback_suffix: str | None = None) -> str:
    if suffix in df.columns:
        return suffix
    matches = [c for c in df.columns if c.endswith(suffix)]
    if matches:
        return matches[0]
    if fallback_suffix:
        if fallback_suffix in df.columns:
            return fallback_suffix
        matches = [c for c in df.columns if c.endswith(fallback_suffix)]
        if matches:
            return matches[0]
    raise KeyError(f"Missing column suffix '{suffix}' in {df.columns}")


def _load_joint_angles(csv_path: Path, angle_scale: float | None = None):
    df = pd.read_csv(csv_path)
    time = df["time"].values
    hip_cols = [
        _find_col(df, "RHipAngles_x", "LHipAngles_x"),
        _find_col(df, "RHipAngles_y", "LHipAngles_y"),
        _find_col(df, "RHipAngles_z", "LHipAngles_z"),
    ]
    knee_cols = [
        _find_col(df, "RKneeAngles_x", "LKneeAngles_x"),
        _find_col(df, "RKneeAngles_y", "LKneeAngles_y"),
        _find_col(df, "RKneeAngles_z", "LKneeAngles_z"),
    ]
    ankle_cols = [
        _find_col(df, "RAnkleAngles_x", "LAnkleAngles_x"),
        _find_col(df, "RAnkleAngles_y", "LAnkleAngles_y"),
        _find_col(df, "RAnkleAngles_z", "LAnkleAngles_z"),
    ]

    hip = df[hip_cols].values
    knee = df[knee_cols].values
    ankle = df[ankle_cols].values

    if angle_scale is not None:
        hip = hip * angle_scale
        knee = knee * angle_scale
        ankle = ankle * angle_scale

    return time, hip, knee, ankle


def _plot_trial_profiles(out_path: Path, pct: np.ndarray, hip, knee, ankle, title: str):
    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    panels = [
        ("ANKLE", ankle[:, 0]),
        ("KNEE", knee[:, 0]),
        ("HIP", hip[:, 0]),
    ]
    for ax, (name, curve) in zip(axes, panels):
        ax.plot(pct, curve, color="black", linewidth=1.5)
        ax.set_title(name)
        ax.set_ylabel("ANGLE (DEGREES)")
        ax.set_xlim(0, 100)
    axes[-1].set_xlabel("PERCENT OF GAIT CYCLE (%)")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_trial_profiles_1d(
    out_path: Path, pct: np.ndarray, hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray, title: str
):
    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    panels = [
        ("ANKLE", ankle),
        ("KNEE", knee),
        ("HIP", hip),
    ]
    for ax, (name, curve) in zip(axes, panels):
        ax.plot(pct, curve, color="black", linewidth=1.5)
        ax.set_title(name)
        ax.set_ylabel("ANGLE (DEGREES)")
        ax.set_xlim(0, 100)
    axes[-1].set_xlabel("PERCENT OF GAIT CYCLE (%)")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_knee_overlay(out_path: Path, pct: np.ndarray, canonical: np.ndarray, cycles: list[np.ndarray]):
    fig, ax = plt.subplots(figsize=(6, 4))
    for curve in cycles:
        ax.plot(pct, curve, color="gray", alpha=0.3, linewidth=0.9)
    ax.plot(pct, canonical, color="black", linewidth=1.8, label="Canonical")
    ax.set_title("Knee flexion: trials vs canonical")
    ax.set_xlabel("PERCENT OF GAIT CYCLE (%)")
    ax.set_ylabel("ANGLE (DEGREES)")
    ax.set_xlim(0, 100)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_all_trials_overlay(
    out_path: Path,
    pct: np.ndarray,
    canonical: dict[str, np.ndarray],
    trials: list[dict[str, np.ndarray]],
    subject_id: str,
):
    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    panels = [
        ("ANKLE", "ankle"),
        ("KNEE", "knee"),
        ("HIP", "hip"),
    ]
    for ax, (title, key) in zip(axes, panels):
        for trial in trials:
            ax.plot(pct, trial[key], color="gray", alpha=0.35, linewidth=0.9)
        ax.plot(pct, canonical[key], color="black", linewidth=1.8, label="Canonical")
        ax.set_title(title)
        ax.set_ylabel("ANGLE (DEGREES)")
        ax.set_xlim(0, 100)
    axes[-1].set_xlabel("PERCENT OF GAIT CYCLE (%)")
    fig.suptitle(f"{subject_id} all trials vs canonical")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def process_subject(
    subject_dir: Path,
    eurobench_root: Path,
    processed_root: Path,
    plots_root: Path,
    canonical_means: dict[str, np.ndarray],
    n_points: int,
    min_s: float,
    max_s: float,
    length_step: int,
    best_only: bool = False,
    only_trial: int | None = None,
):
    import ezc3d

    subject_id = subject_dir.name
    out_eurobench = eurobench_root / subject_id
    out_processed = processed_root / subject_id
    out_plots = plots_root / subject_id

    out_eurobench.mkdir(parents=True, exist_ok=True)
    out_processed.mkdir(parents=True, exist_ok=True)
    out_plots.mkdir(parents=True, exist_ok=True)

    results = []
    knee_cycles = []
    trial_curves = []
    pct = np.linspace(0.0, 100.0, n_points)

    for c3d_file in sorted(subject_dir.glob("*.c3d")):
        if only_trial is not None:
            if f"({only_trial})" not in c3d_file.stem:
                continue
        trial_basename = _sanitize_basename(c3d_file.stem)
        try:
            trajectories_csv = out_eurobench / f"{c3d_file.stem}_Trajectories.csv"
            info_yaml = out_eurobench / f"{c3d_file.stem}_info.yaml"
            events_yaml = out_eurobench / f"{c3d_file.stem}_events.yaml"
            gait_events_yaml = out_eurobench / f"{c3d_file.stem}_gaitEvents.yaml"

            needs_trajectory_export = (
                not trajectories_csv.exists()
                or not info_yaml.exists()
                or not events_yaml.exists()
            )
            if needs_trajectory_export:
                convert_dir_c3d_to_eurobench_using_predefined_types(
                    dir_in=str(subject_dir),
                    dir_out=str(out_eurobench),
                    predefined_types="TRAJECTORY",
                    list_filter_cols=[],
                    pattern_c3d=c3d_file.name,
                    pattern_subject_condition_run=r"(?P<condition>SUBJ)(?P<subject>\d+)\s*\((?P<run>\d+)\)",
                    group_names=["subject", "condition", "run"],
                    b_save_data=True,
                    b_save_analogs=False,
                    b_save_events=True,
                    b_save_info=True,
                    writing_mode="w",
                    **TRAJECTORY_MARKER_STANDARDIZATION,
                )
            if trajectories_csv.exists() and not gait_events_yaml.exists():
                detect_gait_events_markers(
                    str(trajectories_csv),
                    out_yaml=str(gait_events_yaml),
                    axis_mode="vertical",
                    axis_override="z",
                )

            angles_csv = out_eurobench / f"{c3d_file.stem}_jointAngles.csv"
            if not angles_csv.exists():
                convert_dir_c3d_to_eurobench_using_predefined_types(
                    dir_in=str(subject_dir),
                    dir_out=str(out_eurobench),
                    predefined_types="ANGLE",
                    list_filter_cols=[],
                    pattern_c3d=c3d_file.name,
                    pattern_subject_condition_run=r"(?P<condition>SUBJ)(?P<subject>\d+)\s*\((?P<run>\d+)\)",
                    group_names=["subject", "condition", "run"],
                    b_save_data=True,
                    b_save_analogs=False,
                    b_save_events=False,
                    b_save_info=False,
                    writing_mode="w",
                )
            if not angles_csv.exists():
                continue

            c3d_reader = ezc3d.c3d(str(c3d_file))
            point_scale = c3d_reader["parameters"]["POINT"]["SCALE"]["value"][0]
            angle_scale = (1.0 / point_scale) if point_scale != 0 else None

            time, hip, knee, ankle = _load_joint_angles(angles_csv, angle_scale=angle_scale)
            knee_signal = knee[:, 0]

            best = _best_match_cycle(
                time,
                knee_signal,
                canonical_means["knee"],
                n_points=n_points,
                min_s=min_s,
                max_s=max_s,
                length_step=length_step,
            )

            start_idx = int(best["start_idx"])
            end_idx = int(best["end_idx"])
            start_t = float(time[start_idx])
            end_t = float(time[end_idx])

            hip_seg = hip[start_idx : end_idx + 1]
            knee_seg = knee[start_idx : end_idx + 1]
            ankle_seg = ankle[start_idx : end_idx + 1]

            hip_norm = np.column_stack(
                [
                    _resample_segment(hip_seg[:, 0], n_points),
                    _resample_segment(hip_seg[:, 1], n_points),
                    _resample_segment(hip_seg[:, 2], n_points),
                ]
            )
            knee_norm = np.column_stack(
                [
                    _resample_segment(knee_seg[:, 0], n_points),
                    _resample_segment(knee_seg[:, 1], n_points),
                    _resample_segment(knee_seg[:, 2], n_points),
                ]
            )
            ankle_norm = np.column_stack(
                [
                    _resample_segment(ankle_seg[:, 0], n_points),
                    _resample_segment(ankle_seg[:, 1], n_points),
                    _resample_segment(ankle_seg[:, 2], n_points),
                ]
            )

            knee_cycles.append(knee_norm[:, 0])
            trial_curves.append(
                {
                    "trial": trial_basename,
                    "hip": hip_norm[:, 0],
                    "knee": knee_norm[:, 0],
                    "ankle": ankle_norm[:, 0],
                }
            )

            if not best_only:
                df_out = pd.DataFrame(
                    {
                        "pct": pct,
                        "hip_flexion": hip_norm[:, 0],
                        "knee_flexion": knee_norm[:, 0],
                        "ankle_dorsiflexion": ankle_norm[:, 0],
                    }
                )
                df_out.to_csv(
                    out_processed / f"{trial_basename}_canonical_flexion_norm101.csv",
                    index=False,
                )
                np.savez(
                    out_processed / f"{trial_basename}_canonical_flexion_norm101.npz",
                    pct=pct,
                    hip_flexion=hip_norm[:, 0],
                    knee_flexion=knee_norm[:, 0],
                    ankle_dorsiflexion=ankle_norm[:, 0],
                )

                _plot_trial_profiles(
                    out_plots / f"{trial_basename}_canonical_profiles.png",
                    pct,
                    hip_norm,
                    knee_norm,
                    ankle_norm,
                    title=f"{subject_id} {trial_basename} canonical cycle",
                )

            results.append(
                {
                    "subject_id": subject_id,
                    "trial": trial_basename,
                    "start_time": start_t,
                    "end_time": end_t,
                    "score": best["score"],
                    "samples": int(end_idx - start_idx + 1),
                }
            )
        except Exception as exc:
            print(f"Skipping {subject_id} {trial_basename}: {exc}")
            continue

    best_curve = None
    if results:
        pd.DataFrame(results).to_csv(out_processed / "canonical_cycles_summary.csv", index=False)
        if knee_cycles and not best_only:
            _plot_knee_overlay(
                out_plots / f"{subject_id}_knee_canonical_overlay.png",
                pct,
                canonical_means["knee"],
                knee_cycles,
            )
        if trial_curves and not best_only:
            _plot_all_trials_overlay(
                out_plots / f"{subject_id}_all_trials_canonical_overlay.png",
                pct,
                canonical_means,
                trial_curves,
                subject_id=subject_id,
            )

        best_idx = int(np.argmax([r["score"] for r in results]))
        best_trial = results[best_idx]["trial"]
        best_curve = trial_curves[best_idx]
        if best_only and best_curve is not None:
            df_out = pd.DataFrame(
                {
                    "pct": pct,
                    "hip_flexion": best_curve["hip"],
                    "knee_flexion": best_curve["knee"],
                    "ankle_dorsiflexion": best_curve["ankle"],
                }
            )
            df_out.to_csv(
                out_processed / f"{best_trial}_best_canonical_flexion_norm101.csv",
                index=False,
            )
            np.savez(
                out_processed / f"{best_trial}_best_canonical_flexion_norm101.npz",
                pct=pct,
                hip_flexion=best_curve["hip"],
                knee_flexion=best_curve["knee"],
                ankle_dorsiflexion=best_curve["ankle"],
            )
            _plot_trial_profiles_1d(
                out_plots / f"{best_trial}_best_canonical_profiles.png",
                pct,
                best_curve["hip"],
                best_curve["knee"],
                best_curve["ankle"],
                title=f"{subject_id} {best_trial} BEST canonical cycle",
            )

    return best_curve, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        default="data/HealthyPiG/138_HealthyPiG/raw",
        help="Root directory with raw subject folders",
    )
    parser.add_argument(
        "--test-root",
        default="data/HealthyPiG/138_HealthyPiG/test",
        help="Root directory for test outputs",
    )
    parser.add_argument(
        "--canonical-npz",
        default="data/HealthyPiG/138_HealthyPiG/processed/population_angles_norm101.npz",
        help="Canonical population angles NPZ (knee_mean used).",
    )
    parser.add_argument("--subject", default="SUBJ138", help="Subject ID, e.g. SUBJ138")
    parser.add_argument("--all", action="store_true", help="Process all subjects")
    parser.add_argument("--best-only", action="store_true", help="Save only the best trial per subject")
    parser.add_argument(
        "--only-trial",
        type=int,
        default=None,
        help="Process only a specific trial index (e.g., 0, 1, 2, 3).",
    )
    parser.add_argument("--n-points", type=int, default=101)
    parser.add_argument("--min-s", type=float, default=0.8)
    parser.add_argument("--max-s", type=float, default=1.6)
    parser.add_argument("--length-step", type=int, default=2)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    test_root = Path(args.test_root)
    eurobench_root = test_root / "eurobench"
    processed_root = test_root / "processed"
    plots_root = test_root / "plots"

    canonical_means = _load_canonical_means(Path(args.canonical_npz))

    if args.all:
        subject_dirs = sorted(p for p in raw_root.glob("SUBJ*") if p.is_dir())
    else:
        subject_dirs = [raw_root / args.subject]

    best_rows = []
    best_curves = []
    for subject_dir in subject_dirs:
        if not subject_dir.exists():
            print(f"Skipping missing {subject_dir}")
            continue
        best_curve, results = process_subject(
            subject_dir,
            eurobench_root,
            processed_root,
            plots_root,
            canonical_means=canonical_means,
            n_points=args.n_points,
            min_s=args.min_s,
            max_s=args.max_s,
            length_step=args.length_step,
            best_only=args.best_only,
            only_trial=args.only_trial,
        )
        if results:
            best_idx = int(np.argmax([r["score"] for r in results]))
            best_rows.append(results[best_idx])
        if best_curve is not None:
            best_curves.append(best_curve)

    if args.best_only and best_rows:
        summary_path = processed_root / "best_trials_summary.csv"
        pd.DataFrame(best_rows).to_csv(summary_path, index=False)

        if best_curves:
            pct = np.linspace(0.0, 100.0, args.n_points)
            _plot_all_trials_overlay(
                plots_root / "best_trials_all_subjects_overlay.png",
                pct,
                canonical_means,
                best_curves,
                subject_id="BEST_TRIALS_ALL_SUBJECTS",
            )


if __name__ == "__main__":
    main()
