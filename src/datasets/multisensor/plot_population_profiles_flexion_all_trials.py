import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KNEE_ROM_MIN = 30.0
HIP_ROM_MIN = 10.0
PEAK_MIN = 60.0
PEAK_MAX = 80.0
SQUARED_FRAMES = 10  # frames planos consecutivos
SQUARED_ATOL = 1e-1  # tolerancia mayor para detectar mesetas con ruido
HIP_STEP_MAX = 80.0
HIP_ABS_MAX = 180.0


def _load_flexion_csv(path: Path):
    df = pd.read_csv(path)
    required = {"pct", "hip_flexion", "knee_flexion", "ankle_dorsiflexion"}
    if not required.issubset(df.columns):
        return None
    return df


def _day_from_name(name: str) -> str:
    if "_day1_" in name:
        return "day1"
    if "_day2_" in name:
        return "day2"
    return "unknown"


def _is_squared(signal: np.ndarray) -> bool:
    # Detecta mesetas: diferencias consecutivas ~0 por SQUARED_FRAMES
    diffs = np.diff(signal)
    is_flat = np.isclose(diffs, 0, atol=SQUARED_ATOL)
    run = 0
    for flat in is_flat:
        if flat:
            run += 1
            if run >= SQUARED_FRAMES:
                return True
        else:
            run = 0
    return False


def _fix_angle_continuity(angles: np.ndarray) -> np.ndarray:
    """
    Unwrap and recenter hip angles to reduce +/-180 jumps.
    """
    rad = np.deg2rad(angles)
    unwrapped = np.unwrap(rad)
    deg = np.rad2deg(unwrapped)
    mean_val = float(np.mean(deg))
    shift = round(mean_val / 360.0) * 360.0
    deg = deg - shift
    if np.mean(deg) > 90.0:
        deg = deg - 180.0
    if np.mean(deg) < -90.0:
        deg = deg + 180.0
    return deg


def dynamic_time_warp_alignment(curve: np.ndarray, peak_idx: int, target_peak_pct: float = 70.0, n_points: int = 101):
    """
    Re-muestrea la curva en dos fases para alinear el pico al porcentaje deseado.
    """
    n_points_pre = max(2, int(round(n_points * (target_peak_pct / 100.0))))
    n_points_post = max(2, n_points - n_points_pre)

    pre_peak = np.interp(
        np.linspace(0, 1, n_points_pre),
        np.linspace(0, 1, peak_idx + 1),
        curve[: peak_idx + 1],
    )
    post_peak = np.interp(
        np.linspace(0, 1, n_points_post),
        np.linspace(0, 1, len(curve) - peak_idx),
        curve[peak_idx:],
    )
    # evitar duplicar el punto del pico
    aligned = np.concatenate([pre_peak[:-1], post_peak])
    # Si por redondeo falta/sobra un punto, ajustar
    if aligned.size < n_points:
        aligned = np.concatenate([aligned, [aligned[-1]]])
    elif aligned.size > n_points:
        aligned = aligned[:n_points]
    return aligned


def plot_population_all_trials(
    processed_root: Path,
    out_path: Path,
    summary_path: Path | None = None,
    hip_qc: bool = False,
    hip_step_max: float = HIP_STEP_MAX,
    hip_abs_max: float = HIP_ABS_MAX,
    hip_fix_continuity: bool = False,
):
    files = sorted(processed_root.glob("user*/**/*_marker_angles_norm101.csv"))
    if not files:
        raise FileNotFoundError("No *_marker_angles_norm101.csv found.")

    pct_ref = None
    hips, knees, ankles = [], [], []
    rows = []
    total_files = 0
    kept_files = 0

    for path in files:
        total_files += 1
        df = _load_flexion_csv(path)
        if df is None:
            continue
        pct = df["pct"].values
        if pct_ref is None:
            pct_ref = pct
        elif not np.allclose(pct_ref, pct):
            continue

        hip_curve_raw = df["hip_flexion"].values
        hip_curve = _fix_angle_continuity(hip_curve_raw) if hip_fix_continuity else hip_curve_raw
        knee_curve = df["knee_flexion"].values
        ankle_curve = df["ankle_dorsiflexion"].values

        hip_rom = float(hip_curve.max() - hip_curve.min())
        knee_rom = float(knee_curve.max() - knee_curve.min())
        hip_step_val = float(np.max(np.abs(np.diff(hip_curve)))) if hip_curve.size > 1 else np.nan
        hip_absmax = float(np.max(np.abs(hip_curve)))
        peak_idx = int(np.argmax(knee_curve))
        peak_pct = float(pct[peak_idx])

        squared_knee = _is_squared(knee_curve)
        squared_hip = _is_squared(hip_curve)

        base_valid = (
            (knee_rom > KNEE_ROM_MIN)
            and (hip_rom > HIP_ROM_MIN)
            and (PEAK_MIN <= peak_pct <= PEAK_MAX)
            and not squared_knee
            and not squared_hip
        )
        hip_qc_valid = (hip_step_val <= hip_step_max) and (hip_absmax <= hip_abs_max)
        valid = base_valid and (hip_qc_valid if hip_qc else True)

        rows.append(
            {
                "user": path.parent.name,
                "trial": path.name,
                "day": _day_from_name(path.name),
                "knee_rom": knee_rom,
                "hip_rom": hip_rom,
                "knee_peak_pct": peak_pct,
                "squared_knee": squared_knee,
                "squared_hip": squared_hip,
                "hip_max_step": hip_step_val,
                "hip_absmax": hip_absmax,
                "hip_qc_enabled": hip_qc,
                "hip_qc_valid": hip_qc_valid,
                "base_valid": base_valid,
                "kept": valid,
            }
        )

        if not valid:
            continue

        aligned_knee = dynamic_time_warp_alignment(knee_curve, peak_idx, target_peak_pct=70.0)
        aligned_hip = dynamic_time_warp_alignment(hip_curve, peak_idx, target_peak_pct=70.0)
        aligned_ankle = dynamic_time_warp_alignment(ankle_curve, peak_idx, target_peak_pct=70.0)

        hips.append(aligned_hip)
        knees.append(aligned_knee)
        ankles.append(aligned_ankle)
        kept_files += 1

    if not hips:
        raise FileNotFoundError("No usable flexion curves found after QC.")

    hips = np.stack(hips, axis=0)
    knees = np.stack(knees, axis=0)
    ankles = np.stack(ankles, axis=0)

    median_hip = np.median(hips, axis=0)
    median_knee = np.median(knees, axis=0)
    median_ankle = np.median(ankles, axis=0)

    landmarks = [
        ("ANKLE", ankles, median_ankle),
        ("KNEE", knees, median_knee),
        ("HIP", hips, median_hip),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    for ax, (name, all_vals, mean_vals) in zip(axes, landmarks):
        for trial_curve in all_vals:
            ax.plot(pct_ref, trial_curve, color="gray", alpha=0.2, linewidth=0.7)
        ax.plot(pct_ref, mean_vals, color="black", linewidth=1.5)
        ax.set_title(name)
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.set_xlim(0, 100)

    axes[-1].set_xlabel("PERCENT OF GAIT CYCLE (%)")
    qc_label = "on" if hip_qc else "off"
    fig.suptitle(f"multisensor_gait all trials (n={len(hips)} kept of {total_files}, hip_qc={qc_label})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(
        f"kept={len(hips)} of {total_files}, "
        f"hip_qc={hip_qc}, hip_fix_continuity={hip_fix_continuity}, "
        f"hip_max_step={hip_step_max}, hip_abs_max={hip_abs_max}"
    )

    if summary_path:
        df_sum = pd.DataFrame(rows)
        df_sum.to_csv(summary_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-root",
        default="data/multisensor_gait/processed",
        help="Root with per-trial marker angles",
    )
    parser.add_argument(
        "--out-path",
        default="data/multisensor_gait/test/plots/population_profiles_flexion_all_trials.png",
        help="Output plot path",
    )
    parser.add_argument(
        "--summary-path",
        default="data/multisensor_gait/test/analysis/all_trials_summary.csv",
        help="CSV with list of trials used",
    )
    parser.add_argument(
        "--hip-qc",
        action="store_true",
        help="Enable additional hip quality control (step and absolute bounds).",
    )
    parser.add_argument(
        "--hip-max-step",
        type=float,
        default=HIP_STEP_MAX,
        help="Max allowed frame-to-frame hip jump (degrees) when --hip-qc is enabled.",
    )
    parser.add_argument(
        "--hip-abs-max",
        type=float,
        default=HIP_ABS_MAX,
        help="Max absolute hip angle (degrees) when --hip-qc is enabled.",
    )
    parser.add_argument(
        "--hip-fix-continuity",
        action="store_true",
        help="Apply unwrap+recenter to hip before QC and plotting.",
    )
    args = parser.parse_args()

    plot_population_all_trials(
        Path(args.processed_root),
        Path(args.out_path),
        summary_path=Path(args.summary_path),
        hip_qc=args.hip_qc,
        hip_step_max=args.hip_max_step,
        hip_abs_max=args.hip_abs_max,
        hip_fix_continuity=args.hip_fix_continuity,
    )


if __name__ == "__main__":
    main()
