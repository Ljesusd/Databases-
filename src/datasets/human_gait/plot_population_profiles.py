import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _hip_curve_is_valid(curve: np.ndarray, hip_abs_max: float, hip_max_step: float) -> bool:
    if curve.size < 2:
        return False
    if not np.all(np.isfinite(curve)):
        return False
    abs_max = float(np.max(np.abs(curve)))
    max_step = float(np.max(np.abs(np.diff(curve))))
    return abs_max <= hip_abs_max and max_step <= hip_max_step


def _flip_if_needed(curve: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, bool]:
    a = curve - curve.mean()
    b = reference - reference.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return curve, False
    corr = float(np.dot(a, b) / denom)
    if not np.isfinite(corr):
        return curve, False
    if corr < 0:
        return -curve, True
    return curve, False


def _load_angles(csv_path: Path):
    df = pd.read_csv(csv_path)
    required = {"pct", "hip_flexion", "knee_flexion", "ankle_dorsiflexion"}
    if not required.issubset(df.columns):
        return None
    return df


def _ideal_hip_reference(n_points: int) -> np.ndarray:
    # Starts positive, goes negative mid-stance, returns positive.
    return np.cos(np.linspace(0.0, 2.0 * np.pi, n_points))


def plot_population(
    processed_root: Path,
    out_path: Path,
    task: str,
    fix_hip_sign: bool = True,
    hip_reference: str = "ideal",
    hip_qc: bool = False,
    hip_abs_max: float = 120.0,
    hip_max_step: float = 25.0,
):
    files = list(processed_root.rglob("*_marker_angles_norm101.csv"))
    if not files:
        raise FileNotFoundError(f"No marker angles found in {processed_root}")

    hips, knees, ankles = [], [], []
    pct_ref = None
    n_total = 0
    n_filtered_hip_qc = 0
    for csv_path in files:
        df = _load_angles(csv_path)
        if df is None:
            continue
        n_total += 1
        pct = df["pct"].values
        if pct_ref is None:
            pct_ref = pct
        elif not np.allclose(pct_ref, pct):
            continue
        hip_curve = df["hip_flexion"].values
        if hip_qc and not _hip_curve_is_valid(hip_curve, hip_abs_max=hip_abs_max, hip_max_step=hip_max_step):
            n_filtered_hip_qc += 1
            continue
        hips.append(hip_curve)
        knees.append(df["knee_flexion"].values)
        ankles.append(df["ankle_dorsiflexion"].values)

    if not hips:
        raise FileNotFoundError("No valid flexion curves found.")

    hips = np.stack(hips)
    knees = np.stack(knees)
    ankles = np.stack(ankles)

    hip_med = np.median(hips, axis=0)
    flipped = 0
    if fix_hip_sign:
        if hip_reference == "ideal":
            reference = _ideal_hip_reference(hip_med.size)
        else:
            reference = hip_med
        corrected = []
        for curve in hips:
            curve_fixed, was_flipped = _flip_if_needed(curve, reference)
            if was_flipped:
                flipped += 1
            corrected.append(curve_fixed)
        hips = np.stack(corrected)
        hip_med = np.median(hips, axis=0)
    knee_med = np.median(knees, axis=0)
    ankle_med = np.median(ankles, axis=0)

    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    panels = [
        ("ANKLE", ankles, ankle_med),
        ("KNEE", knees, knee_med),
        ("HIP", hips, hip_med),
    ]
    for ax, (title, stack, med) in zip(axes, panels):
        for curve in stack:
            ax.plot(pct_ref, curve, color="gray", alpha=0.25, linewidth=0.8)
        ax.plot(pct_ref, med, color="black", linewidth=1.8)
        ax.set_title(title)
        ax.set_xlim(0, 100)
        ax.set_yticks([])

    axes[-1].set_xlabel("PERCENT OF GAIT CYCLE (%)")
    title = f"human_gait {task} (n={hips.shape[0]})"
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(
        f"task={task},total={n_total},kept={hips.shape[0]},"
        f"filtered_hip_qc={n_filtered_hip_qc},hip_qc={hip_qc}"
    )


def plot_population_global(
    processed_root: Path,
    out_path: Path,
    label: str,
    fix_hip_sign: bool = True,
    hip_reference: str = "ideal",
    hip_qc: bool = False,
    hip_abs_max: float = 120.0,
    hip_max_step: float = 25.0,
):
    files = list(processed_root.rglob("*_marker_angles_norm101.csv"))
    if not files:
        raise FileNotFoundError(f"No marker angles found in {processed_root}")

    hips, knees, ankles = [], [], []
    pct_ref = None
    n_total = 0
    n_filtered_hip_qc = 0
    for csv_path in files:
        df = _load_angles(csv_path)
        if df is None:
            continue
        n_total += 1
        pct = df["pct"].values
        if pct_ref is None:
            pct_ref = pct
        elif not np.allclose(pct_ref, pct):
            continue
        hip_curve = df["hip_flexion"].values
        if hip_qc and not _hip_curve_is_valid(hip_curve, hip_abs_max=hip_abs_max, hip_max_step=hip_max_step):
            n_filtered_hip_qc += 1
            continue
        hips.append(hip_curve)
        knees.append(df["knee_flexion"].values)
        ankles.append(df["ankle_dorsiflexion"].values)

    if not hips:
        raise FileNotFoundError("No valid flexion curves found.")

    hips = np.stack(hips)
    knees = np.stack(knees)
    ankles = np.stack(ankles)

    hip_med = np.median(hips, axis=0)
    if fix_hip_sign:
        if hip_reference == "ideal":
            reference = _ideal_hip_reference(hip_med.size)
        else:
            reference = hip_med
        corrected = []
        for curve in hips:
            curve_fixed, _ = _flip_if_needed(curve, reference)
            corrected.append(curve_fixed)
        hips = np.stack(corrected)
        hip_med = np.median(hips, axis=0)

    knee_med = np.median(knees, axis=0)
    ankle_med = np.median(ankles, axis=0)

    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    panels = [
        ("ANKLE", ankles, ankle_med),
        ("KNEE", knees, knee_med),
        ("HIP", hips, hip_med),
    ]
    for ax, (title, stack, med) in zip(axes, panels):
        for curve in stack:
            ax.plot(pct_ref, curve, color="gray", alpha=0.2, linewidth=0.7)
        ax.plot(pct_ref, med, color="black", linewidth=1.8)
        ax.set_title(title)
        ax.set_xlim(0, 100)
        ax.set_yticks([])

    axes[-1].set_xlabel("PERCENT OF GAIT CYCLE (%)")
    fig.suptitle(f"{label} (n={hips.shape[0]})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(
        f"label={label},total={n_total},kept={hips.shape[0]},"
        f"filtered_hip_qc={n_filtered_hip_qc},hip_qc={hip_qc}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-root",
        default="data/human_gait/processed",
        help="Root with processed task folders",
    )
    parser.add_argument(
        "--plots-root",
        default="data/human_gait/plots",
        help="Root folder for plots",
    )
    parser.add_argument(
        "--tasks",
        default="Gait,FastGait,2minWalk",
        help="Comma-separated tasks",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Also create one global population plot across all task folders.",
    )
    parser.add_argument(
        "--all-label",
        default="human_gait all tasks",
        help="Title label for the global population plot.",
    )
    parser.add_argument(
        "--all-out-path",
        default="data/human_gait/plots/population_profiles_all_tasks.png",
        help="Output path for the global population plot.",
    )
    parser.add_argument(
        "--no-fix-hip-sign",
        action="store_true",
        help="Disable hip sign correction (default: apply correction).",
    )
    parser.add_argument(
        "--hip-reference",
        choices=["ideal", "median"],
        default="ideal",
        help="Reference used to decide hip sign flips.",
    )
    parser.add_argument(
        "--hip-qc",
        action="store_true",
        help="Filter out trials with hip discontinuities/extreme amplitudes.",
    )
    parser.add_argument(
        "--hip-abs-max",
        type=float,
        default=120.0,
        help="Maximum absolute hip angle allowed when --hip-qc is enabled.",
    )
    parser.add_argument(
        "--hip-max-step",
        type=float,
        default=25.0,
        help="Maximum point-to-point hip jump allowed when --hip-qc is enabled.",
    )
    args = parser.parse_args()

    processed_root = Path(args.processed_root)
    plots_root = Path(args.plots_root)

    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        task_root = processed_root / task
        out_path = plots_root / task / "population_profiles_flexion.png"
        plot_population(
            task_root,
            out_path,
            task,
            fix_hip_sign=not args.no_fix_hip_sign,
            hip_reference=args.hip_reference,
            hip_qc=args.hip_qc,
            hip_abs_max=args.hip_abs_max,
            hip_max_step=args.hip_max_step,
        )
        print(out_path)

    if args.include_all:
        plot_population_global(
            processed_root=processed_root,
            out_path=Path(args.all_out_path),
            label=args.all_label,
            fix_hip_sign=not args.no_fix_hip_sign,
            hip_reference=args.hip_reference,
            hip_qc=args.hip_qc,
            hip_abs_max=args.hip_abs_max,
            hip_max_step=args.hip_max_step,
        )
        print(Path(args.all_out_path))


if __name__ == "__main__":
    main()
