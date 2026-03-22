import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_angles(csv_path: Path):
    df = pd.read_csv(csv_path)
    required = {"pct", "hip_flexion", "knee_flexion", "ankle_dorsiflexion"}
    if not required.issubset(df.columns):
        return None
    return df


def _ideal_hip_reference(n_points: int) -> np.ndarray:
    return np.cos(np.linspace(0.0, 2.0 * np.pi, n_points))


def _flip_if_needed(curve: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, bool]:
    a = curve - curve.mean()
    b = reference - reference.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return curve, False
    corr = float(np.dot(a, b) / denom)
    if not np.isfinite(corr):
        return curve, False
    if corr < 0.0:
        return -curve, True
    return curve, False


def plot_population_global(
    processed_root: Path,
    out_path: Path,
    label: str,
    fix_hip_sign: bool = False,
    hip_reference: str = "ideal",
):
    files = sorted(processed_root.rglob("*_canonical_marker_angles_norm101.csv"))
    if not files:
        raise FileNotFoundError(f"No canonical marker angles found in {processed_root}")

    hips, knees, ankles = [], [], []
    pct_ref = None
    rows = []

    for csv_path in files:
        df = _load_angles(csv_path)
        if df is None:
            continue
        pct = df["pct"].values
        if pct_ref is None:
            pct_ref = pct
        elif not np.allclose(pct_ref, pct):
            continue

        hips.append(df["hip_flexion"].values)
        knees.append(df["knee_flexion"].values)
        ankles.append(df["ankle_dorsiflexion"].values)
        rows.append({"subject": csv_path.parent.name, "file": str(csv_path)})

    if not hips:
        raise FileNotFoundError("No usable canonical flexion curves found.")

    hips = np.stack(hips)
    knees = np.stack(knees)
    ankles = np.stack(ankles)

    hip_med = np.median(hips, axis=0)
    flipped = 0
    if fix_hip_sign:
        reference = _ideal_hip_reference(hip_med.size) if hip_reference == "ideal" else hip_med
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
            ax.plot(pct_ref, curve, color="gray", alpha=0.2, linewidth=0.7)
        ax.plot(pct_ref, med, color="black", linewidth=1.8)
        ax.set_title(title)
        ax.set_xlim(0, 100)
        ax.set_yticks([])

    axes[-1].set_xlabel("PERCENT OF GAIT CYCLE (%)")
    fig.suptitle(f"{label} (n={hips.shape[0]}, hip_flipped={flipped})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    summary_path = out_path.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"kept={len(hips)},hip_flipped={flipped}")
    print(out_path)
    print(summary_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-root",
        default="data/multisensor_gait/processed_canonical",
        help="Root with canonical per-subject marker-angle outputs",
    )
    parser.add_argument(
        "--out-path",
        default="data/multisensor_gait/plots_canonical_population/population_profiles_all_subjects.png",
        help="Output plot path",
    )
    parser.add_argument(
        "--label",
        default="multisensor_gait canonical population",
        help="Figure title label",
    )
    parser.add_argument(
        "--fix-hip-sign",
        action="store_true",
        help="Flip hip curves if their correlation with the reference is negative.",
    )
    parser.add_argument(
        "--hip-reference",
        choices=["ideal", "median"],
        default="ideal",
        help="Reference used when --fix-hip-sign is enabled.",
    )
    args = parser.parse_args()

    plot_population_global(
        processed_root=Path(args.processed_root),
        out_path=Path(args.out_path),
        label=args.label,
        fix_hip_sign=args.fix_hip_sign,
        hip_reference=args.hip_reference,
    )


if __name__ == "__main__":
    main()
