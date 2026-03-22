import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")


def _load_norm_curves(norm_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(norm_csv)
    required = {"pct", "RKneeAngles_x", "RAnkleAngles_x", "RHipAngles_x"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{norm_csv.name} missing columns: {sorted(missing)}")
    pct = pd.to_numeric(df["pct"], errors="coerce").to_numpy(dtype=float)
    knee = pd.to_numeric(df["RKneeAngles_x"], errors="coerce").to_numpy(dtype=float)
    ankle = pd.to_numeric(df["RAnkleAngles_x"], errors="coerce").to_numpy(dtype=float)
    hip = pd.to_numeric(df["RHipAngles_x"], errors="coerce").to_numpy(dtype=float)
    return pct, hip, knee, ankle


def _collect_norm_files(norm_root: Path) -> list[Path]:
    return sorted(norm_root.glob("AB*/*_gaitCycle_norm101.csv"))


def _plot_population(
    pct: np.ndarray,
    hip_stack: np.ndarray,
    knee_stack: np.ndarray,
    ankle_stack: np.ndarray,
    out_png: Path,
    max_curves_draw: int,
) -> None:
    hip_mean = np.nanmean(hip_stack, axis=0)
    knee_mean = np.nanmean(knee_stack, axis=0)
    ankle_mean = np.nanmean(ankle_stack, axis=0)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True, constrained_layout=True)
    fig.patch.set_facecolor("#ebebeb")
    for ax in axes:
        ax.set_facecolor("#ebebeb")
        ax.grid(alpha=0.18, color="#bcbcbc")
        ax.axhline(0.0, color="#7a7a7a", linewidth=0.8, alpha=0.55)

    def _sample_idx(n: int, max_n: int) -> np.ndarray:
        if n <= max_n:
            return np.arange(n, dtype=int)
        return np.linspace(0, n - 1, max_n, dtype=int)

    idx = _sample_idx(ankle_stack.shape[0], max(1, int(max_curves_draw)))

    for curve in ankle_stack[idx]:
        axes[0].plot(pct, curve, color="gray", alpha=0.22, linewidth=0.9)
    axes[0].plot(pct, ankle_mean, color="black", linewidth=2.4)
    axes[0].set_title("ANKLE", fontsize=16)
    axes[0].set_ylabel("ANGLE (DEGREES)")

    for curve in knee_stack[idx]:
        axes[1].plot(pct, curve, color="gray", alpha=0.22, linewidth=0.9)
    axes[1].plot(pct, knee_mean, color="black", linewidth=2.4)
    axes[1].set_title("KNEE", fontsize=16)
    axes[1].set_ylabel("ANGLE (DEGREES)")

    for curve in hip_stack[idx]:
        axes[2].plot(pct, curve, color="gray", alpha=0.22, linewidth=0.9)
    axes[2].plot(pct, hip_mean, color="black", linewidth=2.4)
    axes[2].set_title("HIP (N/A in source, filled with 0)", fontsize=16)
    axes[2].set_ylabel("ANGLE (DEGREES)")
    axes[2].set_xlabel("PERCENT OF GAIT CYCLE (%)")
    axes[2].set_xlim(0.0, 100.0)

    fig.suptitle(f"benchmark bilateral lower limb - population profile (n={ankle_stack.shape[0]} trials)", fontsize=14)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create one population profile plot across all normalized gait-cycle trials."
    )
    parser.add_argument(
        "--norm-root",
        default="data/benchmark_datasets_for_bilateral_lower_limb/plots",
        help="Root folder containing AB*/ *_gaitCycle_norm101.csv files.",
    )
    parser.add_argument(
        "--out-root",
        default="data/benchmark_datasets_for_bilateral_lower_limb/plots",
        help="Output folder for global profile files.",
    )
    parser.add_argument(
        "--name",
        default="all_trials",
        help="Suffix for output filenames.",
    )
    parser.add_argument(
        "--max-curves-draw",
        type=int,
        default=1000,
        help="Max individual curves to draw as background.",
    )
    args = parser.parse_args()

    norm_root = Path(args.norm_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    files = _collect_norm_files(norm_root)
    if not files:
        raise FileNotFoundError(f"No *_gaitCycle_norm101.csv files found under {norm_root}")

    pct_ref = None
    hips, knees, ankles = [], [], []
    used_files = []
    skipped = []

    for f in files:
        try:
            pct, hip, knee, ankle = _load_norm_curves(f)
            if pct_ref is None:
                pct_ref = pct
            elif pct.shape != pct_ref.shape or not np.allclose(pct, pct_ref):
                skipped.append({"file": str(f), "reason": "pct_mismatch"})
                continue
            if not (np.all(np.isfinite(hip)) and np.all(np.isfinite(knee)) and np.all(np.isfinite(ankle))):
                skipped.append({"file": str(f), "reason": "non_finite_values"})
                continue
            hips.append(hip)
            knees.append(knee)
            ankles.append(ankle)
            used_files.append(str(f))
        except Exception as exc:  # noqa: BLE001
            skipped.append({"file": str(f), "reason": str(exc)})

    if not hips:
        raise RuntimeError("No valid normalized trials available for population profile.")

    hip_stack = np.vstack(hips)
    knee_stack = np.vstack(knees)
    ankle_stack = np.vstack(ankles)

    out_png = out_root / f"population_profile_{args.name}.png"
    _plot_population(
        pct=pct_ref,
        hip_stack=hip_stack,
        knee_stack=knee_stack,
        ankle_stack=ankle_stack,
        out_png=out_png,
        max_curves_draw=int(args.max_curves_draw),
    )

    out_csv = out_root / f"population_profile_{args.name}.csv"
    summary_df = pd.DataFrame(
        {
            "pct": pct_ref,
            "hip_mean": np.nanmean(hip_stack, axis=0),
            "hip_std": np.nanstd(hip_stack, axis=0),
            "knee_mean": np.nanmean(knee_stack, axis=0),
            "knee_std": np.nanstd(knee_stack, axis=0),
            "ankle_mean": np.nanmean(ankle_stack, axis=0),
            "ankle_std": np.nanstd(ankle_stack, axis=0),
        }
    )
    summary_df.to_csv(out_csv, index=False)

    out_yaml = out_root / f"population_profile_{args.name}.yaml"
    payload = {
        "trials_found": int(len(files)),
        "trials_used": int(len(used_files)),
        "trials_skipped": int(len(skipped)),
        "output_png": str(out_png),
        "output_csv": str(out_csv),
    }
    out_yaml.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")

    if skipped:
        skipped_csv = out_root / f"population_profile_{args.name}_skipped.csv"
        pd.DataFrame(skipped).to_csv(skipped_csv, index=False)
        print(skipped_csv)

    print(out_png)
    print(out_csv)
    print(out_yaml)


if __name__ == "__main__":
    main()
