import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_subject_flexion(csv_path: Path):
    df = pd.read_csv(csv_path)
    required = {"pct", "hip_flexion", "knee_flexion", "ankle_dorsiflexion"}
    if not required.issubset(df.columns):
        return None
    return {
        "pct": df["pct"].values,
        "hip": df["hip_flexion"].values,
        "knee": df["knee_flexion"].values,
        "ankle": df["ankle_dorsiflexion"].values,
    }


def plot_flattening_subjects(
    processed_root: Path,
    out_path: Path,
    top_n: int = 10,
    file_suffix: str = "_flexion_norm101.csv",
):
    subjects = []
    curves = {}
    pct_ref = None

    for subj_dir in sorted(p for p in processed_root.glob("SUBJ*") if p.is_dir()):
        subj = subj_dir.name
        csv_path = subj_dir / f"{subj}{file_suffix}"
        if not csv_path.exists():
            matches = list(subj_dir.glob(f"*{file_suffix}"))
            if not matches:
                continue
            csv_path = matches[0]
        data = _load_subject_flexion(csv_path)
        if data is None:
            continue
        if pct_ref is None:
            pct_ref = data["pct"]
        elif not np.allclose(pct_ref, data["pct"]):
            continue
        subjects.append(subj)
        curves[subj] = data

    if not subjects:
        raise FileNotFoundError("No flexion curves found.")

    rows = []
    for subj in subjects:
        for lm in ["hip", "knee", "ankle"]:
            vals = curves[subj][lm]
            rng = float(vals.max() - vals.min())
            rows.append({"subject_id": subj, "landmark": lm, "range": rng})
    ranges = pd.DataFrame(rows)

    picks = {}
    for lm in ["hip", "knee", "ankle"]:
        picks[lm] = (
            ranges[ranges["landmark"] == lm]
            .sort_values("range")
            .head(top_n)["subject_id"]
            .tolist()
        )

    # population mean (all subjects)
    mean = {}
    for lm in ["hip", "knee", "ankle"]:
        stack = np.stack([curves[s][lm] for s in subjects], axis=0)
        mean[lm] = stack.mean(axis=0)

    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    panels = [("ANKLE", "ankle"), ("KNEE", "knee"), ("HIP", "hip")]
    for ax, (title, key) in zip(axes, panels):
        for subj in picks[key]:
            ax.plot(pct_ref, curves[subj][key], color="gray", alpha=0.35, linewidth=0.9)
        ax.plot(pct_ref, mean[key], color="black", linewidth=1.8, label="Population mean")
        ax.set_title(f"{title} (lowest range, n={len(picks[key])})")
        ax.set_ylabel("ANGLE (DEGREES)")
        ax.set_xlim(0, 100)
    axes[-1].set_xlabel("PERCENT OF GAIT CYCLE (%)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-root",
        default="data/138_HealthyPiG/processed",
        help="Root with subject flexion CSVs",
    )
    parser.add_argument(
        "--out-path",
        default="data/138_HealthyPiG/test/plots/flattening_subjects_flexion.png",
        help="Output plot path",
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--file-suffix",
        default="_flexion_norm101.csv",
        help="Suffix for per-subject flexion files.",
    )
    args = parser.parse_args()

    plot_flattening_subjects(
        Path(args.processed_root),
        Path(args.out_path),
        top_n=args.top_n,
        file_suffix=args.file_suffix,
    )


if __name__ == "__main__":
    main()
