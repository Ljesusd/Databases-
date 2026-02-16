import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def _extract_curve_from_band(
    image_path: Path,
    band: tuple[int, int, int, int],
    threshold: int = 120,
    smooth: int = 5,
):
    img = Image.open(image_path).convert("L")
    arr = np.array(img)
    x0, y0, x1, y1 = band
    sub = arr[y0:y1, x0:x1]
    h, w = sub.shape

    ys = []
    prev = h // 2
    for x in range(w):
        col = sub[:, x]
        candidates = np.where(col < threshold)[0]
        if candidates.size == 0:
            y = prev
        else:
            # Keep continuity while preferring darker pixels.
            costs = np.abs(candidates - prev) * 2.0 + (col[candidates] / 255.0)
            y = candidates[np.argmin(costs)]
        ys.append(float(y))
        prev = y

    ys = np.array(ys, dtype=float)
    if smooth > 1:
        kernel = np.ones(smooth, dtype=float) / smooth
        ys = np.convolve(ys, kernel, mode="same")

    ys = (h - 1) - ys
    x_old = np.linspace(0, 100, len(ys))
    x_new = np.linspace(0, 100, 101)
    y_new = np.interp(x_new, x_old, ys)
    return x_new, y_new, (x0, y0, x1, y1)


def _zscore(arr: np.ndarray):
    std = np.std(arr)
    if std == 0:
        return arr - np.mean(arr)
    return (arr - np.mean(arr)) / std


def _load_subject_trials(processed_root: Path):
    files = sorted(processed_root.glob("user*/**/*_marker_angles_norm101.csv"))
    if not files:
        raise FileNotFoundError("No *_marker_angles_norm101.csv found under processed root.")

    per_user: dict[str, list[dict]] = {}
    for path in files:
        df = pd.read_csv(path)
        required = {"pct", "hip_flexion", "knee_flexion", "ankle_dorsiflexion"}
        if not required.issubset(df.columns):
            continue
        user = path.parent.name
        per_user.setdefault(user, []).append(
            {
                "path": path,
                "pct": df["pct"].to_numpy(),
                "hip": df["hip_flexion"].to_numpy(),
                "knee": df["knee_flexion"].to_numpy(),
                "ankle": df["ankle_dorsiflexion"].to_numpy(),
            }
        )
    return per_user


def _compute_distance(curve, template):
    curve_n = _zscore(curve)
    template_n = _zscore(template)
    return float(np.sqrt(np.mean((curve_n - template_n) ** 2)))


def _subject_score(trials, template, mode: str):
    if mode == "mean":
        hip = np.mean([t["hip"] for t in trials], axis=0)
        knee = np.mean([t["knee"] for t in trials], axis=0)
        ankle = np.mean([t["ankle"] for t in trials], axis=0)
        d_hip = _compute_distance(hip, template["hip"])
        d_knee = _compute_distance(knee, template["knee"])
        d_ankle = _compute_distance(ankle, template["ankle"])
        return {
            "hip": d_hip,
            "knee": d_knee,
            "ankle": d_ankle,
            "total": (d_hip + d_knee + d_ankle) / 3.0,
            "best_trial": None,
        }

    best = None
    for trial in trials:
        d_hip = _compute_distance(trial["hip"], template["hip"])
        d_knee = _compute_distance(trial["knee"], template["knee"])
        d_ankle = _compute_distance(trial["ankle"], template["ankle"])
        total = (d_hip + d_knee + d_ankle) / 3.0
        if best is None or total < best["total"]:
            best = {
                "hip": d_hip,
                "knee": d_knee,
                "ankle": d_ankle,
                "total": total,
                "best_trial": trial["path"].name,
            }
    return best


def _plot_best_subjects(output_path: Path, template, selected):
    fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    joints = ["ankle", "knee", "hip"]
    titles = ["ANKLE", "KNEE", "HIP"]

    for ax, joint, title in zip(axes, joints, titles):
        for subject in selected:
            ax.plot(subject["pct"], subject[joint], color="gray", alpha=0.25, linewidth=0.7)
        ax.plot(template["pct"], template[joint], color="black", linewidth=1.5)
        ax.set_title(title)
        ax.set_ylabel("")
        ax.set_yticks([])
        ax.set_xlim(0, 100)

    axes[-1].set_xlabel("PERCENT OF GAIT CYCLE (%)")
    fig.suptitle(f"Closest subjects to literature (n={len(selected)})")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_extraction_overlay(image_path: Path, curves, output_path: Path):
    img = Image.open(image_path)
    fig, ax = plt.subplots(figsize=(4, 6))
    ax.imshow(img)
    for name, data in curves.items():
        x, y, band = data
        x0, y0, x1, y1 = band
        # Map curve back to image coordinates
        x_img = x0 + (x / 100.0) * (x1 - x0)
        y_img = y0 + ((y1 - y0 - 1) - y)
        ax.plot(x_img, y_img, linewidth=1.2, label=name)
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=6)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-root",
        default="data/multisensor_gait/processed",
        help="Root with per-trial marker angles",
    )
    parser.add_argument(
        "--literature-image",
        default="literature graph/giat cycle.jpg",
        help="Path to literature image with hip/knee/ankle curves",
    )
    parser.add_argument(
        "--mode",
        choices=["mean", "best"],
        default="mean",
        help="Use mean of trials per subject or best-matching trial",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of closest subjects to plot",
    )
    parser.add_argument(
        "--summary-path",
        default="data/multisensor_gait/test/analysis/literature_match_scores.csv",
        help="Output CSV with distances",
    )
    parser.add_argument(
        "--plot-path",
        default="data/multisensor_gait/test/plots/literature_closest_subjects.png",
        help="Output plot path",
    )
    parser.add_argument(
        "--overlay-path",
        default="data/multisensor_gait/test/plots/literature_extraction_overlay.png",
        help="Debug overlay plot path",
    )
    args = parser.parse_args()

    image_path = Path(args.literature_image)

    # Manual bands for hip/knee/ankle within the literature image.
    # Format: (x0, y0, x1, y1)
    bands = {
        "hip": (40, 90, 305, 200),
        "knee": (40, 210, 305, 320),
        "ankle": (40, 330, 305, 440),
    }

    curves = {}
    template = {"pct": np.linspace(0, 100, 101)}
    for name, band in bands.items():
        x, y, band = _extract_curve_from_band(image_path, band)
        curves[name] = (x, y, band)
        template[name] = y

    overlay_path = Path(args.overlay_path)
    _plot_extraction_overlay(image_path, curves, overlay_path)

    per_user = _load_subject_trials(Path(args.processed_root))
    rows = []
    selected_curves = []

    for user, trials in per_user.items():
        if not trials:
            continue
        score = _subject_score(trials, template, args.mode)
        rows.append(
            {
                "user": user,
                "n_trials": len(trials),
                "distance_total": score["total"],
                "distance_ankle": score["ankle"],
                "distance_knee": score["knee"],
                "distance_hip": score["hip"],
                "best_trial": score["best_trial"],
            }
        )

    df = pd.DataFrame(rows).sort_values("distance_total")
    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_path, index=False)

    top = df.head(args.top_n)
    for _, row in top.iterrows():
        user = row["user"]
        trials = per_user[user]
        if args.mode == "best" and row["best_trial"]:
            trial = next(t for t in trials if t["path"].name == row["best_trial"])
            selected_curves.append(
                {
                    "pct": trial["pct"],
                    "hip": trial["hip"],
                    "knee": trial["knee"],
                    "ankle": trial["ankle"],
                }
            )
        else:
            selected_curves.append(
                {
                    "pct": trials[0]["pct"],
                    "hip": np.mean([t["hip"] for t in trials], axis=0),
                    "knee": np.mean([t["knee"] for t in trials], axis=0),
                    "ankle": np.mean([t["ankle"] for t in trials], axis=0),
                }
            )

    plot_path = Path(args.plot_path)
    _plot_best_subjects(plot_path, template, selected_curves)


if __name__ == "__main__":
    main()
