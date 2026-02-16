from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_fusion_knee(csv_path: str, out_path: str, joint: str = "knee"):
    df = pd.read_csv(csv_path)
    pct = df["pct"].values
    col_map = {
        "hip": ("hip_flexion_mean", "hip_flexion_std"),
        "knee": ("knee_flexion_mean", "knee_flexion_std"),
        "ankle": ("ankle_dorsiflexion_mean", "ankle_dorsiflexion_std"),
    }
    title_map = {
        "hip": "Hip angle during gait (fusion)",
        "knee": "Knee angle during gait (fusion)",
        "ankle": "Ankle angle during gait (fusion)",
    }
    if joint not in col_map:
        raise ValueError("joint must be one of: hip, knee, ankle")
    mean_col, std_col = col_map[joint]
    mean = df[mean_col].values
    std = df[std_col].values

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pct, mean, color="black", linewidth=1.5)
    ax.fill_between(pct, mean - std, mean + std, color="gray", alpha=0.25)
    ax.set_title(title_map[joint])
    ax.set_xlabel("Percentage of gait cycle (%)")
    ax.set_ylabel("ANGLE (DEGREES)")
    ax.set_xlim(0, 100)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)


def main():
    plot_fusion_knee(
        csv_path="data/138_HealthyPiG/processed/fusion_SUBJ01_04_flexion_norm101.csv",
        out_path="data/138_HealthyPiG/processed/fusion_SUBJ01_04_knee.png",
        joint="knee",
    )


if __name__ == "__main__":
    main()
