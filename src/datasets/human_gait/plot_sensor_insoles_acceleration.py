import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _read_sensor_insoles_txt(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        _meta = f.readline()
        header = f.readline().strip()

    if header.startswith("#"):
        header = header[1:].strip()
    columns = header.split("\t")

    df = pd.read_csv(path, sep="\t", skiprows=2, names=columns)
    needed = [
        "time",
        "left acceleration X[g]",
        "left acceleration Y[g]",
        "left acceleration Z[g]",
        "right acceleration X[g]",
        "right acceleration Y[g]",
        "right acceleration Z[g]",
    ]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time"])
    return df


def plot_acceleration(in_path: Path, out_path: Path, max_seconds: float | None = None) -> None:
    df = _read_sensor_insoles_txt(in_path)
    if max_seconds is not None:
        df = df[df["time"] <= max_seconds].copy()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, constrained_layout=True)

    axes[0].plot(df["time"], df["left acceleration X[g]"], label="X", linewidth=0.8)
    axes[0].plot(df["time"], df["left acceleration Y[g]"], label="Y", linewidth=0.8)
    axes[0].plot(df["time"], df["left acceleration Z[g]"], label="Z", linewidth=0.8)
    axes[0].set_title("Human Gait - Aceleracion plantilla izquierda")
    axes[0].set_ylabel("Aceleracion [g]")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(df["time"], df["right acceleration X[g]"], label="X", linewidth=0.8)
    axes[1].plot(df["time"], df["right acceleration Y[g]"], label="Y", linewidth=0.8)
    axes[1].plot(df["time"], df["right acceleration Z[g]"], label="Z", linewidth=0.8)
    axes[1].set_title("Human Gait - Aceleracion plantilla derecha")
    axes[1].set_xlabel("Tiempo [s]")
    axes[1].set_ylabel("Aceleracion [g]")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot left/right insole accelerometer signals from Sensor_insoles.txt."
    )
    parser.add_argument(
        "--input",
        default="data/human_gait/raw/researchdata/P09_S01/RAW_DATA/P09_S01_Sensor_insoles.txt",
        help="Path to *_Sensor_insoles.txt file.",
    )
    parser.add_argument(
        "--out",
        default="data/human_gait/plots/P09_S01_sensor_insoles_acceleration_full.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Optional time window in seconds (e.g. 60 for a zoom).",
    )
    args = parser.parse_args()

    plot_acceleration(Path(args.input), Path(args.out), max_seconds=args.max_seconds)
    print(args.out)


if __name__ == "__main__":
    main()
