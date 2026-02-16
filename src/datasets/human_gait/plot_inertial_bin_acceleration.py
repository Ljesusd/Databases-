import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLOCK_SIZE = 512
GROUP_SIZE = 8
ACC_PACKET_ID = 0x13
SLOTS_PER_BLOCK = 20


def _count_acc_groups(block: bytes, packet_id: int = ACC_PACKET_ID) -> int:
    count = 0
    for i in range(8, BLOCK_SIZE, GROUP_SIZE):
        if block[i] == packet_id:
            count += 1
    return count


def _find_data_start_block(data: bytes, block_size: int = BLOCK_SIZE) -> int:
    n_blocks = len(data) // block_size
    for bi in range(n_blocks):
        off = bi * block_size
        block = data[off : off + block_size]
        # Data blocks in this format usually carry ~20 acceleration groups.
        if _count_acc_groups(block) >= 10:
            return bi
    raise ValueError("No data blocks with acceleration packets were found in BIN file.")


def _parse_bin_acceleration(
    in_path: Path,
    block_size: int = BLOCK_SIZE,
    slots_per_block: int = SLOTS_PER_BLOCK,
    packet_id: int = ACC_PACKET_ID,
) -> pd.DataFrame:
    data = in_path.read_bytes()
    n_full_blocks = len(data) // block_size
    if n_full_blocks < 2:
        raise ValueError("BIN file is too short.")

    start_block = _find_data_start_block(data, block_size=block_size)
    n_blocks = n_full_blocks - start_block
    if n_blocks <= 0:
        raise ValueError("No full data blocks available.")

    ticks = np.empty(n_blocks, dtype=np.float64)
    for bi in range(n_blocks):
        off = (start_block + bi) * block_size
        block = data[off : off + block_size]
        ticks[bi] = int.from_bytes(block[4:8], "big", signed=False)

    if n_blocks > 1:
        dt_blocks = np.diff(ticks)
        valid = dt_blocks[dt_blocks > 0]
        fallback_dt = float(np.median(valid)) if valid.size else 40.0
        dt_blocks = np.append(dt_blocks, fallback_dt)
        dt_blocks[dt_blocks <= 0] = fallback_dt
    else:
        dt_blocks = np.array([40.0], dtype=np.float64)

    cap = n_blocks * slots_per_block
    t_ms = np.empty(cap, dtype=np.float64)
    acc_x = np.empty(cap, dtype=np.float64)
    acc_y = np.empty(cap, dtype=np.float64)
    acc_z = np.empty(cap, dtype=np.float64)
    n = 0

    for bi in range(n_blocks):
        off = (start_block + bi) * block_size
        block = data[off : off + block_size]
        base_t = ticks[bi]
        dt = dt_blocks[bi]

        block_samples: list[tuple[int, int, int, int]] = []
        for i in range(8, block_size, GROUP_SIZE):
            g = block[i : i + GROUP_SIZE]
            if g[0] != packet_id:
                continue
            slot = int(g[1]) // 2
            if slot < 0 or slot >= slots_per_block:
                continue
            x = int.from_bytes(g[2:4], "big", signed=True)
            y = int.from_bytes(g[4:6], "big", signed=True)
            z = int.from_bytes(g[6:8], "big", signed=True)
            block_samples.append((slot, x, y, z))

        if not block_samples:
            continue

        block_samples.sort(key=lambda s: s[0])
        for slot, x, y, z in block_samples:
            t_ms[n] = base_t + (slot / slots_per_block) * dt
            acc_x[n] = x
            acc_y[n] = y
            acc_z[n] = z
            n += 1

    if n == 0:
        raise ValueError("No acceleration samples decoded from BIN file.")

    t_ms = t_ms[:n]
    acc_x = acc_x[:n]
    acc_y = acc_y[:n]
    acc_z = acc_z[:n]

    if np.any(np.diff(t_ms) < 0):
        order = np.argsort(t_ms, kind="stable")
        t_ms = t_ms[order]
        acc_x = acc_x[order]
        acc_y = acc_y[order]
        acc_z = acc_z[order]

    return pd.DataFrame(
        {
            "time_s": t_ms / 1000.0,
            "acc_x_counts": acc_x,
            "acc_y_counts": acc_y,
            "acc_z_counts": acc_z,
        }
    )


def plot_bin_acceleration(
    in_path: Path,
    out_path: Path,
    max_seconds: float | None = None,
    counts_per_g: float = 2048.0,
    raw_counts: bool = False,
    max_plot_points: int = 250_000,
    export_csv: Path | None = None,
) -> Path:
    df = _parse_bin_acceleration(in_path)

    if max_seconds is not None:
        df = df[df["time_s"] <= max_seconds].copy()
    if df.empty:
        raise ValueError("No samples left after applying max_seconds filter.")

    if not raw_counts:
        df["acc_x"] = df["acc_x_counts"] / counts_per_g
        df["acc_y"] = df["acc_y_counts"] / counts_per_g
        df["acc_z"] = df["acc_z_counts"] / counts_per_g
        y_label = "Aceleracion [g]"
    else:
        df["acc_x"] = df["acc_x_counts"]
        df["acc_y"] = df["acc_y_counts"]
        df["acc_z"] = df["acc_z_counts"]
        y_label = "Aceleracion [counts]"

    if export_csv is not None:
        export_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(export_csv, index=False)

    n = len(df)
    step = max(1, n // max_plot_points)
    view = df.iloc[::step].copy()

    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    ax.plot(view["time_s"], view["acc_x"], label="acc x", linewidth=0.8)
    ax.plot(view["time_s"], view["acc_y"], label="acc y", linewidth=0.8)
    ax.plot(view["time_s"], view["acc_z"], label="acc z", linewidth=0.8)
    ax.set_title(f"Human Gait - Aceleracion BIN ({in_path.stem})")
    ax.set_xlabel("Tiempo [s]")
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _default_out_from_input(in_path: Path) -> Path:
    name = in_path.stem.replace("_Inertial_sensor", "_inertial_sensor")
    return Path("data/human_gait/plots") / f"{name}_acceleration_from_bin_full.png"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode accelerometer data from *_Inertial_sensor.BIN and plot it."
    )
    parser.add_argument(
        "--input",
        default="data/human_gait/raw/researchdata/P09_S01/RAW_DATA/P09_S01_LF_Inertial_sensor.BIN",
        help="Path to *_Inertial_sensor.BIN file.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path. Defaults to data/human_gait/plots/<name>_acceleration_from_bin_full.png",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Optional time window in seconds (e.g. 60 for a zoom).",
    )
    parser.add_argument(
        "--counts-per-g",
        type=float,
        default=2048.0,
        help="Scale factor to convert raw counts to g (default: 2048).",
    )
    parser.add_argument(
        "--raw-counts",
        action="store_true",
        help="Plot raw accelerometer counts instead of converting to g.",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=250000,
        help="Maximum plotted points after decimation (default: 250000).",
    )
    parser.add_argument(
        "--export-csv",
        default=None,
        help="Optional output CSV path for decoded acceleration.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.out) if args.out else _default_out_from_input(in_path)
    export_csv = Path(args.export_csv) if args.export_csv else None

    out = plot_bin_acceleration(
        in_path=in_path,
        out_path=out_path,
        max_seconds=args.max_seconds,
        counts_per_g=args.counts_per_g,
        raw_counts=args.raw_counts,
        max_plot_points=args.max_plot_points,
        export_csv=export_csv,
    )
    print(out)


if __name__ == "__main__":
    main()
