import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def _as_float_list(value: object) -> list[float]:
    if value is None:
        return []
    if isinstance(value, list):
        return [float(v) for v in value]
    return [float(value)]


def _closest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


def _extract_cycle(
    trajectories_path: Path,
    events_path: Path,
    marker_column: str,
    side: str,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    df = pd.read_csv(trajectories_path)
    if "time" not in df.columns:
        raise ValueError(f"Missing 'time' column in {trajectories_path}")
    if marker_column not in df.columns:
        raise ValueError(f"Missing marker column '{marker_column}' in {trajectories_path}")

    events = yaml.safe_load(events_path.read_text()) or {}
    side_prefix = "l" if side.lower() == "left" else "r"
    hs1_key = f"{side_prefix}_heel_strike1"
    hs2_key = f"{side_prefix}_heel_strike2"
    toe_key = f"{side_prefix}_toe_off"

    hs1_values = _as_float_list(events.get(hs1_key))
    hs2_values = _as_float_list(events.get(hs2_key))
    if not hs1_values or not hs2_values:
        raise ValueError(f"Missing heel-strike events in {events_path}")

    hs1 = hs1_values[0]
    hs2 = hs2_values[0]
    if hs2 <= hs1:
        raise ValueError(f"Invalid heel-strike order in {events_path}: {hs1} -> {hs2}")

    toe_candidates = [t for t in _as_float_list(events.get(toe_key)) if hs1 <= t <= hs2]
    toe_off = toe_candidates[0] if toe_candidates else None

    time = pd.to_numeric(df["time"], errors="coerce").to_numpy()
    signal = pd.to_numeric(df[marker_column], errors="coerce").to_numpy()
    valid = np.isfinite(time) & np.isfinite(signal)
    time = time[valid]
    signal = signal[valid]

    start_idx = _closest_index(time, hs1)
    end_idx = _closest_index(time, hs2)
    if end_idx <= start_idx:
        raise ValueError(f"Invalid cycle frame range in {trajectories_path}")

    signal_cycle = signal[start_idx : end_idx + 1]
    gait_percent = np.linspace(0.0, 100.0, len(signal_cycle))
    toe_percent = None if toe_off is None else 100.0 * (toe_off - hs1) / (hs2 - hs1)
    return gait_percent, signal_cycle, toe_percent


def _available_trials(subject_dir: Path, condition: str) -> list[tuple[str, Path, Path]]:
    trials: list[tuple[str, Path, Path]] = []
    for trajectories_path in sorted(subject_dir.glob(f"*_{condition}_*_Trajectories.csv")):
        trial_id = trajectories_path.name.replace("_Trajectories.csv", "")
        events_path = subject_dir / f"{trial_id}_point_gaitEvents.yaml"
        if not events_path.exists():
            continue
        run = trial_id.split("_")[-1]
        if not run.isdigit():
            continue
        trials.append((run, trajectories_path, events_path))
    return sorted(trials, key=lambda item: int(item[0]))


def plot_gait_cycle(
    eurobench_root: Path,
    subject: str,
    condition: str,
    marker_column: str,
    side: str,
    n_trials: int,
    out_path: Path,
    runs: list[str] | None = None,
) -> Path:
    subject_dir = eurobench_root / subject
    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject folder not found: {subject_dir}")

    trials = _available_trials(subject_dir, condition)
    if runs:
        wanted = {run.zfill(2) for run in runs}
        trials = [trial for trial in trials if trial[0].zfill(2) in wanted]

    if len(trials) < n_trials:
        raise ValueError(
            f"Need at least {n_trials} trials for subject {subject} condition {condition}, found {len(trials)}."
        )

    selected = trials[:n_trials]
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)

    for run, trajectories_path, events_path in selected:
        gait_percent, signal_cycle, toe_percent = _extract_cycle(
            trajectories_path=trajectories_path,
            events_path=events_path,
            marker_column=marker_column,
            side=side,
        )
        label = f"{condition}_{run.zfill(2)}"
        ax.plot(gait_percent, signal_cycle, linewidth=1.3, label=label)
        if toe_percent is not None:
            idx = int(np.argmin(np.abs(gait_percent - toe_percent)))
            ax.scatter([gait_percent[idx]], [signal_cycle[idx]], s=18)

    ax.set_title(
        f"Multimodal walking speeds - gait cycle ({subject}, {condition}, {side}, {marker_column})"
    )
    ax.set_xlabel("Gait cycle [%]")
    ax.set_ylabel(f"{marker_column} [m]")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot one gait cycle (HS1 to HS2) for one subject using more than one trial."
    )
    parser.add_argument(
        "--eurobench-root",
        default="data/multimodal_walking_speeds/eurobench",
        help="Root with Eurobench converted files.",
    )
    parser.add_argument("--subject", default="2014002", help="Subject ID (e.g., 2014002).")
    parser.add_argument("--condition", default="C1", help="Condition code (e.g., C1).")
    parser.add_argument(
        "--side",
        default="left",
        choices=["left", "right"],
        help="Side used to define the cycle events.",
    )
    parser.add_argument(
        "--marker-column",
        default=None,
        help="Marker coordinate column from *_Trajectories.csv (e.g., L_FCC_z).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=2,
        help="Number of trials to overlay.",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Optional list of runs to use (e.g., 01 02 03).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output PNG path.",
    )
    args = parser.parse_args()

    marker_column = args.marker_column
    if marker_column is None:
        marker_column = "L_FCC_z" if args.side == "left" else "R_FCC_z"

    if args.out:
        out_path = Path(args.out)
    else:
        out_name = f"gait_cycle_{args.subject}_{args.condition}_{args.side}_{marker_column}.png"
        out_path = Path("data/multimodal_walking_speeds/plots") / out_name

    out = plot_gait_cycle(
        eurobench_root=Path(args.eurobench_root),
        subject=args.subject,
        condition=args.condition,
        marker_column=marker_column,
        side=args.side,
        n_trials=args.n_trials,
        out_path=out_path,
        runs=args.runs,
    )
    print(out)


if __name__ == "__main__":
    main()
