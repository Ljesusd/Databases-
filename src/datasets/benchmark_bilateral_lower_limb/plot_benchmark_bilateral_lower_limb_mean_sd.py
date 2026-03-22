#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
from typing import Dict, Iterable, List
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


DATASET_ROOT = Path("data/benchmark_datasets_for_bilateral_lower_limb")
EUROBENCH_ROOT = DATASET_ROOT / "eurobench"
OUTPUT_DIR = DATASET_ROOT / "plots" / "level_walking"
OUTPUT_PNG = OUTPUT_DIR / "population_profile_level_walking_mean_sd.png"
OUTPUT_CSV = OUTPUT_DIR / "population_profile_level_walking_mean_sd_cycles.csv"
OUTPUT_YAML = OUTPUT_DIR / "population_profile_level_walking_mean_sd.yaml"

LEVEL_WALKING_MODE = 1
PURITY_THRESHOLD = 0.95
MIN_DURATION_S = 0.6
MAX_DURATION_S = 1.8
PCT_GRID = np.linspace(0.0, 100.0, 101)

SIDE_LABEL = {"l": "Left", "r": "Right"}
HIP_TITLE = "Hip (N/A in source)"
JOINT_ORDER = ("hip", "knee", "ankle")
JOINT_TITLES = {"hip": HIP_TITLE, "knee": "Knee", "ankle": "Ankle"}
JOINT_COLORS = {"hip": "#1f77b4", "knee": "#2ca02c", "ankle": "#d62728"}


@dataclass
class CycleRecord:
    subject: str
    trial: str
    side: str
    cycle_index: int
    start_s: float
    end_s: float
    duration_s: float
    purity: float
    hip: np.ndarray
    knee: np.ndarray
    ankle: np.ndarray


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_source_csv(path_str: str) -> pd.DataFrame:
    if "::" in path_str:
        zip_path_str, member_name = path_str.split("::", 1)
        with zipfile.ZipFile(Path(zip_path_str)) as zf:
            with zf.open(member_name, "r") as handle:
                return pd.read_csv(io.TextIOWrapper(handle, encoding="utf-8"), usecols=["Mode"])
    return pd.read_csv(Path(path_str), usecols=["Mode"])


def build_trial_prefix(joint_angles_path: Path) -> str:
    return joint_angles_path.name.replace("_jointAngles.csv", "")


def find_trial_files() -> Iterable[tuple[Path, Path, Path]]:
    for joint_angles_path in sorted(EUROBENCH_ROOT.rglob("*_jointAngles.csv")):
        trial_prefix = build_trial_prefix(joint_angles_path)
        info_path = joint_angles_path.with_name(f"{trial_prefix}_info.yaml")
        gait_events_path = joint_angles_path.with_name(f"{trial_prefix}_gaitEvents.yaml")
        if info_path.exists() and gait_events_path.exists():
            yield joint_angles_path, gait_events_path, info_path


def signal_for_side(df: pd.DataFrame, side: str, joint: str) -> np.ndarray:
    side_prefix = "L" if side == "l" else "R"
    if joint == "hip":
        return np.zeros(len(df), dtype=float)
    if joint == "knee":
        raw = df[f"{side_prefix}KneeAngles_x"].to_numpy(dtype=float)
        return -(raw + 90.0)
    if joint == "ankle":
        raw = df[f"{side_prefix}AnkleAngles_x"].to_numpy(dtype=float)
        return -raw
    raise ValueError(f"Unsupported joint: {joint}")


def interpolate_cycle(time_s: np.ndarray, values: np.ndarray, start_s: float, end_s: float) -> np.ndarray | None:
    start_idx = int(np.searchsorted(time_s, start_s, side="left"))
    end_idx = int(np.searchsorted(time_s, end_s, side="right"))
    if end_idx - start_idx < 3:
        return None
    seg_time = time_s[start_idx:end_idx]
    seg_values = values[start_idx:end_idx]
    if seg_time.size < 3:
        return None
    rel_pct = (seg_time - start_s) / max(end_s - start_s, 1e-9) * 100.0
    rel_pct[0] = 0.0
    rel_pct[-1] = 100.0
    return np.interp(PCT_GRID, rel_pct, seg_values)


def extract_cycles_for_side(
    angles_df: pd.DataFrame,
    mode_series: np.ndarray,
    gait_events: dict,
    subject: str,
    trial: str,
    side: str,
) -> List[CycleRecord]:
    heel_key = f"{side}_heel_strike"
    heel_strikes = sorted(float(v) for v in gait_events.get(heel_key, []))
    if len(heel_strikes) < 2:
        return []

    time_s = angles_df["time"].to_numpy(dtype=float)
    hip_signal = signal_for_side(angles_df, side, "hip")
    knee_signal = signal_for_side(angles_df, side, "knee")
    ankle_signal = signal_for_side(angles_df, side, "ankle")

    records: List[CycleRecord] = []
    for cycle_index, (start_s, end_s) in enumerate(zip(heel_strikes[:-1], heel_strikes[1:]), start=1):
        duration_s = end_s - start_s
        if duration_s < MIN_DURATION_S or duration_s > MAX_DURATION_S:
            continue

        start_idx = max(0, int(round(start_s * 1000.0)))
        end_idx = min(len(mode_series), int(round(end_s * 1000.0)))
        if end_idx - start_idx < 3:
            continue
        mode_window = mode_series[start_idx:end_idx]
        purity = float(np.mean(mode_window == LEVEL_WALKING_MODE))
        if purity < PURITY_THRESHOLD:
            continue

        hip = interpolate_cycle(time_s, hip_signal, start_s, end_s)
        knee = interpolate_cycle(time_s, knee_signal, start_s, end_s)
        ankle = interpolate_cycle(time_s, ankle_signal, start_s, end_s)
        if hip is None or knee is None or ankle is None:
            continue

        records.append(
            CycleRecord(
                subject=subject,
                trial=trial,
                side=side,
                cycle_index=cycle_index,
                start_s=start_s,
                end_s=end_s,
                duration_s=duration_s,
                purity=purity,
                hip=hip,
                knee=knee,
                ankle=ankle,
            )
        )
    return records


def extract_all_cycles() -> List[CycleRecord]:
    all_records: List[CycleRecord] = []
    angle_usecols = [
        "time",
        "LKneeAngles_x",
        "LAnkleAngles_x",
        "RKneeAngles_x",
        "RAnkleAngles_x",
    ]
    for joint_angles_path, gait_events_path, info_path in find_trial_files():
        trial_prefix = build_trial_prefix(joint_angles_path)
        subject = trial_prefix.split("_")[0]
        info = load_yaml(info_path)
        gait_events = load_yaml(gait_events_path)
        angles_df = pd.read_csv(joint_angles_path, usecols=angle_usecols)
        source_df = read_source_csv(str(info["source_file"]))
        mode_series = source_df["Mode"].to_numpy(dtype=int)
        all_records.extend(extract_cycles_for_side(angles_df, mode_series, gait_events, subject, trial_prefix, "l"))
        all_records.extend(extract_cycles_for_side(angles_df, mode_series, gait_events, subject, trial_prefix, "r"))
    return all_records


def records_to_long_dataframe(records: List[CycleRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        for idx, pct in enumerate(PCT_GRID):
            rows.append(
                {
                    "subject": record.subject,
                    "trial": record.trial,
                    "side": record.side,
                    "cycle_index": record.cycle_index,
                    "start_s": record.start_s,
                    "end_s": record.end_s,
                    "duration_s": record.duration_s,
                    "level_walking_ratio": record.purity,
                    "pct": pct,
                    "hip_deg": record.hip[idx],
                    "knee_deg": record.knee[idx],
                    "ankle_deg": record.ankle[idx],
                }
            )
    return pd.DataFrame(rows)


def stack_joint(records: List[CycleRecord], joint: str) -> np.ndarray:
    return np.vstack([getattr(record, joint) for record in records])


def plot_joint_cycles(records: List[CycleRecord]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    side_records: Dict[str, List[CycleRecord]] = {
        "l": [record for record in records if record.side == "l"],
        "r": [record for record in records if record.side == "r"],
    }

    plt.style.use("default")
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)
    fig.patch.set_facecolor("#eeeeee")

    for col, side in enumerate(("l", "r")):
        n_cycles = len(side_records[side])
        axes[0, col].set_title(f"{SIDE_LABEL[side]} (n={n_cycles} cycles)", fontsize=15)
        for row, joint in enumerate(JOINT_ORDER):
            ax = axes[row, col]
            ax.set_facecolor("#f2f2f2")
            ax.grid(True, color="#c7c7c7", alpha=0.5, linewidth=0.8)
            ax.axhline(0.0, color="#8d99a5", linewidth=1.0, alpha=0.8)
            ax.set_xlim(0.0, 100.0)
            ax.set_ylabel(f"{JOINT_TITLES[joint]} [deg]", fontsize=12)

            side_joint_records = side_records[side]
            if not side_joint_records:
                continue

            for record in side_joint_records:
                ax.plot(PCT_GRID, getattr(record, joint), color="#9b9b9b", alpha=0.12, linewidth=1.0)

            curves = stack_joint(side_joint_records, joint)
            mean_curve = curves.mean(axis=0)
            sd_curve = curves.std(axis=0, ddof=0)
            color = JOINT_COLORS[joint]
            ax.fill_between(PCT_GRID, mean_curve - sd_curve, mean_curve + sd_curve, color=color, alpha=0.18)
            ax.plot(PCT_GRID, mean_curve, color=color, linewidth=2.6)

            if joint == "hip":
                ax.text(
                    0.98,
                    0.92,
                    "N/A in source",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=10,
                    color="#666666",
                )

    axes[-1, 0].set_xlabel("Gait cycle [%]", fontsize=13)
    axes[-1, 1].set_xlabel("Gait cycle [%]", fontsize=13)
    fig.suptitle("Benchmark bilateral lower limb: level walking joint cycles (mean ± SD)", fontsize=18, y=0.995)
    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_summary(records: List[CycleRecord]) -> None:
    long_df = records_to_long_dataframe(records)
    long_df.to_csv(OUTPUT_CSV, index=False)

    summary = {
        "dataset": "benchmark_datasets_for_bilateral_lower_limb",
        "plot_type": "level_walking_joint_cycles_mean_sd",
        "mode_code": LEVEL_WALKING_MODE,
        "mode_label": "level_walking",
        "purity_threshold": PURITY_THRESHOLD,
        "min_duration_s": MIN_DURATION_S,
        "max_duration_s": MAX_DURATION_S,
        "n_cycles_total": int(len(records)),
        "n_cycles_left": int(sum(record.side == "l" for record in records)),
        "n_cycles_right": int(sum(record.side == "r" for record in records)),
        "output_png": str(OUTPUT_PNG),
        "output_csv": str(OUTPUT_CSV),
    }
    with OUTPUT_YAML.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)


def main() -> None:
    records = extract_all_cycles()
    if not records:
        raise SystemExit("No level-walking cycles were found.")
    plot_joint_cycles(records)
    write_summary(records)


if __name__ == "__main__":
    main()
