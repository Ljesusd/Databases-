import argparse
from io import TextIOWrapper
from pathlib import Path
import zipfile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


LEVEL_WALKING_MODE = 1


def _safe_events(path: Path) -> dict[str, np.ndarray]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        return {}
    out: dict[str, np.ndarray] = {}
    for key, values in payload.items():
        if not isinstance(values, list):
            continue
        clean = []
        for value in values:
            try:
                clean.append(float(value))
            except (TypeError, ValueError):
                continue
        if clean:
            out[str(key)] = np.array(sorted(set(clean)), dtype=float)
    return out


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid YAML payload in {path}")
    return payload


def _read_mode_source(source_file: str) -> pd.Series:
    if "::" in source_file:
        zip_path_str, member = source_file.split("::", 1)
        with zipfile.ZipFile(Path(zip_path_str), "r") as zf:
            with zf.open(member, "r") as fh:
                df = pd.read_csv(TextIOWrapper(fh, encoding="utf-8"), usecols=["Mode"])
    else:
        df = pd.read_csv(source_file, usecols=["Mode"])
    return pd.to_numeric(df["Mode"], errors="coerce")


def _trial_triplets(root: Path) -> list[tuple[Path, Path, Path]]:
    triplets: list[tuple[Path, Path, Path]] = []
    for joint_csv in sorted(root.rglob("*_jointAngles.csv")):
        base = joint_csv.name.replace("_jointAngles.csv", "")
        events_yaml = joint_csv.with_name(f"{base}_gaitEvents.yaml")
        info_yaml = joint_csv.with_name(f"{base}_info.yaml")
        if events_yaml.exists() and info_yaml.exists():
            triplets.append((joint_csv, events_yaml, info_yaml))
    return triplets


def _normalize_signal(time: np.ndarray, signal: np.ndarray, start: float, end: float, n_points: int) -> np.ndarray:
    mask = (time >= start) & (time <= end)
    t = time[mask]
    y = signal[mask]
    if t.size < 2:
        raise ValueError("Not enough samples inside cycle window")
    duration = t[-1] - t[0]
    if duration <= 0:
        raise ValueError("Cycle duration must be positive")
    t_norm = (t - t[0]) / duration
    target = np.linspace(0.0, 1.0, n_points)
    return np.interp(target, t_norm, y)


def _extract_level_walking_cycles(
    joint_csv: Path,
    events_yaml: Path,
    info_yaml: Path,
    n_points: int,
    purity_threshold: float,
    min_duration_s: float,
    max_duration_s: float,
) -> tuple[list[dict], list[dict]]:
    info = _load_yaml(info_yaml)
    source_file = str(info.get("source_file") or "").strip()
    if not source_file:
        raise ValueError(f"Missing source_file in {info_yaml.name}")

    df = pd.read_csv(joint_csv)
    required = ["time", "RHipAngles_x", "RKneeAngles_x", "RAnkleAngles_x"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{joint_csv.name} missing columns: {missing}")

    time = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
    hip = pd.to_numeric(df["RHipAngles_x"], errors="coerce").to_numpy(dtype=float)
    knee = pd.to_numeric(df["RKneeAngles_x"], errors="coerce").to_numpy(dtype=float)
    ankle = pd.to_numeric(df["RAnkleAngles_x"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(time) & np.isfinite(hip) & np.isfinite(knee) & np.isfinite(ankle)
    time = time[valid]
    hip = hip[valid]
    knee = knee[valid]
    ankle = ankle[valid]
    if time.size < 3:
        raise ValueError(f"Not enough valid angle samples in {joint_csv.name}")

    mode = _read_mode_source(source_file).to_numpy(dtype=float)
    if mode.size == 0:
        raise ValueError(f"No Mode column values in source for {joint_csv.name}")
    if mode.size != time.size:
        n = min(mode.size, time.size)
        mode = mode[:n]
        time = time[:n]
        hip = hip[:n]
        knee = knee[:n]
        ankle = ankle[:n]

    events = _safe_events(events_yaml)
    rhs = events.get("r_heel_strike", np.array([], dtype=float))
    if rhs.size < 2:
        return [], [{"file": joint_csv.name, "reason": "not_enough_r_heel_strike"}]

    pct = np.linspace(0.0, 100.0, n_points)
    cycles: list[dict] = []
    skipped: list[dict] = []
    for idx in range(rhs.size - 1):
        start = float(rhs[idx])
        end = float(rhs[idx + 1])
        duration = end - start
        if duration < min_duration_s or duration > max_duration_s:
            skipped.append({"file": joint_csv.name, "cycle_index": idx + 1, "reason": "duration_out_of_range"})
            continue

        mask = (time >= start) & (time <= end)
        if np.count_nonzero(mask) < 3:
            skipped.append({"file": joint_csv.name, "cycle_index": idx + 1, "reason": "not_enough_samples"})
            continue

        mode_seg = mode[mask]
        finite_mode = mode_seg[np.isfinite(mode_seg)]
        if finite_mode.size == 0:
            skipped.append({"file": joint_csv.name, "cycle_index": idx + 1, "reason": "mode_missing"})
            continue
        level_ratio = float(np.mean(finite_mode == LEVEL_WALKING_MODE))
        dominant_mode = int(pd.Series(finite_mode).mode().iloc[0])
        if dominant_mode != LEVEL_WALKING_MODE or level_ratio < purity_threshold:
            skipped.append(
                {
                    "file": joint_csv.name,
                    "cycle_index": idx + 1,
                    "reason": "not_level_walking",
                    "dominant_mode": dominant_mode,
                    "level_ratio": level_ratio,
                }
            )
            continue

        cycles.append(
            {
                "subject": joint_csv.parent.name,
                "trial": joint_csv.stem.replace("_jointAngles", ""),
                "cycle_index": idx + 1,
                "start_s": start,
                "end_s": end,
                "duration_s": duration,
                "level_walking_ratio": level_ratio,
                "pct": pct,
                "hip": _normalize_signal(time, hip, start, end, n_points),
                "knee": _normalize_signal(time, knee, start, end, n_points),
                "ankle": _normalize_signal(time, ankle, start, end, n_points),
            }
        )

    return cycles, skipped


def _plot_population(
    pct: np.ndarray,
    hip_stack: np.ndarray,
    knee_stack: np.ndarray,
    ankle_stack: np.ndarray,
    out_png: Path,
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

    for curve in ankle_stack:
        axes[0].plot(pct, curve, color="gray", alpha=0.22, linewidth=0.9)
    axes[0].plot(pct, ankle_mean, color="black", linewidth=2.4)
    axes[0].set_title("ANKLE", fontsize=16)
    axes[0].set_ylabel("ANGLE (DEGREES)")

    for curve in knee_stack:
        axes[1].plot(pct, curve, color="gray", alpha=0.22, linewidth=0.9)
    axes[1].plot(pct, knee_mean, color="black", linewidth=2.4)
    axes[1].set_title("KNEE", fontsize=16)
    axes[1].set_ylabel("ANGLE (DEGREES)")

    for curve in hip_stack:
        axes[2].plot(pct, curve, color="gray", alpha=0.22, linewidth=0.9)
    axes[2].plot(pct, hip_mean, color="black", linewidth=2.4)
    axes[2].set_title("HIP (N/A in source, filled with 0)", fontsize=16)
    axes[2].set_ylabel("ANGLE (DEGREES)")
    axes[2].set_xlabel("PERCENT OF GAIT CYCLE (%)")
    axes[2].set_xlim(0.0, 100.0)

    fig.suptitle(f"benchmark bilateral lower limb - level walking profile (n={ankle_stack.shape[0]} cycles)", fontsize=14)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a population profile using only level-walking cycles from benchmark bilateral lower limb."
    )
    parser.add_argument(
        "--eurobench-root",
        default="data/benchmark_datasets_for_bilateral_lower_limb/eurobench",
        help="Root with benchmark EUROBENCH outputs.",
    )
    parser.add_argument(
        "--out-root",
        default="data/benchmark_datasets_for_bilateral_lower_limb/plots/level_walking",
        help="Output folder for level_walking population files.",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=101,
        help="Number of normalized points per cycle.",
    )
    parser.add_argument(
        "--purity-threshold",
        type=float,
        default=0.95,
        help="Minimum proportion of Mode==1 samples inside the cycle window.",
    )
    parser.add_argument(
        "--min-duration-s",
        type=float,
        default=0.6,
        help="Minimum cycle duration.",
    )
    parser.add_argument(
        "--max-duration-s",
        type=float,
        default=1.8,
        help="Maximum cycle duration.",
    )
    args = parser.parse_args()

    eurobench_root = Path(args.eurobench_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    triplets = _trial_triplets(eurobench_root)
    if not triplets:
        raise FileNotFoundError(f"No trial triplets found under {eurobench_root}")

    all_cycles: list[dict] = []
    skipped_rows: list[dict] = []
    for joint_csv, events_yaml, info_yaml in triplets:
        try:
            cycles, skipped = _extract_level_walking_cycles(
                joint_csv=joint_csv,
                events_yaml=events_yaml,
                info_yaml=info_yaml,
                n_points=int(args.n_points),
                purity_threshold=float(args.purity_threshold),
                min_duration_s=float(args.min_duration_s),
                max_duration_s=float(args.max_duration_s),
            )
            all_cycles.extend(cycles)
            skipped_rows.extend(skipped)
        except Exception as exc:  # noqa: BLE001
            skipped_rows.append({"file": joint_csv.name, "reason": str(exc)})

    if not all_cycles:
        raise RuntimeError("No valid level-walking cycles extracted.")

    pct = all_cycles[0]["pct"]
    hip_stack = np.vstack([cycle["hip"] for cycle in all_cycles])
    knee_stack = np.vstack([cycle["knee"] for cycle in all_cycles])
    ankle_stack = np.vstack([cycle["ankle"] for cycle in all_cycles])

    out_png = out_root / "population_profile_level_walking.png"
    _plot_population(pct, hip_stack, knee_stack, ankle_stack, out_png)

    summary_csv = out_root / "population_profile_level_walking.csv"
    pd.DataFrame(
        {
            "pct": pct,
            "hip_mean": np.nanmean(hip_stack, axis=0),
            "hip_std": np.nanstd(hip_stack, axis=0),
            "knee_mean": np.nanmean(knee_stack, axis=0),
            "knee_std": np.nanstd(knee_stack, axis=0),
            "ankle_mean": np.nanmean(ankle_stack, axis=0),
            "ankle_std": np.nanstd(ankle_stack, axis=0),
        }
    ).to_csv(summary_csv, index=False)

    cycles_long_csv = out_root / "level_walking_cycles_long.csv"
    rows = []
    for cycle in all_cycles:
        for i, pct_val in enumerate(cycle["pct"]):
            rows.append(
                {
                    "subject": cycle["subject"],
                    "trial": cycle["trial"],
                    "cycle_index": cycle["cycle_index"],
                    "start_s": cycle["start_s"],
                    "end_s": cycle["end_s"],
                    "duration_s": cycle["duration_s"],
                    "level_walking_ratio": cycle["level_walking_ratio"],
                    "pct": float(pct_val),
                    "RHipAngles_x": float(cycle["hip"][i]),
                    "RKneeAngles_x": float(cycle["knee"][i]),
                    "RAnkleAngles_x": float(cycle["ankle"][i]),
                }
            )
    pd.DataFrame(rows).to_csv(cycles_long_csv, index=False)

    summary_yaml = out_root / "population_profile_level_walking.yaml"
    payload = {
        "trials_found": int(len(triplets)),
        "cycles_used": int(len(all_cycles)),
        "cycles_skipped": int(len(skipped_rows)),
        "mode_filter": {
            "mode_code": LEVEL_WALKING_MODE,
            "mode_label": "level_walking",
            "purity_threshold": float(args.purity_threshold),
        },
        "output_png": str(out_png),
        "output_csv": str(summary_csv),
        "output_cycles_long_csv": str(cycles_long_csv),
    }
    summary_yaml.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")

    if skipped_rows:
        skipped_csv = out_root / "level_walking_cycles_skipped.csv"
        pd.DataFrame(skipped_rows).to_csv(skipped_csv, index=False)
        print(skipped_csv)

    print(out_png)
    print(summary_csv)
    print(summary_yaml)


if __name__ == "__main__":
    main()
