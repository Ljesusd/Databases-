import argparse
import io
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io


MAT_PATTERN = re.compile(
    r"^Level 3 - MATLAB files/V3D exported data/P(?P<subject>\d+)exportedfiles/"
    r"p(?P<subject2>\d+)export_T(?P<trial>\d+)\.mat$",
    re.IGNORECASE,
)


@dataclass
class TrialCycle:
    subject: str
    condition: str
    run: str
    pct: np.ndarray
    ankle: np.ndarray
    knee: np.ndarray
    hip: np.ndarray
    cycle_start_s: float
    cycle_end_s: float
    source_mat: str


def _moving_average(x: np.ndarray, window: int = 7) -> np.ndarray:
    if window <= 1:
        return x
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x, kernel, mode="same")


def _local_minima(x: np.ndarray) -> np.ndarray:
    if x.size < 3:
        return np.array([], dtype=int)
    return np.where((x[1:-1] < x[:-2]) & (x[1:-1] <= x[2:]))[0] + 1


def _enforce_min_interval(indices: np.ndarray, min_frames: int) -> np.ndarray:
    if indices.size == 0:
        return indices
    kept = [int(indices[0])]
    for idx in indices[1:]:
        if int(idx) - kept[-1] >= min_frames:
            kept.append(int(idx))
    return np.asarray(kept, dtype=int)


def _pick_cycle_from_knee(
    knee_flex_deg: np.ndarray,
    fs_hz: float,
    min_stride_s: float = 0.5,
    max_stride_s: float = 2.0,
) -> tuple[int, int]:
    knee = np.asarray(knee_flex_deg, dtype=float)
    knee_smooth = _moving_average(knee, window=7)

    mins = _local_minima(knee_smooth)
    mins = _enforce_min_interval(mins, min_frames=max(1, int(0.45 * fs_hz)))
    if mins.size < 2:
        raise ValueError("Not enough knee minima to define cycle.")

    best: tuple[float, int, int] | None = None
    fallback: tuple[int, int] | None = None

    for a, b in zip(mins[:-1], mins[1:]):
        dur = (b - a) / fs_hz
        if not (min_stride_s <= dur <= max_stride_s):
            continue
        seg = knee_smooth[a : b + 1]
        if seg.size < 3:
            continue
        peak_i = int(np.argmax(seg))
        peak_frac = peak_i / float(seg.size - 1)
        amp = float(np.nanmax(seg) - 0.5 * (seg[0] + seg[-1]))
        score = -abs(peak_frac - 0.70) + 0.02 * amp

        if fallback is None:
            fallback = (a, b)
        if 0.35 <= peak_frac <= 0.95:
            if best is None or score > best[0]:
                best = (score, a, b)

    if best is not None:
        return best[1], best[2]
    if fallback is not None:
        return fallback
    raise ValueError("No valid knee-min to knee-min cycle found.")


def _normalize_segment(signal: np.ndarray, start_idx: int, end_idx: int, n_points: int) -> np.ndarray:
    if end_idx <= start_idx:
        raise ValueError("Invalid segment indices.")
    seg = np.asarray(signal[start_idx : end_idx + 1], dtype=float)
    if seg.size < 2:
        raise ValueError("Segment too short.")
    x_old = np.linspace(0.0, 1.0, seg.size)
    x_new = np.linspace(0.0, 1.0, n_points)
    return np.interp(x_new, x_old, seg)


def _unwrap_deg(signal_deg: np.ndarray) -> np.ndarray:
    raw = np.asarray(signal_deg, dtype=float)
    unwrapped = np.degrees(np.unwrap(np.radians(raw)))
    shift = 360.0 * round((np.nanmedian(unwrapped) - np.nanmedian(raw)) / 360.0)
    return unwrapped - shift


def _extract_trial_cycle_from_mat(
    mat_data: dict,
    subject: str,
    condition: str,
    run: str,
    source_mat: str,
    side: str,
    n_points: int,
    min_stride_s: float,
    max_stride_s: float,
) -> TrialCycle:
    side = side.lower()
    ankle_key = f"{side}_ank_angle"
    knee_key = f"{side}_kne_angle"
    hip_key = f"{side}_hip_angle"

    if any(k not in mat_data for k in [ankle_key, knee_key, hip_key]):
        raise ValueError(f"Missing angle keys for side '{side}'.")

    ankle = np.asarray(mat_data[ankle_key], dtype=float)
    knee = np.asarray(mat_data[knee_key], dtype=float)
    hip = np.asarray(mat_data[hip_key], dtype=float)
    if ankle.ndim != 2 or knee.ndim != 2 or hip.ndim != 2:
        raise ValueError("Unexpected angle array dimensions.")
    if min(ankle.shape[1], knee.shape[1], hip.shape[1]) < 1:
        raise ValueError("Angle arrays without sagittal column.")

    fs_hz = float(mat_data.get("FRAME_RATE", 120.0))
    knee_flex = knee[:, 0]
    ankle_flex = ankle[:, 0]
    hip_flex = _unwrap_deg(hip[:, 0])

    start_idx, end_idx = _pick_cycle_from_knee(
        knee_flex_deg=knee_flex,
        fs_hz=fs_hz,
        min_stride_s=min_stride_s,
        max_stride_s=max_stride_s,
    )

    ankle_norm = _normalize_segment(ankle_flex, start_idx, end_idx, n_points)
    knee_norm = _normalize_segment(knee_flex, start_idx, end_idx, n_points)
    hip_norm = _normalize_segment(hip_flex, start_idx, end_idx, n_points)
    pct = np.linspace(0.0, 100.0, n_points)

    return TrialCycle(
        subject=subject,
        condition=condition,
        run=run,
        pct=pct,
        ankle=ankle_norm,
        knee=knee_norm,
        hip=hip_norm,
        cycle_start_s=start_idx / fs_hz,
        cycle_end_s=end_idx / fs_hz,
        source_mat=source_mat,
    )


def _plot_cycles(cycles: list[TrialCycle], out_png: Path, title: str) -> None:
    pct = cycles[0].pct
    ankle = np.array([c.ankle for c in cycles])
    knee = np.array([c.knee for c in cycles])
    hip = np.array([c.hip for c in cycles])

    mean_ankle = np.nanmean(ankle, axis=0)
    mean_knee = np.nanmean(knee, axis=0)
    mean_hip = np.nanmean(hip, axis=0)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True, constrained_layout=True)
    fig.patch.set_facecolor("#ebebeb")
    for ax in axes:
        ax.set_facecolor("#ebebeb")
        ax.grid(alpha=0.22)

    for curve in ankle:
        axes[0].plot(pct, curve, color="gray", alpha=0.16, linewidth=1.0)
    axes[0].plot(pct, mean_ankle, color="black", linewidth=2.2)
    axes[0].set_title("ANKLE", fontsize=16, pad=8)

    for curve in knee:
        axes[1].plot(pct, curve, color="gray", alpha=0.16, linewidth=1.0)
    axes[1].plot(pct, mean_knee, color="black", linewidth=2.2)
    axes[1].set_title("KNEE", fontsize=16, pad=8)

    for curve in hip:
        axes[2].plot(pct, curve, color="gray", alpha=0.16, linewidth=1.0)
    axes[2].plot(pct, mean_hip, color="black", linewidth=2.2)
    axes[2].set_title("HIP", fontsize=16, pad=8)

    axes[2].set_xlabel("PERCENT OF GAIT CYCLE (%)", fontsize=14)
    axes[2].set_xlim(0.0, 100.0)
    fig.suptitle(title, fontsize=17, y=1.01)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _save_long_csv(cycles: list[TrialCycle], out_csv: Path) -> None:
    rows = []
    for c in cycles:
        for i, pct in enumerate(c.pct):
            rows.append(
                {
                    "subject": c.subject,
                    "condition": c.condition,
                    "run": c.run,
                    "pct": float(pct),
                    "ankle": float(c.ankle[i]),
                    "knee": float(c.knee[i]),
                    "hip": float(c.hip[i]),
                    "cycle_start_s": float(c.cycle_start_s),
                    "cycle_end_s": float(c.cycle_end_s),
                    "source_mat": c.source_mat,
                }
            )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot gait cycles directly from MATLAB processed angles (Level 3 zip).")
    parser.add_argument(
        "--level3-zip",
        default="/Volumes/Trabajo/A biomechanics dataset of healthy human walking/Level 3 - MATLAB files.zip",
        help="Path to Level 3 MATLAB zip.",
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        help="Optional conditions to include (e.g., C01 C02). Default: all.",
    )
    parser.add_argument(
        "--side",
        choices=["r", "l"],
        default="r",
        help="Side to plot (r or l).",
    )
    parser.add_argument(
        "--out-dir",
        default="data/biomechanics_human_walking/plots",
        help="Output directory for plot and CSV.",
    )
    parser.add_argument(
        "--name",
        default="all_matlab_angles",
        help="Output suffix name.",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=101,
        help="Number of normalized points.",
    )
    parser.add_argument(
        "--min-stride-s",
        type=float,
        default=0.5,
        help="Minimum knee-min to knee-min duration.",
    )
    parser.add_argument(
        "--max-stride-s",
        type=float,
        default=2.0,
        help="Maximum knee-min to knee-min duration.",
    )
    parser.add_argument(
        "--split-by-condition",
        action="store_true",
        help="Additionally save one PNG+CSV per condition (C01..C33) from the same run.",
    )
    args = parser.parse_args()

    include_conditions = None
    if args.conditions:
        include_conditions = {c.strip().upper() for c in args.conditions if c.strip()}

    cycles: list[TrialCycle] = []
    failures: list[str] = []

    with zipfile.ZipFile(args.level3_zip) as zf:
        mats = sorted([n for n in zf.namelist() if n.lower().endswith(".mat")])
        for mat_name in mats:
            match = MAT_PATTERN.match(mat_name)
            if match is None:
                continue

            subject = f"{int(match.group('subject')):02d}"
            subject2 = f"{int(match.group('subject2')):02d}"
            if subject != subject2:
                failures.append(f"{mat_name}: subject mismatch in filename.")
                continue

            trial_num = int(match.group("trial"))
            condition = f"C{trial_num:02d}"
            if include_conditions is not None and condition not in include_conditions:
                continue

            run = "01"
            if subject == "08" and condition == "C28" and mat_name.lower().endswith("p8export_t28.mat"):
                # Level 3 has one file for p8/T28; keep run 01 here.
                run = "01"

            try:
                mat_bytes = zf.read(mat_name)
                mat_data = scipy.io.loadmat(io.BytesIO(mat_bytes), simplify_cells=True)
                cycle = _extract_trial_cycle_from_mat(
                    mat_data=mat_data,
                    subject=subject,
                    condition=condition,
                    run=run,
                    source_mat=mat_name,
                    side=args.side,
                    n_points=args.n_points,
                    min_stride_s=args.min_stride_s,
                    max_stride_s=args.max_stride_s,
                )
                cycles.append(cycle)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{mat_name}: {exc}")

    if not cycles:
        print("status,no_valid_cycles")
        if failures:
            print("first_failures")
            for line in failures[:20]:
                print(line)
        return

    out_dir = Path(args.out_dir)

    def save_bundle(bundle_cycles: list[TrialCycle], bundle_name: str, bundle_label: str) -> tuple[Path, Path]:
        out_png_i = out_dir / f"gait_cycle_{bundle_name}_ankle_knee_hip.png"
        out_csv_i = out_dir / f"gait_cycle_{bundle_name}_ankle_knee_hip_long.csv"
        n_subjects_i = len({c.subject for c in bundle_cycles})
        title_i = (
            f"biomechanics_human_walking MATLAB angles "
            f"({bundle_label}, side={args.side.upper()}, subjects={n_subjects_i}, n={len(bundle_cycles)})"
        )
        _save_long_csv(bundle_cycles, out_csv_i)
        _plot_cycles(bundle_cycles, out_png_i, title_i)
        return out_png_i, out_csv_i

    cond_label = "all" if include_conditions is None else ",".join(sorted(include_conditions))
    out_png, out_csv = save_bundle(cycles, args.name, cond_label)

    print(f"saved_png,{out_png}")
    print(f"saved_csv,{out_csv}")
    print(f"n_subjects,{len({c.subject for c in cycles})}")
    print(f"n_cycles,{len(cycles)}")
    if args.split_by_condition:
        by_condition: dict[str, list[TrialCycle]] = defaultdict(list)
        for c in cycles:
            by_condition[c.condition].append(c)
        for cond in sorted(by_condition.keys(), key=lambda c: int(c[1:])):
            cond_cycles = by_condition[cond]
            cond_name = f"{args.name}_{cond}"
            cond_png, cond_csv = save_bundle(cond_cycles, cond_name, cond)
            print(f"saved_png,{cond_png}")
            print(f"saved_csv,{cond_csv}")
            print(f"condition,{cond},n_cycles,{len(cond_cycles)}")

    print(f"n_failures,{len(failures)}")
    if failures:
        print("first_failures")
        for line in failures[:10]:
            print(line)


if __name__ == "__main__":
    main()
