import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
MYPREDICT_DIR = THIS_DIR / "mypredict"
if str(MYPREDICT_DIR) not in sys.path:
    sys.path.insert(0, str(MYPREDICT_DIR))

import build_canonical_gait_profiles as canonical_profiles
import build_canonical_gait_trajectories as mypredict_canonical
from segment_gait_cycle_marker_angles import segment_and_normalize_marker_angles


DEFAULT_N_POINTS = 101
HUMAN_WALK_TASKS = {"Gait", "FastGait", "SlowGait", "2minWalk"}


@dataclass(frozen=True)
class DatasetPaths:
    name: str
    subject_manifest: Path
    population_root: Path


DATASETS = {
    "human_gait": DatasetPaths(
        name="human_gait",
        subject_manifest=Path("data/human_gait/subject_manifest.csv"),
        population_root=Path("data/human_gait/plots_canonical_population"),
    ),
    "multisensor_gait": DatasetPaths(
        name="multisensor_gait",
        subject_manifest=Path("data/multisensor_gait/subject_manifest.csv"),
        population_root=Path("data/multisensor_gait/plots_canonical_population"),
    ),
    "gait_analysis_assessment": DatasetPaths(
        name="gait_analysis_assessment",
        subject_manifest=Path("data/gait_analysis_assessment/subject_manifest.csv"),
        population_root=Path("data/gait_analysis_assessment/plots_canonical_population"),
    ),
    "benchmark_bilateral_lower_limb": DatasetPaths(
        name="benchmark_bilateral_lower_limb",
        subject_manifest=Path("data/benchmark_datasets_for_bilateral_lower_limb/subject_manifest.csv"),
        population_root=Path("data/benchmark_datasets_for_bilateral_lower_limb/plots_canonical_population"),
    ),
    "mypredict": DatasetPaths(
        name="mypredict",
        subject_manifest=Path("data/mypredict/subject_manifest.csv"),
        population_root=Path("data/mypredict/plots_canonical_population"),
    ),
}


def _text_or_empty(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def _load_angles(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    required = {"pct", "hip_flexion", "knee_flexion", "ankle_dorsiflexion"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"{csv_path.name} missing columns: {missing}")
    pct = pd.to_numeric(df["pct"], errors="coerce").to_numpy(dtype=float)
    hip = pd.to_numeric(df["hip_flexion"], errors="coerce").to_numpy(dtype=float)
    knee = pd.to_numeric(df["knee_flexion"], errors="coerce").to_numpy(dtype=float)
    ankle = pd.to_numeric(df["ankle_dorsiflexion"], errors="coerce").to_numpy(dtype=float)
    return pct, hip, knee, ankle


def _truthy_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y"})


def _ideal_hip_reference(n_points: int) -> np.ndarray:
    return np.cos(np.linspace(0.0, 2.0 * np.pi, n_points))


def _flip_if_needed(curve: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, bool]:
    centered_curve = curve - curve.mean()
    centered_ref = reference - reference.mean()
    denom = float(np.linalg.norm(centered_curve) * np.linalg.norm(centered_ref))
    if denom == 0.0:
        return curve, False
    corr = float(np.dot(centered_curve, centered_ref) / denom)
    if not np.isfinite(corr):
        return curve, False
    if corr < 0.0:
        return -curve, True
    return curve, False


def _save_population_plot(
    pct: np.ndarray,
    hips: np.ndarray,
    knees: np.ndarray,
    ankles: np.ndarray,
    hip_median: np.ndarray,
    knee_median: np.ndarray,
    ankle_median: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 8.5), sharex=True)
    panels = [
        ("ANKLE", ankles, ankle_median),
        ("KNEE", knees, knee_median),
        ("HIP", hips, hip_median),
    ]
    for ax, (panel_title, stack, median_curve) in zip(axes, panels):
        draw_stack = stack
        if draw_stack.shape[0] > 250:
            idx = np.linspace(0, draw_stack.shape[0] - 1, 250, dtype=int)
            draw_stack = draw_stack[idx]
        for curve in draw_stack:
            ax.plot(pct, curve, color="#8c8c8c", alpha=0.18, linewidth=0.8)
        ax.plot(pct, median_curve, color="black", linewidth=2.0)
        ax.set_title(panel_title)
        ax.set_xlim(0.0, 100.0)
        ax.set_yticks([])
    axes[-1].set_xlabel("PERCENT OF GAIT CYCLE (%)")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _offset_candidates(signed_curve: np.ndarray, template: np.ndarray) -> dict[str, float]:
    return {
        "none": 0.0,
        "start": float(signed_curve[0] - template[0]),
        "median": float(np.nanmedian(signed_curve) - np.nanmedian(template)),
        "mid": float(signed_curve[len(signed_curve) // 2] - template[len(template) // 2]),
    }


def _apply_existing_selection(
    joint_name: str,
    raw_curve: np.ndarray,
    template: np.ndarray,
    variant: str,
    sign: int,
    offset_kind: str = "",
    offset_deg: float | None = None,
) -> tuple[np.ndarray, str, float]:
    variants = canonical_profiles._curve_variants(np.asarray(raw_curve, dtype=float))
    if variant not in variants:
        raise KeyError(f"Unknown variant '{variant}' for {joint_name}")
    signed_curve = int(sign) * variants[variant]
    offsets = _offset_candidates(signed_curve, template)
    if offset_kind and offset_kind in offsets and offset_deg is not None and np.isfinite(offset_deg):
        return signed_curve - float(offset_deg), offset_kind, float(offset_deg)

    best_curve = None
    best_kind = ""
    best_offset = 0.0
    best_score = math.inf
    for kind, offset in offsets.items():
        candidate = signed_curve - offset
        score, _ = canonical_profiles._score_joint_curve(joint_name, candidate, template)
        if score < best_score:
            best_score = float(score)
            best_curve = candidate
            best_kind = kind
            best_offset = float(offset)
    if best_curve is None:
        raise ValueError(f"Unable to rebuild {joint_name} curve")
    return best_curve, best_kind, best_offset


def _build_population_rows(
    dataset_name: str,
    level: str,
    group: str,
    rows: list[dict],
    out_root: Path,
) -> dict:
    pct = rows[0]["pct"]
    hips = np.stack([row["hip_curve"] for row in rows], axis=0)
    knees = np.stack([row["knee_curve"] for row in rows], axis=0)
    ankles = np.stack([row["ankle_curve"] for row in rows], axis=0)

    reference = _ideal_hip_reference(hips.shape[1])
    corrected_hips = []
    hip_flipped = 0
    for curve in hips:
        curve_fixed, was_flipped = _flip_if_needed(curve, reference)
        if was_flipped:
            hip_flipped += 1
        corrected_hips.append(curve_fixed)
    hips = np.stack(corrected_hips, axis=0)

    hip_median = np.nanmedian(hips, axis=0)
    knee_median = np.nanmedian(knees, axis=0)
    ankle_median = np.nanmedian(ankles, axis=0)

    group_dir = out_root / level / group
    group_dir.mkdir(parents=True, exist_ok=True)
    angles_csv = group_dir / "population_marker_angles_norm101.csv"
    plot_png = group_dir / "population_profiles.png"
    sources_csv = group_dir / "population_sources.csv"

    pd.DataFrame(
        {
            "pct": pct,
            "hip_flexion": hip_median,
            "knee_flexion": knee_median,
            "ankle_dorsiflexion": ankle_median,
        }
    ).to_csv(angles_csv, index=False)

    source_rows = []
    for row in rows:
        source_rows.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"pct", "hip_curve", "knee_curve", "ankle_curve"}
            }
        )
    pd.DataFrame(source_rows).to_csv(sources_csv, index=False)

    title = f"{dataset_name} {level} population [{group}] (n={len(rows)}, hip_flipped={hip_flipped})"
    _save_population_plot(pct, hips, knees, ankles, hip_median, knee_median, ankle_median, title, plot_png)

    return {
        "dataset": dataset_name,
        "level": level,
        "group": group,
        "n_curves": int(len(rows)),
        "hip_flipped": int(hip_flipped),
        "angles_csv": str(angles_csv),
        "plot_png": str(plot_png),
        "sources_csv": str(sources_csv),
    }


def _subject_level_rows(dataset: str, subject_manifest: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    ok_df = subject_manifest[subject_manifest["status"].fillna("ok") == "ok"].copy()
    for _, row in ok_df.iterrows():
        pct, hip, knee, ankle = _load_angles(Path(row["angles_csv"]))
        rows.append(
            {
                "dataset": dataset,
                "level": "subject_level",
                "subject": _text_or_empty(row.get("subject")),
                "condition": _text_or_empty(row.get("condition")),
                "group_id": _text_or_empty(row.get("group_id")),
                "source_trial": "",
                "cycle_idx": np.nan,
                "selected_group": _text_or_empty(row.get("group_id")),
                "pct": pct,
                "hip_curve": hip,
                "knee_curve": knee,
                "ankle_curve": ankle,
            }
        )
    return rows


def _selected_metrics_rows(metrics_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    return df[_truthy_mask(df["selected"])].copy()


def _resolve_multisensor_legacy_path(legacy_processed_root: Path, subject: str, trial_name: str) -> Path:
    direct_path = legacy_processed_root / subject / trial_name
    if direct_path.exists():
        return direct_path
    anomalous_path = legacy_processed_root / "usuarios_anomalos" / subject / trial_name
    if anomalous_path.exists():
        return anomalous_path
    matches = list(legacy_processed_root.rglob(trial_name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Legacy multisensor angles not found for {subject}/{trial_name}")


def _human_like_trial_rows(dataset: str, subject_manifest: pd.DataFrame) -> list[dict]:
    templates = canonical_profiles._canonical_templates(np.linspace(0.0, 100.0, DEFAULT_N_POINTS))
    trial_cache: dict[Path, dict] = {}
    rows: list[dict] = []

    for _, manifest_row in subject_manifest[subject_manifest["status"].fillna("ok") == "ok"].iterrows():
        subject = _text_or_empty(manifest_row.get("subject"))
        condition = _text_or_empty(manifest_row.get("condition"))
        group_id = _text_or_empty(manifest_row.get("group_id"))
        angles_csv = Path(manifest_row["angles_csv"])
        metrics_csv = Path(str(angles_csv).replace("_canonical_marker_angles_norm101.csv", "_canonical_cycle_metrics.csv"))
        selected_df = _selected_metrics_rows(metrics_csv)

        for _, cycle_row in selected_df.iterrows():
            trial_name = _text_or_empty(cycle_row.get("trial"))
            if not trial_name:
                continue
            if dataset == "human_gait":
                traj_path = Path("data/human_gait/eurobench") / subject / trial_name
            else:
                traj_path = Path("data/gait_analysis_assessment/eurobench") / subject / trial_name

            cache = trial_cache.get(traj_path)
            if cache is None:
                traj_df = pd.read_csv(traj_path)
                traj_time = pd.to_numeric(traj_df["time"], errors="coerce").to_numpy(dtype=float)
                marker_data_all = canonical_profiles._extract_marker_data(traj_df)
                cache = {"time": traj_time, "marker_data_all": marker_data_all}
                trial_cache[traj_path] = cache

            start_t = float(cycle_row["start_time_s"])
            end_t = float(cycle_row["end_time_s"])
            traj_time = cache["time"]
            mask = (traj_time >= start_t) & (traj_time <= end_t)
            if int(np.count_nonzero(mask)) < 5:
                continue

            time_seg = traj_time[mask]
            marker_seg = {marker: values[mask] for marker, values in cache["marker_data_all"].items()}
            standardized, _ = canonical_profiles._standardize_cycle_frame(marker_seg)
            raw_candidates = canonical_profiles._build_right_joint_candidates(standardized)

            curves = {}
            for joint_name in ["hip", "knee", "ankle"]:
                source = _text_or_empty(cycle_row.get(f"{joint_name}_source"))
                variant = _text_or_empty(cycle_row.get(f"{joint_name}_variant"))
                sign = int(cycle_row.get(f"{joint_name}_sign", 1))
                raw_item = next((item for item in raw_candidates[joint_name] if item["source"] == source), None)
                if raw_item is None:
                    raise KeyError(f"{traj_path.name} missing {joint_name} source '{source}'")
                raw_curve = canonical_profiles._normalize_vector(time_seg, raw_item["curve"], DEFAULT_N_POINTS)
                curve, _, _ = _apply_existing_selection(
                    joint_name=joint_name,
                    raw_curve=raw_curve,
                    template=templates[joint_name],
                    variant=variant,
                    sign=sign,
                )
                curves[joint_name] = curve

            rows.append(
                {
                    "dataset": dataset,
                    "level": "trial_level",
                    "subject": subject,
                    "condition": condition,
                    "group_id": group_id,
                    "source_trial": trial_name,
                    "cycle_idx": int(cycle_row["cycle_idx"]),
                    "selected_group": group_id,
                    "pct": np.linspace(0.0, 100.0, DEFAULT_N_POINTS),
                    "hip_curve": curves["hip"],
                    "knee_curve": curves["knee"],
                    "ankle_curve": curves["ankle"],
                }
            )
    return rows


def _multisensor_trial_rows(subject_manifest: pd.DataFrame, legacy_processed_root: Path) -> list[dict]:
    rows: list[dict] = []
    templates = canonical_profiles._canonical_templates(np.linspace(0.0, 100.0, DEFAULT_N_POINTS))
    raw_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    for _, manifest_row in subject_manifest[subject_manifest["status"].fillna("ok") == "ok"].iterrows():
        subject = _text_or_empty(manifest_row.get("subject"))
        group_id = _text_or_empty(manifest_row.get("group_id"))
        angles_csv = Path(manifest_row["angles_csv"])
        metrics_csv = Path(str(angles_csv).replace("_canonical_marker_angles_norm101.csv", "_canonical_cycle_metrics.csv"))
        selected_df = _selected_metrics_rows(metrics_csv)

        for _, cycle_row in selected_df.iterrows():
            trial_name = _text_or_empty(cycle_row.get("trial"))
            cache_key = (subject, trial_name)
            curves = raw_cache.get(cache_key)
            if curves is None:
                traj_path = Path("data/multisensor_gait/eurobench") / subject / trial_name.replace(
                    "_marker_angles_norm101.csv",
                    "_Trajectories.csv",
                )
                events_path = Path("data/multisensor_gait/eurobench") / subject / trial_name.replace(
                    "_marker_angles_norm101.csv",
                    "_gaitEvents.yaml",
                )
                if traj_path.exists() and events_path.exists():
                    _, hip_raw, knee_raw, ankle_raw = segment_and_normalize_marker_angles(
                        str(traj_path),
                        str(events_path),
                        n_points=DEFAULT_N_POINTS,
                        angle_mode="3d",
                        cycle_mode="knee_min",
                        ankle_zero_90=True,
                        hip_sagittal=True,
                    )
                else:
                    legacy_path = _resolve_multisensor_legacy_path(legacy_processed_root, subject, trial_name)
                    legacy_df = pd.read_csv(legacy_path)
                    hip_raw = pd.to_numeric(legacy_df["hip_flexion"], errors="coerce").to_numpy(dtype=float)
                    knee_raw = pd.to_numeric(legacy_df["knee_flexion"], errors="coerce").to_numpy(dtype=float)
                    ankle_raw = pd.to_numeric(legacy_df["ankle_dorsiflexion"], errors="coerce").to_numpy(dtype=float)
                curves = {"hip": hip_raw, "knee": knee_raw, "ankle": ankle_raw}
                raw_cache[cache_key] = curves

            selected_curves = {}
            for joint_name in ["hip", "knee", "ankle"]:
                variant = _text_or_empty(cycle_row.get(f"{joint_name}_variant"))
                sign = int(cycle_row.get(f"{joint_name}_sign", 1))
                offset_kind = _text_or_empty(cycle_row.get(f"{joint_name}_offset_kind"))
                offset_deg = pd.to_numeric(pd.Series([cycle_row.get(f"{joint_name}_offset_deg")]), errors="coerce").iloc[0]
                curve, _, _ = _apply_existing_selection(
                    joint_name=joint_name,
                    raw_curve=curves[joint_name],
                    template=templates[joint_name],
                    variant=variant,
                    sign=sign,
                    offset_kind=offset_kind,
                    offset_deg=float(offset_deg) if np.isfinite(offset_deg) else None,
                )
                selected_curves[joint_name] = curve

            rows.append(
                {
                    "dataset": "multisensor_gait",
                    "level": "trial_level",
                    "subject": subject,
                    "condition": "",
                    "group_id": group_id,
                    "source_trial": trial_name,
                    "cycle_idx": np.nan,
                    "selected_group": group_id,
                    "pct": np.linspace(0.0, 100.0, DEFAULT_N_POINTS),
                    "hip_curve": selected_curves["hip"],
                    "knee_curve": selected_curves["knee"],
                    "ankle_curve": selected_curves["ankle"],
                }
            )
    return rows


def _benchmark_trial_rows(subject_manifest: pd.DataFrame, purity_threshold: float) -> list[dict]:
    rows: list[dict] = []
    templates = canonical_profiles._canonical_templates(np.linspace(0.0, 100.0, DEFAULT_N_POINTS))
    cycle_cache: dict[Path, dict[int, dict]] = {}

    for _, manifest_row in subject_manifest[subject_manifest["status"].fillna("ok") == "ok"].iterrows():
        subject = _text_or_empty(manifest_row.get("subject"))
        group_id = _text_or_empty(manifest_row.get("group_id"))
        angles_csv = Path(manifest_row["angles_csv"])
        metrics_csv = Path(str(angles_csv).replace("_canonical_marker_angles_norm101.csv", "_canonical_cycle_metrics.csv"))
        selected_df = _selected_metrics_rows(metrics_csv)

        for _, cycle_row in selected_df.iterrows():
            trial_name = _text_or_empty(cycle_row.get("trial"))
            trial_base = trial_name.replace("_jointAngles.csv", "")
            joint_csv = (
                Path("data/benchmark_datasets_for_bilateral_lower_limb/eurobench")
                / subject
                / f"{trial_base}_jointAngles.csv"
            )
            events_yaml = joint_csv.with_name(joint_csv.name.replace("_jointAngles.csv", "_gaitEvents.yaml"))
            info_yaml = joint_csv.with_name(joint_csv.name.replace("_jointAngles.csv", "_info.yaml"))
            cycles_by_idx = cycle_cache.get(joint_csv)
            if cycles_by_idx is None:
                cycles, _ = canonical_profiles._benchmark_cycle_candidates(
                    joint_csv=joint_csv,
                    events_yaml=events_yaml,
                    info_yaml=info_yaml,
                    n_points=DEFAULT_N_POINTS,
                    purity_threshold=purity_threshold,
                    min_duration_s=0.6,
                    max_duration_s=2.0,
                )
                cycles_by_idx = {int(item["cycle_idx"]): item for item in cycles}
                cycle_cache[joint_csv] = cycles_by_idx
            cycle = cycles_by_idx.get(int(cycle_row["cycle_idx"]))
            if cycle is None:
                continue

            selected_curves = {}
            hip_source = _text_or_empty(cycle_row.get("hip_source"))
            if hip_source:
                hip_variant = _text_or_empty(cycle_row.get("hip_variant"))
                hip_sign = int(cycle_row.get("hip_sign", 1))
                selected_curves["hip"], _, _ = _apply_existing_selection(
                    joint_name="hip",
                    raw_curve=cycle["hip_raw"],
                    template=templates["hip"],
                    variant=hip_variant,
                    sign=hip_sign,
                )
            else:
                selected_curves["hip"] = np.zeros(DEFAULT_N_POINTS, dtype=float)

            for joint_name in ["knee", "ankle"]:
                variant = _text_or_empty(cycle_row.get(f"{joint_name}_variant"))
                sign = int(cycle_row.get(f"{joint_name}_sign", 1))
                selected_curves[joint_name], _, _ = _apply_existing_selection(
                    joint_name=joint_name,
                    raw_curve=cycle[f"{joint_name}_raw"],
                    template=templates[joint_name],
                    variant=variant,
                    sign=sign,
                )

            rows.append(
                {
                    "dataset": "benchmark_bilateral_lower_limb",
                    "level": "trial_level",
                    "subject": subject,
                    "condition": "",
                    "group_id": group_id,
                    "source_trial": trial_name,
                    "cycle_idx": int(cycle_row["cycle_idx"]),
                    "selected_group": group_id,
                    "pct": np.linspace(0.0, 100.0, DEFAULT_N_POINTS),
                    "hip_curve": selected_curves["hip"],
                    "knee_curve": selected_curves["knee"],
                    "ankle_curve": selected_curves["ankle"],
                }
            )
    return rows


def _mypredict_trial_rows(subject_manifest: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    joint_cache: dict[Path, tuple[np.ndarray, pd.DataFrame]] = {}

    for _, manifest_row in subject_manifest[subject_manifest["status"].fillna("ok") == "ok"].iterrows():
        subject = _text_or_empty(manifest_row.get("subject"))
        group_id = _text_or_empty(manifest_row.get("group_id"))
        angles_csv = Path(manifest_row["angles_csv"])
        metrics_csv = Path(str(angles_csv).replace("_canonical_marker_angles_norm101.csv", "_canonical_cycle_metrics.csv"))
        selected_df = _selected_metrics_rows(metrics_csv)

        for _, cycle_row in selected_df.iterrows():
            trial_name = _text_or_empty(cycle_row.get("trial"))
            joint_csv = Path("data/mypredict/eurobench") / subject / trial_name.replace("_Trajectories.csv", "_jointAngles.csv")
            cached = joint_cache.get(joint_csv)
            if cached is None:
                joint_df = pd.read_csv(joint_csv)
                joint_time = pd.to_numeric(joint_df["time"], errors="coerce").to_numpy(dtype=float)
                cached = (joint_time, joint_df)
                joint_cache[joint_csv] = cached
            joint_time, joint_df = cached
            start_t = float(cycle_row["start_time_s"])
            end_t = float(cycle_row["end_time_s"])
            mask = (joint_time >= start_t) & (joint_time <= end_t)
            if int(np.count_nonzero(mask)) < 5:
                continue
            time_seg = joint_time[mask]

            curves = {}
            for joint_name, prefix in mypredict_canonical.JOINT_PREFIXES.items():
                axis = _text_or_empty(cycle_row.get(f"{joint_name}_axis"))
                sign = int(cycle_row.get(f"{joint_name}_sign", 1))
                col = f"{prefix}_{axis}"
                values = pd.to_numeric(joint_df[col], errors="coerce").to_numpy(dtype=float)[mask]
                curves[joint_name] = sign * mypredict_canonical._normalize_vector(time_seg, values, DEFAULT_N_POINTS)

            rows.append(
                {
                    "dataset": "mypredict",
                    "level": "trial_level",
                    "subject": subject,
                    "condition": "",
                    "group_id": group_id,
                    "source_trial": trial_name,
                    "cycle_idx": int(cycle_row["cycle_idx"]),
                    "selected_group": group_id,
                    "pct": np.linspace(0.0, 100.0, DEFAULT_N_POINTS),
                    "hip_curve": curves["hip"],
                    "knee_curve": curves["knee"],
                    "ankle_curve": curves["ankle"],
                }
            )
    return rows


def _trial_level_rows(dataset: str, subject_manifest: pd.DataFrame, args) -> list[dict]:
    if dataset == "human_gait":
        return _human_like_trial_rows(dataset, subject_manifest)
    if dataset == "gait_analysis_assessment":
        return _human_like_trial_rows(dataset, subject_manifest)
    if dataset == "multisensor_gait":
        return _multisensor_trial_rows(subject_manifest, Path(args.multisensor_legacy_processed_root))
    if dataset == "benchmark_bilateral_lower_limb":
        return _benchmark_trial_rows(subject_manifest, purity_threshold=args.benchmark_purity_threshold)
    if dataset == "mypredict":
        return _mypredict_trial_rows(subject_manifest)
    raise KeyError(f"Unsupported dataset '{dataset}'")


def _group_names(subject_rows: list[dict], trial_rows: list[dict]) -> list[str]:
    condition_set = {
        row["condition"]
        for row in subject_rows + trial_rows
        if _text_or_empty(row.get("condition"))
    }
    return ["all"] + sorted(condition_set)


def _rows_for_group(rows: list[dict], group: str) -> list[dict]:
    if group == "all":
        return rows
    return [row for row in rows if row["condition"] == group]


def build_dataset_levels(dataset: str, args) -> pd.DataFrame:
    dataset_paths = DATASETS[dataset]
    subject_manifest = pd.read_csv(dataset_paths.subject_manifest)
    subject_rows = _subject_level_rows(dataset, subject_manifest)
    trial_rows = _trial_level_rows(dataset, subject_manifest, args)
    groups = _group_names(subject_rows, trial_rows)

    out_rows: list[dict] = []
    for level, rows in [("subject_level", subject_rows), ("trial_level", trial_rows)]:
        for group in groups:
            selected_rows = _rows_for_group(rows, group)
            if not selected_rows:
                continue
            out_rows.append(
                _build_population_rows(
                    dataset_name=dataset,
                    level=level,
                    group=group,
                    rows=selected_rows,
                    out_root=dataset_paths.population_root,
                )
            )

    out_df = pd.DataFrame(out_rows)
    dataset_paths.population_root.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(dataset_paths.population_root / "population_manifest.csv", index=False)
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build subject-level and trial-level population plots in place.")
    parser.add_argument(
        "--datasets",
        default="human_gait,multisensor_gait,gait_analysis_assessment,benchmark_bilateral_lower_limb,mypredict",
        help="Comma-separated dataset names.",
    )
    parser.add_argument(
        "--multisensor-legacy-processed-root",
        default="data/multisensor_gait/processed 11-08-23-207",
    )
    parser.add_argument("--benchmark-purity-threshold", type=float, default=0.95)
    args = parser.parse_args()

    wanted = [item.strip() for item in args.datasets.split(",") if item.strip()]
    for name in wanted:
        if name not in DATASETS:
            raise KeyError(f"Unknown dataset '{name}'")
        build_dataset_levels(name, args)
        print(DATASETS[name].population_root / "population_manifest.csv")


if __name__ == "__main__":
    main()
