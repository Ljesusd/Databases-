import argparse
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


DEFAULT_N_POINTS = 101

HEALTHYPIG_TRIAL_RE = re.compile(r"^SUBJ(?P<subject>\d+)_(?P<run>\d+)$", re.IGNORECASE)
MULTIMODAL_VIDEO_RE = re.compile(
    r"^subject_(?P<subject>S\d+)_cond_(?P<condition>A\d+)_run_(?P<run>\d+)_Trajectories\.csv$",
    re.IGNORECASE,
)
LOWER_LIMB_RE = re.compile(
    r"^Subject(?P<subject>\d+)_(?P<condition>V\d+)_(?P<run>\d+)_Trajectories\.csv$",
    re.IGNORECASE,
)
RUNNING_CLINIC_RE = re.compile(
    r"^Subject(?P<subject>\d+)_(?P<condition>WALK|RUN)_(?P<run>\d+)_Trajectories\.csv$",
    re.IGNORECASE,
)
BIOMECH_RE = re.compile(
    r"^(?P<subject>\d+)_(?P<condition>C\d+)_(?P<run>\d+)_Trajectories\.csv$",
    re.IGNORECASE,
)
MULTIMODAL_SPEED_RE = re.compile(
    r"^(?P<subject>\d+)_(?P<condition>C\d+)_(?P<run>\d+)_Trajectories\.csv$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DatasetLayout:
    name: str
    root: Path
    trial_manifest: Path
    subject_manifest: Path
    processed_root: Path
    plots_root: Path
    population_root: Path


DATASETS = {
    "healthypig": DatasetLayout(
        name="healthypig",
        root=Path("data/HealthyPiG/138_HealthyPiG"),
        trial_manifest=Path("data/HealthyPiG/138_HealthyPiG/trial_manifest.csv"),
        subject_manifest=Path("data/HealthyPiG/138_HealthyPiG/subject_manifest.csv"),
        processed_root=Path("data/HealthyPiG/138_HealthyPiG/processed_canonical"),
        plots_root=Path("data/HealthyPiG/138_HealthyPiG/plots_canonical"),
        population_root=Path("data/HealthyPiG/138_HealthyPiG/plots_canonical_population"),
    ),
    "multimodal_video_imu": DatasetLayout(
        name="multimodal_video_imu",
        root=Path("data/Multimodal video and IMU kinematic"),
        trial_manifest=Path("data/Multimodal video and IMU kinematic/trial_manifest.csv"),
        subject_manifest=Path("data/Multimodal video and IMU kinematic/subject_manifest.csv"),
        processed_root=Path("data/Multimodal video and IMU kinematic/processed_canonical"),
        plots_root=Path("data/Multimodal video and IMU kinematic/plots_canonical"),
        population_root=Path("data/Multimodal video and IMU kinematic/plots_canonical_population"),
    ),
    "lower_limb_kinematic": DatasetLayout(
        name="lower_limb_kinematic",
        root=Path("data/lower_limb_kinematic"),
        trial_manifest=Path("data/lower_limb_kinematic/trial_manifest.csv"),
        subject_manifest=Path("data/lower_limb_kinematic/subject_manifest.csv"),
        processed_root=Path("data/lower_limb_kinematic/processed_canonical"),
        plots_root=Path("data/lower_limb_kinematic/plots_canonical"),
        population_root=Path("data/lower_limb_kinematic/plots_canonical_population"),
    ),
    "running_injury_clinic_kinematic": DatasetLayout(
        name="running_injury_clinic_kinematic",
        root=Path("data/running_injury_clinic_kinematic/healthy"),
        trial_manifest=Path("data/running_injury_clinic_kinematic/healthy/trial_manifest.csv"),
        subject_manifest=Path("data/running_injury_clinic_kinematic/healthy/subject_manifest.csv"),
        processed_root=Path("data/running_injury_clinic_kinematic/healthy/processed_canonical"),
        plots_root=Path("data/running_injury_clinic_kinematic/healthy/plots_canonical"),
        population_root=Path("data/running_injury_clinic_kinematic/healthy/plots_canonical_population"),
    ),
    "biomechanics_human_walking": DatasetLayout(
        name="biomechanics_human_walking",
        root=Path("data/biomechanics_human_walking"),
        trial_manifest=Path("data/biomechanics_human_walking/trial_manifest.csv"),
        subject_manifest=Path("data/biomechanics_human_walking/subject_manifest.csv"),
        processed_root=Path("data/biomechanics_human_walking/processed_canonical"),
        plots_root=Path("data/biomechanics_human_walking/plots_canonical"),
        population_root=Path("data/biomechanics_human_walking/plots_canonical_population"),
    ),
    "multimodal_walking_speeds": DatasetLayout(
        name="multimodal_walking_speeds",
        root=Path("data/multimodal_walking_speeds"),
        trial_manifest=Path("data/multimodal_walking_speeds/trial_manifest.csv"),
        subject_manifest=Path("data/multimodal_walking_speeds/subject_manifest.csv"),
        processed_root=Path("data/multimodal_walking_speeds/processed_canonical"),
        plots_root=Path("data/multimodal_walking_speeds/plots_canonical"),
        population_root=Path("data/multimodal_walking_speeds/plots_canonical_population"),
    ),
}


def _truthy_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y"})


def _path_or_empty(path: Path | None) -> str:
    if path is None:
        return ""
    return str(path)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _safe_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


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


def _save_overlay_plot(
    curves: list[dict],
    title: str,
    out_path: Path,
) -> None:
    pct = curves[0]["pct"]
    hips = np.stack([row["hip_curve"] for row in curves], axis=0)
    knees = np.stack([row["knee_curve"] for row in curves], axis=0)
    ankles = np.stack([row["ankle_curve"] for row in curves], axis=0)
    hip_med = np.nanmedian(hips, axis=0)
    knee_med = np.nanmedian(knees, axis=0)
    ankle_med = np.nanmedian(ankles, axis=0)

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 8.5), sharex=True)
    panels = [
        ("ANKLE", ankles, ankle_med),
        ("KNEE", knees, knee_med),
        ("HIP", hips, hip_med),
    ]
    for ax, (panel_title, stack, median_curve) in zip(axes, panels):
        for curve in stack:
            ax.plot(pct, curve, color="#8c8c8c", alpha=0.22, linewidth=0.9)
        ax.plot(pct, median_curve, color="black", linewidth=2.0)
        ax.set_title(panel_title)
        ax.set_xlim(0.0, 100.0)
        ax.set_yticks([])
    axes[-1].set_xlabel("PERCENT OF GAIT CYCLE (%)")
    fig.suptitle(title)
    fig.tight_layout()
    _ensure_parent(out_path)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _save_population_plot(
    curves: list[dict],
    title: str,
    out_path: Path,
) -> int:
    pct = curves[0]["pct"]
    hips = np.stack([row["hip_curve"] for row in curves], axis=0)
    knees = np.stack([row["knee_curve"] for row in curves], axis=0)
    ankles = np.stack([row["ankle_curve"] for row in curves], axis=0)

    reference = _ideal_hip_reference(hips.shape[1])
    hip_flipped = 0
    corrected_hips = []
    for curve in hips:
        fixed, was_flipped = _flip_if_needed(curve, reference)
        corrected_hips.append(fixed)
        if was_flipped:
            hip_flipped += 1
    hips = np.stack(corrected_hips, axis=0)

    hip_med = np.nanmedian(hips, axis=0)
    knee_med = np.nanmedian(knees, axis=0)
    ankle_med = np.nanmedian(ankles, axis=0)

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 8.5), sharex=True)
    panels = [
        ("ANKLE", ankles, ankle_med),
        ("KNEE", knees, knee_med),
        ("HIP", hips, hip_med),
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
    _ensure_parent(out_path)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return hip_flipped


def _save_curve_csv(row: dict, out_path: Path) -> None:
    pd.DataFrame(
        {
            "pct": row["pct"],
            "hip_flexion": row["hip_curve"],
            "knee_flexion": row["knee_curve"],
            "ankle_dorsiflexion": row["ankle_curve"],
        }
    ).to_csv(out_path, index=False)


def _group_output_dir(root: Path, condition: str, subject: str) -> Path:
    return root / condition / subject if condition else root / subject


def _group_population_names(rows: list[dict]) -> list[str]:
    conditions = sorted({_safe_text(row.get("condition")) for row in rows if _safe_text(row.get("condition"))})
    return ["all"] + conditions


def _rows_for_group(rows: list[dict], group: str) -> list[dict]:
    if group == "all":
        return rows
    return [row for row in rows if _safe_text(row.get("condition")) == group]


def _build_population_artifacts(
    dataset_name: str,
    rows: list[dict],
    population_root: Path,
) -> pd.DataFrame:
    manifest_rows = []
    for level in ["subject_level", "trial_level"]:
        level_rows = [row for row in rows if row["level"] == level]
        if not level_rows:
            continue
        for group in _group_population_names(level_rows):
            curr = _rows_for_group(level_rows, group)
            if not curr:
                continue
            out_dir = population_root / level / group
            out_dir.mkdir(parents=True, exist_ok=True)
            angles_csv = out_dir / "population_marker_angles_norm101.csv"
            plot_png = out_dir / "population_profiles.png"
            sources_csv = out_dir / "population_sources.csv"

            hip_flipped = _save_population_plot(
                curr,
                title=f"{dataset_name} {level} population [{group}] (n={len(curr)})",
                out_path=plot_png,
            )
            pct = curr[0]["pct"]
            hips = np.stack([row["hip_curve"] for row in curr], axis=0)
            knees = np.stack([row["knee_curve"] for row in curr], axis=0)
            ankles = np.stack([row["ankle_curve"] for row in curr], axis=0)
            reference = _ideal_hip_reference(hips.shape[1])
            corrected_hips = np.stack([_flip_if_needed(curve, reference)[0] for curve in hips], axis=0)
            pd.DataFrame(
                {
                    "pct": pct,
                    "hip_flexion": np.nanmedian(corrected_hips, axis=0),
                    "knee_flexion": np.nanmedian(knees, axis=0),
                    "ankle_dorsiflexion": np.nanmedian(ankles, axis=0),
                }
            ).to_csv(angles_csv, index=False)

            source_rows = []
            for row in curr:
                source_rows.append(
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"pct", "hip_curve", "knee_curve", "ankle_curve"}
                    }
                )
            pd.DataFrame(source_rows).to_csv(sources_csv, index=False)
            manifest_rows.append(
                {
                    "dataset": dataset_name,
                    "level": level,
                    "group": group,
                    "n_curves": int(len(curr)),
                    "hip_flipped": int(hip_flipped),
                    "angles_csv": str(angles_csv),
                    "plot_png": str(plot_png),
                    "sources_csv": str(sources_csv),
                }
            )
    manifest_df = pd.DataFrame(manifest_rows)
    population_root.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(population_root / "population_manifest.csv", index=False)
    return manifest_df


def _trial_rows_from_long_df(
    df: pd.DataFrame,
    dataset: str,
    subject_col: str,
    condition_col: str,
    trial_col: str,
    run_col: str | None,
    pct_col: str,
    hip_col: str,
    knee_col: str,
    ankle_col: str,
    side_col: str | None = None,
    cycle_col: str | None = None,
    group_id_fn=None,
    source_trial_fn=None,
) -> list[dict]:
    id_cols = []
    for col in [subject_col, condition_col, trial_col, run_col, side_col, cycle_col]:
        if col is not None and col not in id_cols:
            id_cols.append(col)

    rows: list[dict] = []
    for key, group_df in df.groupby(id_cols, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        key_map = dict(zip(id_cols, key))
        group_df = group_df.sort_values(pct_col)
        pct = pd.to_numeric(group_df[pct_col], errors="coerce").to_numpy(dtype=float)
        hip = pd.to_numeric(group_df[hip_col], errors="coerce").to_numpy(dtype=float)
        knee = pd.to_numeric(group_df[knee_col], errors="coerce").to_numpy(dtype=float)
        ankle = pd.to_numeric(group_df[ankle_col], errors="coerce").to_numpy(dtype=float)
        subject = _safe_text(key_map[subject_col])
        condition = _safe_text(key_map[condition_col])
        source_trial = (
            source_trial_fn(key_map) if source_trial_fn is not None else _safe_text(key_map[trial_col])
        )
        group_id = group_id_fn(subject, condition) if group_id_fn is not None else f"{subject}_{condition}".strip("_")
        rows.append(
            {
                "dataset": dataset,
                "level": "trial_level",
                "subject": subject,
                "condition": condition,
                "group_id": group_id,
                "source_trial": source_trial,
                "run": _safe_text(key_map.get(run_col)) if run_col is not None else "",
                "side": _safe_text(key_map.get(side_col)) if side_col is not None else "",
                "cycle_idx": (
                    int(key_map[cycle_col])
                    if cycle_col is not None and pd.notna(key_map[cycle_col])
                    else np.nan
                ),
                "pct": pct,
                "hip_curve": hip,
                "knee_curve": knee,
                "ankle_curve": ankle,
            }
        )
    return rows


def _subject_rows_from_trial_rows(
    dataset: str,
    layout: DatasetLayout,
    trial_rows: list[dict],
    candidate_counts: dict[str, int],
    selected_counts: dict[str, int],
) -> tuple[list[dict], pd.DataFrame]:
    subject_rows: list[dict] = []
    manifest_rows = []

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in trial_rows:
        grouped[row["group_id"]].append(row)

    for group_id in sorted(grouped):
        group_curves = grouped[group_id]
        subject = group_curves[0]["subject"]
        condition = _safe_text(group_curves[0]["condition"])
        pct = group_curves[0]["pct"]
        hip = np.nanmedian(np.stack([row["hip_curve"] for row in group_curves], axis=0), axis=0)
        knee = np.nanmedian(np.stack([row["knee_curve"] for row in group_curves], axis=0), axis=0)
        ankle = np.nanmedian(np.stack([row["ankle_curve"] for row in group_curves], axis=0), axis=0)

        out_dir = _group_output_dir(layout.processed_root, condition, subject)
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_dir = _group_output_dir(layout.plots_root, condition, subject)
        plot_dir.mkdir(parents=True, exist_ok=True)

        angles_csv = out_dir / f"{group_id}_canonical_marker_angles_norm101.csv"
        summary_yaml = out_dir / f"{group_id}_canonical_summary.yaml"
        plot_png = plot_dir / f"{group_id}_canonical_profiles.png"

        curve_row = {
            "dataset": dataset,
            "level": "subject_level",
            "subject": subject,
            "condition": condition,
            "group_id": group_id,
            "source_trial": "",
            "run": "",
            "side": "",
            "cycle_idx": np.nan,
            "pct": pct,
            "hip_curve": hip,
            "knee_curve": knee,
            "ankle_curve": ankle,
        }
        _save_curve_csv(curve_row, angles_csv)
        _save_overlay_plot(
            group_curves,
            title=f"{dataset} {group_id} canonical (selected={len(group_curves)})",
            out_path=plot_png,
        )

        trial_names = sorted({row["source_trial"] for row in group_curves if row["source_trial"]})
        summary_payload = {
            "dataset": dataset,
            "subject": subject,
            "condition": condition or None,
            "group_id": group_id,
            "candidate_cycles": int(candidate_counts.get(group_id, len(group_curves))),
            "selected_cycles": int(selected_counts.get(group_id, len(group_curves))),
            "n_trials": int(len(trial_names)),
            "source_trials": trial_names,
        }
        with summary_yaml.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(summary_payload, handle, sort_keys=False)

        subject_rows.append(curve_row)
        manifest_rows.append(
            {
                "dataset": dataset,
                "subject": subject,
                "condition": condition,
                "group_id": group_id,
                "angles_csv": str(angles_csv),
                "trajectories_csv": "",
                "summary_yaml": str(summary_yaml),
                "plot_png": str(plot_png),
                "status": "ok",
                "n_trials": int(len(trial_names)),
                "candidate_cycles": int(candidate_counts.get(group_id, len(group_curves))),
                "selected_cycles": int(selected_counts.get(group_id, len(group_curves))),
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)
    layout.subject_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(layout.subject_manifest, index=False)
    return subject_rows, manifest_df


def _build_trial_manifest_rows(
    dataset: str,
    trajectories_paths: list[Path],
    parse_fn,
    joint_suffix: str = "_jointAngles.csv",
    gait_suffix: str = "_gaitEvents.yaml",
    point_suffix: str = "_point_gaitEvents.yaml",
    info_suffix: str = "_info.yaml",
) -> pd.DataFrame:
    rows = []
    for traj_path in sorted(trajectories_paths):
        parsed = parse_fn(traj_path)
        trial_base = traj_path.stem.replace("_Trajectories", "")
        joint_path = traj_path.with_name(f"{trial_base}{joint_suffix}")
        gait_path = traj_path.with_name(f"{trial_base}{gait_suffix}")
        point_path = traj_path.with_name(f"{trial_base}{point_suffix}")
        info_path = traj_path.with_name(f"{trial_base}{info_suffix}")
        rows.append(
            {
                "dataset": dataset,
                "subject": parsed["subject"],
                "condition": parsed["condition"],
                "trial_id": trial_base,
                "signal_kind": "trajectories",
                "signal_csv": str(traj_path),
                "trajectories_csv": str(traj_path),
                "joint_angles_csv": _path_or_empty(joint_path if joint_path.exists() else None),
                "gait_events_yaml": _path_or_empty(gait_path if gait_path.exists() else None),
                "point_gait_events_yaml": _path_or_empty(point_path if point_path.exists() else None),
                "info_yaml": _path_or_empty(info_path if info_path.exists() else None),
            }
        )
    return pd.DataFrame(rows)


def _parse_healthypig_trial(traj_path: Path) -> dict:
    stem = traj_path.stem.replace("_Trajectories", "")
    subject = traj_path.parent.name
    return {"subject": subject, "condition": "", "trial_id": stem}


def _parse_multimodal_video_trial(traj_path: Path) -> dict:
    match = MULTIMODAL_VIDEO_RE.match(traj_path.name)
    if match is None:
        raise ValueError(f"Unexpected multimodal_video_imu filename: {traj_path.name}")
    return {"subject": match.group("subject"), "condition": match.group("condition")}


def _parse_lower_limb_trial(traj_path: Path) -> dict:
    match = LOWER_LIMB_RE.match(traj_path.name)
    if match is None:
        raise ValueError(f"Unexpected lower_limb_kinematic filename: {traj_path.name}")
    return {"subject": f"Subject{match.group('subject').zfill(2)}", "condition": match.group("condition").upper()}


def _parse_running_trial(traj_path: Path) -> dict:
    match = RUNNING_CLINIC_RE.match(traj_path.name)
    if match is None:
        raise ValueError(f"Unexpected running clinic filename: {traj_path.name}")
    return {"subject": f"Subject{match.group('subject')}", "condition": match.group("condition").upper()}


def _parse_biomech_trial(traj_path: Path) -> dict:
    match = BIOMECH_RE.match(traj_path.name)
    if match is None:
        raise ValueError(f"Unexpected biomechanics filename: {traj_path.name}")
    return {"subject": match.group("subject").zfill(2), "condition": match.group("condition").upper()}


def _parse_multimodal_speed_trial(traj_path: Path) -> dict:
    match = MULTIMODAL_SPEED_RE.match(traj_path.name)
    if match is None:
        raise ValueError(f"Unexpected multimodal walking filename: {traj_path.name}")
    return {"subject": match.group("subject"), "condition": match.group("condition").upper()}


def _healthypig_trial_to_eurobench(subject: str, raw_trial: str) -> str:
    match = HEALTHYPIG_TRIAL_RE.match(raw_trial)
    if match is None:
        raise ValueError(f"Unexpected HealthyPiG trial key: {raw_trial}")
    subject_num = int(match.group("subject"))
    run = int(match.group("run"))
    return f"SUBJ{subject_num:02d} ({run})_Trajectories.csv"


def build_healthypig() -> dict:
    layout = DATASETS["healthypig"]
    traj_paths = sorted((layout.root / "eurobench").rglob("*_Trajectories.csv"))
    trial_manifest = _build_trial_manifest_rows(
        dataset=layout.name,
        trajectories_paths=traj_paths,
        parse_fn=_parse_healthypig_trial,
        point_suffix="_point_events.yaml",
    )
    trial_manifest.to_csv(layout.trial_manifest, index=False)

    trial_rows: list[dict] = []
    candidate_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}

    for summary_path in sorted((layout.root / "processed").glob("SUBJ*/canonical_cycles_summary.csv")):
        subject = summary_path.parent.name
        group_id = subject
        summary_df = pd.read_csv(summary_path)
        candidate_counts[group_id] = int(len(summary_df))
        selected_counts[group_id] = int(len(summary_df))
        for _, summary_row in summary_df.iterrows():
            trial_key = _safe_text(summary_row["trial"])
            best_csv = summary_path.parent / f"{trial_key}_best_canonical_flexion_norm101.csv"
            canonical_csv = summary_path.parent / f"{trial_key}_canonical_flexion_norm101.csv"
            curve_csv = best_csv if best_csv.exists() else canonical_csv
            if not curve_csv.exists():
                continue
            curve_df = pd.read_csv(curve_csv)
            trial_rows.append(
                {
                    "dataset": layout.name,
                    "level": "trial_level",
                    "subject": subject,
                    "condition": "",
                    "group_id": group_id,
                    "source_trial": _healthypig_trial_to_eurobench(subject, trial_key),
                    "run": "",
                    "side": "",
                    "cycle_idx": np.nan,
                    "pct": pd.to_numeric(curve_df["pct"], errors="coerce").to_numpy(dtype=float),
                    "hip_curve": pd.to_numeric(curve_df["hip_flexion"], errors="coerce").to_numpy(dtype=float),
                    "knee_curve": pd.to_numeric(curve_df["knee_flexion"], errors="coerce").to_numpy(dtype=float),
                    "ankle_curve": pd.to_numeric(curve_df["ankle_dorsiflexion"], errors="coerce").to_numpy(dtype=float),
                }
            )

    subject_rows, subject_manifest = _subject_rows_from_trial_rows(
        dataset=layout.name,
        layout=layout,
        trial_rows=trial_rows,
        candidate_counts=candidate_counts,
        selected_counts=selected_counts,
    )
    population_manifest = _build_population_artifacts(layout.name, subject_rows + trial_rows, layout.population_root)
    return {
        "dataset": layout.name,
        "trial_manifest": len(trial_manifest),
        "subject_manifest": len(subject_manifest),
        "population_manifest": len(population_manifest),
    }


def build_multimodal_video_imu() -> dict:
    layout = DATASETS["multimodal_video_imu"]
    traj_paths = []
    for path in sorted((layout.root / "eurobench").glob("*_Trajectories.csv")):
        match = MULTIMODAL_VIDEO_RE.match(path.name)
        if match is None:
            continue
        if match.group("condition") not in {"A01", "A02", "A03"}:
            continue
        traj_paths.append(path)
    trial_manifest = _build_trial_manifest_rows(
        dataset=layout.name,
        trajectories_paths=traj_paths,
        parse_fn=_parse_multimodal_video_trial,
    )
    trial_manifest.to_csv(layout.trial_manifest, index=False)

    metrics = pd.read_csv(layout.root / "analysis/joint_angles/trial_cycle_metrics.csv")
    long_df = pd.read_csv(layout.root / "analysis/joint_angles/joint_cycles_long.csv")
    metrics["subject"] = metrics["subject"].astype(str)
    metrics["condition"] = metrics["condition"].astype(str)
    long_df["subject"] = long_df["subject"].astype(str)
    long_df["condition"] = long_df["condition"].astype(str)
    keep_metrics = metrics[_truthy_mask(metrics["canonical_keep"])].copy()
    keep_keys = keep_metrics[["subject", "condition", "run", "side"]].drop_duplicates()
    selected_long = long_df.merge(keep_keys, on=["subject", "condition", "run", "side"], how="inner")

    trial_rows = _trial_rows_from_long_df(
        df=selected_long,
        dataset=layout.name,
        subject_col="subject",
        condition_col="condition",
        trial_col="run",
        run_col="run",
        pct_col="pct",
        hip_col="hip",
        knee_col="knee",
        ankle_col="ankle",
        side_col="side",
        group_id_fn=lambda subject, condition: f"{subject}_{condition}",
        source_trial_fn=lambda key_map: (
            f"subject_{key_map['subject']}_cond_{key_map['condition']}_run_{int(key_map['run']):03d}_Trajectories.csv"
        ),
    )

    candidate_counts = (
        metrics.groupby(["subject", "condition"]).size().rename("candidate_cycles").reset_index()
    )
    selected_counts = (
        keep_metrics.groupby(["subject", "condition"]).size().rename("selected_cycles").reset_index()
    )
    cand_map = {
        f"{row.subject}_{row.condition}": int(row.candidate_cycles)
        for row in candidate_counts.itertuples(index=False)
    }
    sel_map = {
        f"{row.subject}_{row.condition}": int(row.selected_cycles)
        for row in selected_counts.itertuples(index=False)
    }

    subject_rows, subject_manifest = _subject_rows_from_trial_rows(
        dataset=layout.name,
        layout=layout,
        trial_rows=trial_rows,
        candidate_counts=cand_map,
        selected_counts=sel_map,
    )
    population_manifest = _build_population_artifacts(layout.name, subject_rows + trial_rows, layout.population_root)
    return {
        "dataset": layout.name,
        "trial_manifest": len(trial_manifest),
        "subject_manifest": len(subject_manifest),
        "population_manifest": len(population_manifest),
    }


def build_lower_limb_kinematic() -> dict:
    layout = DATASETS["lower_limb_kinematic"]
    traj_paths = sorted((layout.root / "eurobench").rglob("*_Trajectories.csv"))
    trial_manifest = _build_trial_manifest_rows(
        dataset=layout.name,
        trajectories_paths=traj_paths,
        parse_fn=_parse_lower_limb_trial,
        point_suffix="_point_gaitEvents.yaml",
    )
    trial_manifest.to_csv(layout.trial_manifest, index=False)

    long_df = pd.read_csv(layout.root / "analysis/cycles_long.csv")
    filter_df = pd.read_csv(layout.root / "analysis/reference_shape_filter_log.csv")
    keep_keys = filter_df[_truthy_mask(filter_df["keep"])][["trial", "side", "cycle_id"]].drop_duplicates()
    selected_long = long_df.merge(keep_keys, on=["trial", "side", "cycle_id"], how="inner")
    selected_long = selected_long.rename(
        columns={"speed": "condition", "hip_deg": "hip", "knee_deg": "knee", "ankle_deg": "ankle"}
    )

    trial_rows = _trial_rows_from_long_df(
        df=selected_long,
        dataset=layout.name,
        subject_col="subject",
        condition_col="condition",
        trial_col="trial",
        run_col="run",
        pct_col="pct",
        hip_col="hip",
        knee_col="knee",
        ankle_col="ankle",
        side_col="side",
        cycle_col="cycle_id",
        group_id_fn=lambda subject, condition: f"{subject}_{condition}",
        source_trial_fn=lambda key_map: f"{key_map['trial']}_Trajectories.csv",
    )

    candidate_df = long_df.rename(columns={"speed": "condition"})[
        ["subject", "condition", "trial", "side", "cycle_id"]
    ].drop_duplicates()
    selected_df = selected_long[["subject", "condition", "trial", "side", "cycle_id"]].drop_duplicates()
    cand_map = {
        f"{subject}_{condition}": int(count)
        for (subject, condition), count in candidate_df.groupby(["subject", "condition"]).size().items()
    }
    sel_map = {
        f"{subject}_{condition}": int(count)
        for (subject, condition), count in selected_df.groupby(["subject", "condition"]).size().items()
    }

    subject_rows, subject_manifest = _subject_rows_from_trial_rows(
        dataset=layout.name,
        layout=layout,
        trial_rows=trial_rows,
        candidate_counts=cand_map,
        selected_counts=sel_map,
    )
    population_manifest = _build_population_artifacts(layout.name, subject_rows + trial_rows, layout.population_root)
    return {
        "dataset": layout.name,
        "trial_manifest": len(trial_manifest),
        "subject_manifest": len(subject_manifest),
        "population_manifest": len(population_manifest),
    }


def build_running_injury_clinic_kinematic() -> dict:
    layout = DATASETS["running_injury_clinic_kinematic"]
    traj_paths = sorted((layout.root / "eurobench").rglob("*_WALK_*_Trajectories.csv"))
    trial_manifest = _build_trial_manifest_rows(
        dataset=layout.name,
        trajectories_paths=traj_paths,
        parse_fn=_parse_running_trial,
    )
    trial_manifest.to_csv(layout.trial_manifest, index=False)

    long_df = pd.read_csv(layout.root / "analysis_test/cycles_long_walk.csv")
    filter_df = pd.read_csv(layout.root / "analysis_test/reference_shape_filter_log_walk.csv")
    keep_keys = filter_df[_truthy_mask(filter_df["keep"])][["trial", "condition", "side", "cycle_id"]].drop_duplicates()
    selected_long = long_df.merge(keep_keys, on=["trial", "condition", "side", "cycle_id"], how="inner")
    selected_long = selected_long.rename(columns={"hip_deg": "hip", "knee_deg": "knee", "ankle_deg": "ankle"})

    trial_rows = _trial_rows_from_long_df(
        df=selected_long,
        dataset=layout.name,
        subject_col="subject",
        condition_col="condition",
        trial_col="trial",
        run_col="run",
        pct_col="pct",
        hip_col="hip",
        knee_col="knee",
        ankle_col="ankle",
        side_col="side",
        cycle_col="cycle_id",
        group_id_fn=lambda subject, condition: f"{subject}_{condition}",
        source_trial_fn=lambda key_map: f"{key_map['trial']}_Trajectories.csv",
    )

    candidate_df = long_df[["subject", "condition", "trial", "side", "cycle_id"]].drop_duplicates()
    selected_df = selected_long[["subject", "condition", "trial", "side", "cycle_id"]].drop_duplicates()
    cand_map = {
        f"{subject}_{condition}": int(count)
        for (subject, condition), count in candidate_df.groupby(["subject", "condition"]).size().items()
    }
    sel_map = {
        f"{subject}_{condition}": int(count)
        for (subject, condition), count in selected_df.groupby(["subject", "condition"]).size().items()
    }

    subject_rows, subject_manifest = _subject_rows_from_trial_rows(
        dataset=layout.name,
        layout=layout,
        trial_rows=trial_rows,
        candidate_counts=cand_map,
        selected_counts=sel_map,
    )
    population_manifest = _build_population_artifacts(layout.name, subject_rows + trial_rows, layout.population_root)
    return {
        "dataset": layout.name,
        "trial_manifest": len(trial_manifest),
        "subject_manifest": len(subject_manifest),
        "population_manifest": len(population_manifest),
    }


def _load_simple_long_csv(long_csv: Path, dataset: str) -> pd.DataFrame:
    df = pd.read_csv(long_csv)
    expected = {"subject", "condition", "run", "pct", "ankle", "knee", "hip"}
    missing = sorted(expected.difference(df.columns))
    if missing:
        raise KeyError(f"{long_csv} missing columns: {missing}")
    df["dataset"] = dataset
    return df


def build_biomechanics_human_walking() -> dict:
    layout = DATASETS["biomechanics_human_walking"]
    traj_paths = sorted((layout.root / "eurobench").rglob("*_Trajectories.csv"))
    traj_paths = [path for path in traj_paths if BIOMECH_RE.match(path.name) is not None]
    trial_manifest = _build_trial_manifest_rows(
        dataset=layout.name,
        trajectories_paths=traj_paths,
        parse_fn=_parse_biomech_trial,
        joint_suffix="_jointAngles.csv",
        gait_suffix="_gaitEvents.yaml",
        point_suffix="_point_gaitEvents.yaml",
    )
    trial_manifest.to_csv(layout.trial_manifest, index=False)

    long_df = _load_simple_long_csv(layout.root / "plots/gait_cycle_all_ankle_knee_hip_long.csv", dataset=layout.name)
    long_df["subject"] = long_df["subject"].astype(int).map(lambda value: f"{value:02d}")
    long_df["condition"] = long_df["condition"].astype(str).str.upper()

    trial_rows = _trial_rows_from_long_df(
        df=long_df,
        dataset=layout.name,
        subject_col="subject",
        condition_col="condition",
        trial_col="run",
        run_col="run",
        pct_col="pct",
        hip_col="hip",
        knee_col="knee",
        ankle_col="ankle",
        group_id_fn=lambda subject, condition: f"{subject}_{condition}",
        source_trial_fn=lambda key_map: f"{key_map['subject']}_{key_map['condition']}_{int(key_map['run']):02d}_Trajectories.csv",
    )

    candidate_df = long_df[["subject", "condition", "run"]].drop_duplicates()
    cand_map = {
        f"{subject}_{condition}": int(count)
        for (subject, condition), count in candidate_df.groupby(["subject", "condition"]).size().items()
    }
    subject_rows, subject_manifest = _subject_rows_from_trial_rows(
        dataset=layout.name,
        layout=layout,
        trial_rows=trial_rows,
        candidate_counts=cand_map,
        selected_counts=cand_map,
    )
    population_manifest = _build_population_artifacts(layout.name, subject_rows + trial_rows, layout.population_root)
    return {
        "dataset": layout.name,
        "trial_manifest": len(trial_manifest),
        "subject_manifest": len(subject_manifest),
        "population_manifest": len(population_manifest),
    }


def build_multimodal_walking_speeds() -> dict:
    layout = DATASETS["multimodal_walking_speeds"]
    traj_paths = sorted((layout.root / "eurobench").rglob("*_Trajectories.csv"))
    traj_paths = [path for path in traj_paths if MULTIMODAL_SPEED_RE.match(path.name) is not None]
    trial_manifest = _build_trial_manifest_rows(
        dataset=layout.name,
        trajectories_paths=traj_paths,
        parse_fn=_parse_multimodal_speed_trial,
        joint_suffix="_jointAngles.csv",
        gait_suffix="_gaitEvents.yaml",
        point_suffix="_point_gaitEvents.yaml",
    )
    trial_manifest.to_csv(layout.trial_manifest, index=False)

    frames = []
    for long_csv in sorted((layout.root / "plots").glob("gait_cycle_C*_ankle_knee_hip_long.csv")):
        frames.append(_load_simple_long_csv(long_csv, dataset=layout.name))
    long_df = pd.concat(frames, ignore_index=True)
    long_df["condition"] = long_df["condition"].astype(str).str.upper()
    long_df["subject"] = long_df["subject"].astype(str)

    trial_rows = _trial_rows_from_long_df(
        df=long_df,
        dataset=layout.name,
        subject_col="subject",
        condition_col="condition",
        trial_col="run",
        run_col="run",
        pct_col="pct",
        hip_col="hip",
        knee_col="knee",
        ankle_col="ankle",
        group_id_fn=lambda subject, condition: f"{subject}_{condition}",
        source_trial_fn=lambda key_map: f"{key_map['subject']}_{key_map['condition']}_{int(key_map['run']):02d}_Trajectories.csv",
    )

    candidate_df = long_df[["subject", "condition", "run"]].drop_duplicates()
    cand_map = {
        f"{subject}_{condition}": int(count)
        for (subject, condition), count in candidate_df.groupby(["subject", "condition"]).size().items()
    }
    subject_rows, subject_manifest = _subject_rows_from_trial_rows(
        dataset=layout.name,
        layout=layout,
        trial_rows=trial_rows,
        candidate_counts=cand_map,
        selected_counts=cand_map,
    )
    population_manifest = _build_population_artifacts(layout.name, subject_rows + trial_rows, layout.population_root)
    return {
        "dataset": layout.name,
        "trial_manifest": len(trial_manifest),
        "subject_manifest": len(subject_manifest),
        "population_manifest": len(population_manifest),
    }


BUILDERS = {
    "healthypig": build_healthypig,
    "multimodal_video_imu": build_multimodal_video_imu,
    "lower_limb_kinematic": build_lower_limb_kinematic,
    "running_injury_clinic_kinematic": build_running_injury_clinic_kinematic,
    "biomechanics_human_walking": build_biomechanics_human_walking,
    "multimodal_walking_speeds": build_multimodal_walking_speeds,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate in-place trial, subject and population levels for the remaining gait datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=list(BUILDERS.keys()),
        choices=list(BUILDERS.keys()),
        help="Subset of datasets to process.",
    )
    args = parser.parse_args()

    summary_rows = []
    for dataset in args.datasets:
        summary_rows.append(BUILDERS[dataset]())

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
