import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HUMAN_TASK_REGEX = re.compile(r"P\d+_S\d+_(?P<task>[^_]+)_(?P<run>\d+)$")
MULTISENSOR_DAY_REGEX = re.compile(r".*_(?P<day>day\d+)_(?P<run>\d+)$")
MYPREDICT_DAY_REGEX = re.compile(r"(?P<subject>MP\d+)_(?P<day>Day\d+)_Trial_(?P<trial>\d+)$")
BENCHMARK_CONDITION_REGEX = re.compile(r"(?P<subject>AB\d+)_(?P<condition>[^_]+)_(?P<trial>\d+)$")
HUMAN_WALK_TASKS = {"Gait", "FastGait", "SlowGait", "2minWalk"}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    trial_root: Path
    trial_signal_glob: str
    trial_signal_kind: str
    subject_root: Path
    subject_summary_csv: Path
    plot_root: Path | None


DATASETS = {
    "human_gait": DatasetConfig(
        name="human_gait",
        trial_root=Path("data/human_gait/eurobench"),
        trial_signal_glob="*_Trajectories.csv",
        trial_signal_kind="trajectories",
        subject_root=Path("data/human_gait/processed_canonical"),
        subject_summary_csv=Path("data/human_gait/processed_canonical/human_gait_canonical_groups_summary.csv"),
        plot_root=Path("data/human_gait/plots_canonical"),
    ),
    "multisensor_gait": DatasetConfig(
        name="multisensor_gait",
        trial_root=Path("data/multisensor_gait/eurobench"),
        trial_signal_glob="*_Trajectories.csv",
        trial_signal_kind="trajectories",
        subject_root=Path("data/multisensor_gait/processed_canonical"),
        subject_summary_csv=Path("data/multisensor_gait/processed_canonical/multisensor_gait_canonical_subjects_summary.csv"),
        plot_root=Path("data/multisensor_gait/plots_canonical"),
    ),
    "gait_analysis_assessment": DatasetConfig(
        name="gait_analysis_assessment",
        trial_root=Path("data/gait_analysis_assessment/eurobench"),
        trial_signal_glob="*_Trajectories.csv",
        trial_signal_kind="trajectories",
        subject_root=Path("data/gait_analysis_assessment/processed_canonical"),
        subject_summary_csv=Path("data/gait_analysis_assessment/processed_canonical/gait_analysis_assessment_canonical_subjects_summary.csv"),
        plot_root=Path("data/gait_analysis_assessment/plots_canonical"),
    ),
    "benchmark_bilateral_lower_limb": DatasetConfig(
        name="benchmark_bilateral_lower_limb",
        trial_root=Path("data/benchmark_datasets_for_bilateral_lower_limb/eurobench"),
        trial_signal_glob="*_jointAngles.csv",
        trial_signal_kind="jointAngles",
        subject_root=Path("data/benchmark_datasets_for_bilateral_lower_limb/processed_canonical"),
        subject_summary_csv=Path(
            "data/benchmark_datasets_for_bilateral_lower_limb/processed_canonical/benchmark_bilateral_lower_limb_canonical_subjects_summary.csv"
        ),
        plot_root=Path("data/benchmark_datasets_for_bilateral_lower_limb/plots_canonical"),
    ),
    "mypredict": DatasetConfig(
        name="mypredict",
        trial_root=Path("data/mypredict/eurobench"),
        trial_signal_glob="*_Trajectories.csv",
        trial_signal_kind="trajectories",
        subject_root=Path("data/mypredict/processed_canonical"),
        subject_summary_csv=Path("data/mypredict/processed_canonical/mypredict_canonical_subjects_summary.csv"),
        plot_root=Path("data/mypredict/plots_canonical"),
    ),
}


def _ideal_hip_reference(n_points: int) -> np.ndarray:
    return np.cos(np.linspace(0.0, 2.0 * np.pi, n_points))


def _flip_if_needed(curve: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, bool]:
    a = curve - curve.mean()
    b = reference - reference.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return curve, False
    corr = float(np.dot(a, b) / denom)
    if not np.isfinite(corr):
        return curve, False
    if corr < 0.0:
        return -curve, True
    return curve, False


def _condition_from_trial(dataset: str, trial_stem: str) -> str:
    if dataset == "human_gait":
        match = HUMAN_TASK_REGEX.match(trial_stem)
        return match.group("task") if match else ""
    if dataset == "multisensor_gait":
        match = MULTISENSOR_DAY_REGEX.match(trial_stem)
        return match.group("day") if match else ""
    if dataset == "mypredict":
        match = MYPREDICT_DAY_REGEX.match(trial_stem)
        return match.group("day") if match else ""
    if dataset == "benchmark_bilateral_lower_limb":
        match = BENCHMARK_CONDITION_REGEX.match(trial_stem)
        return match.group("condition") if match else ""
    return ""


def _companion_paths(signal_path: Path, signal_kind: str) -> dict[str, str]:
    if signal_kind == "trajectories":
        stem = signal_path.name.replace("_Trajectories.csv", "")
        trajectories_csv = signal_path
        joint_angles_csv = signal_path.with_name(f"{stem}_jointAngles.csv")
    else:
        stem = signal_path.name.replace("_jointAngles.csv", "")
        trajectories_csv = signal_path.with_name(f"{stem}_Trajectories.csv")
        joint_angles_csv = signal_path
    gait_events_yaml = signal_path.with_name(f"{stem}_gaitEvents.yaml")
    point_gait_events_yaml = signal_path.with_name(f"{stem}_point_gaitEvents.yaml")
    info_yaml = signal_path.with_name(f"{stem}_info.yaml")
    return {
        "trial_stem": stem,
        "trajectories_csv": str(trajectories_csv) if trajectories_csv.exists() else "",
        "joint_angles_csv": str(joint_angles_csv) if joint_angles_csv.exists() else "",
        "gait_events_yaml": str(gait_events_yaml) if gait_events_yaml.exists() else "",
        "point_gait_events_yaml": str(point_gait_events_yaml) if point_gait_events_yaml.exists() else "",
        "info_yaml": str(info_yaml) if info_yaml.exists() else "",
    }


def build_trial_manifest(config: DatasetConfig) -> pd.DataFrame:
    rows: list[dict] = []
    for signal_path in sorted(config.trial_root.rglob(config.trial_signal_glob)):
        subject = signal_path.parent.name
        companions = _companion_paths(signal_path, config.trial_signal_kind)
        condition = _condition_from_trial(config.name, companions["trial_stem"])
        if config.name == "human_gait" and condition not in HUMAN_WALK_TASKS:
            continue
        rows.append(
            {
                "dataset": config.name,
                "subject": subject,
                "condition": condition,
                "trial_id": companions["trial_stem"],
                "signal_kind": config.trial_signal_kind,
                "signal_csv": str(signal_path),
                "trajectories_csv": companions["trajectories_csv"],
                "joint_angles_csv": companions["joint_angles_csv"],
                "gait_events_yaml": companions["gait_events_yaml"],
                "point_gait_events_yaml": companions["point_gait_events_yaml"],
                "info_yaml": companions["info_yaml"],
            }
        )
    return pd.DataFrame(rows)


def _derive_plot_path(config: DatasetConfig, angles_path: Path, subject: str) -> str:
    if config.plot_root is None:
        return ""
    if config.name == "mypredict":
        plot_path = config.plot_root / subject / f"{subject}_canonical_profiles.png"
    else:
        rel = angles_path.relative_to(config.subject_root)
        plot_path = config.plot_root / rel.parent / rel.name.replace(
            "_canonical_marker_angles_norm101.csv",
            "_canonical_profiles.png",
        )
    return str(plot_path) if plot_path.exists() else ""


def build_subject_manifest(config: DatasetConfig) -> pd.DataFrame:
    df = pd.read_csv(config.subject_summary_csv)
    rows: list[dict] = []
    for _, row in df.iterrows():
        if config.name == "mypredict":
            subject = str(row["subject"])
            condition = ""
            group_id = subject
            angles_csv = str(row["angles_file"])
            trajectories_csv = str(row["trajectory_file"])
            summary_yaml = str(config.subject_root / subject / f"{subject}_canonical_summary.yaml")
            plot_png = str(row["plot_file"]) if "plot_file" in row and isinstance(row["plot_file"], str) else ""
            status = "ok"
            n_trials = int(row["trials_seen"])
        else:
            subject = str(row["subject"])
            condition = str(row["task"]) if "task" in row and pd.notna(row["task"]) else ""
            group_id = str(row["basename"]) if "basename" in row else str(row["group"])
            angles_csv = str(row["angles_csv"])
            trajectories_csv = "" if pd.isna(row.get("trajectories_csv")) else str(row["trajectories_csv"])
            summary_yaml = str(row["summary_yaml"]) if "summary_yaml" in row else ""
            plot_png = _derive_plot_path(config, Path(angles_csv), subject)
            status = str(row.get("status", "ok"))
            n_trials = int(row["n_trials"]) if "n_trials" in row and pd.notna(row["n_trials"]) else 0

        rows.append(
            {
                "dataset": config.name,
                "subject": subject,
                "condition": condition,
                "group_id": group_id,
                "angles_csv": angles_csv,
                "trajectories_csv": trajectories_csv,
                "summary_yaml": summary_yaml if Path(summary_yaml).exists() else "",
                "plot_png": plot_png if plot_png and Path(plot_png).exists() else "",
                "status": status,
                "n_trials": n_trials,
                "candidate_cycles": int(row["candidate_cycles"]) if "candidate_cycles" in row and pd.notna(row["candidate_cycles"]) else np.nan,
                "selected_cycles": int(row["selected_cycles"]) if "selected_cycles" in row and pd.notna(row["selected_cycles"]) else np.nan,
            }
        )
    return pd.DataFrame(rows)


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


def _save_population_plot(
    pct: np.ndarray,
    hips: np.ndarray,
    knees: np.ndarray,
    ankles: np.ndarray,
    hip_med: np.ndarray,
    knee_med: np.ndarray,
    ankle_med: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 8.5), sharex=True)
    panels = [
        ("ANKLE", ankles, ankle_med),
        ("KNEE", knees, knee_med),
        ("HIP", hips, hip_med),
    ]
    for ax, (panel_title, stack, median_curve) in zip(axes, panels):
        for curve in stack:
            ax.plot(pct, curve, color="#8c8c8c", alpha=0.2, linewidth=0.8)
        ax.plot(pct, median_curve, color="black", linewidth=2.0)
        ax.set_title(panel_title)
        ax.set_xlim(0, 100)
        ax.set_yticks([])
    axes[-1].set_xlabel("PERCENT OF GAIT CYCLE (%)")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def build_population_level(dataset_name: str, subject_df: pd.DataFrame, out_root: Path) -> pd.DataFrame:
    ok_df = subject_df[subject_df["status"].fillna("ok") == "ok"].copy()
    groups: list[tuple[str, pd.DataFrame]] = [("all", ok_df)]
    valid_conditions = sorted({cond for cond in ok_df["condition"].fillna("").astype(str) if cond})
    if len(valid_conditions) > 1:
        groups.extend((cond, ok_df[ok_df["condition"] == cond].copy()) for cond in valid_conditions)

    rows: list[dict] = []
    for group_name, group_df in groups:
        if group_df.empty:
            continue

        pct_ref = None
        hips, knees, ankles = [], [], []
        source_rows = []
        for _, row in group_df.iterrows():
            angles_path = Path(row["angles_csv"])
            if not angles_path.exists():
                continue
            pct, hip, knee, ankle = _load_angles(angles_path)
            if pct_ref is None:
                pct_ref = pct
            elif not np.allclose(pct_ref, pct):
                continue
            hips.append(hip)
            knees.append(knee)
            ankles.append(ankle)
            source_rows.append(row.to_dict())

        if not hips or pct_ref is None:
            continue

        hips = np.stack(hips)
        knees = np.stack(knees)
        ankles = np.stack(ankles)
        reference = _ideal_hip_reference(hips.shape[1])
        corrected_hips = []
        hip_flipped = 0
        for curve in hips:
            curve_fixed, was_flipped = _flip_if_needed(curve, reference)
            if was_flipped:
                hip_flipped += 1
            corrected_hips.append(curve_fixed)
        hips = np.stack(corrected_hips)

        hip_med = np.median(hips, axis=0)
        knee_med = np.median(knees, axis=0)
        ankle_med = np.median(ankles, axis=0)

        group_dir = out_root / dataset_name / group_name
        csv_path = group_dir / "population_marker_angles_norm101.csv"
        png_path = group_dir / "population_profiles.png"
        src_csv = group_dir / "population_sources.csv"

        group_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "pct": pct_ref,
                "hip_flexion": hip_med,
                "knee_flexion": knee_med,
                "ankle_dorsiflexion": ankle_med,
            }
        ).to_csv(csv_path, index=False)
        pd.DataFrame(source_rows).to_csv(src_csv, index=False)
        title = f"{dataset_name} canonical population [{group_name}] (n={hips.shape[0]}, hip_flipped={hip_flipped})"
        _save_population_plot(pct_ref, hips, knees, ankles, hip_med, knee_med, ankle_med, title, png_path)

        rows.append(
            {
                "dataset": dataset_name,
                "group": group_name,
                "n_curves": int(hips.shape[0]),
                "hip_flipped": int(hip_flipped),
                "angles_csv": str(csv_path),
                "plot_png": str(png_path),
                "sources_csv": str(src_csv),
            }
        )

    return pd.DataFrame(rows)


def build_dataset(config: DatasetConfig, output_root: Path) -> dict[str, int]:
    trial_dir = output_root / "level1_trial" / config.name
    subject_dir = output_root / "level2_subject" / config.name
    population_dir = output_root / "level3_population"
    trial_dir.mkdir(parents=True, exist_ok=True)
    subject_dir.mkdir(parents=True, exist_ok=True)
    population_dir.mkdir(parents=True, exist_ok=True)

    trial_df = build_trial_manifest(config)
    trial_df.to_csv(trial_dir / "trial_manifest.csv", index=False)

    subject_df = build_subject_manifest(config)
    subject_df.to_csv(subject_dir / "subject_manifest.csv", index=False)

    population_df = build_population_level(config.name, subject_df, population_dir)
    (population_dir / config.name).mkdir(parents=True, exist_ok=True)
    population_df.to_csv(population_dir / config.name / "population_manifest.csv", index=False)

    return {
        "dataset": config.name,
        "trial_entries": int(len(trial_df)),
        "subject_entries": int(len(subject_df)),
        "population_entries": int(len(population_df)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="human_gait,multisensor_gait,gait_analysis_assessment,benchmark_bilateral_lower_limb,mypredict",
        help="Comma-separated dataset names",
    )
    parser.add_argument(
        "--output-root",
        default="data/interoperable_gait",
        help="Root directory for the 3-level interoperability hub",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    wanted = [name.strip() for name in args.datasets.split(",") if name.strip()]
    rows = []
    for name in wanted:
        if name not in DATASETS:
            raise KeyError(f"Unknown dataset '{name}'")
        rows.append(build_dataset(DATASETS[name], output_root))

    summary_df = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_root / "datasets_summary.csv", index=False)
    print(output_root / "datasets_summary.csv")


if __name__ == "__main__":
    main()
