import argparse
from io import TextIOWrapper
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd
import yaml


DATASET_TAG = "benchmark_datasets_for_bilateral_lower_limb"
TRIAL_RE = re.compile(r"^(?P<subject>AB\d+)_Circuit_(?P<run>\d+)_post\.csv$", re.IGNORECASE)
MODE_MAP = {
    0: "sitting",
    1: "level_walking",
    2: "ramp_ascent",
    3: "ramp_descent",
    4: "stair_ascent",
    5: "stair_descent",
    6: "standing",
}

WANTED_COLUMNS = {
    "Right_Ankle",
    "Right_Knee",
    "Left_Ankle",
    "Left_Knee",
    "Mode",
    "Right_Heel_Contact",
    "Right_Toe_Off",
    "Left_Heel_Contact",
    "Left_Toe_Off",
}

EVENT_COLUMN_MAP = {
    "Right_Heel_Contact": "r_heel_strike",
    "Right_Toe_Off": "r_toe_off",
    "Left_Heel_Contact": "l_heel_strike",
    "Left_Toe_Off": "l_toe_off",
}


def _sanitize_key(key: str) -> str:
    out = re.sub(r"[^0-9a-zA-Z]+", "_", str(key).strip().lower()).strip("_")
    return out or "field"


def _to_builtin(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        return value if value else None
    return value


def _compress_mode_sequence(mode_values: pd.Series) -> list[int]:
    mode_series = pd.to_numeric(mode_values, errors="coerce").dropna()
    if mode_series.empty:
        return []
    seq: list[int] = []
    for val in mode_series.astype(int).tolist():
        if not seq or seq[-1] != val:
            seq.append(val)
    return seq


def _extract_event_times(sample_index_values: pd.Series, n_samples: int, sample_rate_hz: float) -> list[float]:
    vals = pd.to_numeric(sample_index_values, errors="coerce").dropna()
    if vals.empty:
        return []

    cleaned_idx: list[int] = []
    for raw in vals.tolist():
        idx = int(round(float(raw)))
        if idx < 1:
            continue
        if idx > n_samples:
            if idx == n_samples + 1:
                idx = n_samples
            else:
                continue
        cleaned_idx.append(idx)

    if not cleaned_idx:
        return []

    unique_sorted = sorted(set(cleaned_idx))
    # Indices in source files are 1-based sample positions.
    return [round((i - 1) / float(sample_rate_hz), 6) for i in unique_sorted]


def _build_joint_angles(df: pd.DataFrame, sample_rate_hz: float) -> pd.DataFrame:
    n = len(df)
    out = pd.DataFrame()

    out["LHipAngles_x"] = np.zeros(n, dtype=float)
    out["LHipAngles_y"] = np.zeros(n, dtype=float)
    out["LHipAngles_z"] = np.zeros(n, dtype=float)
    out["LKneeAngles_x"] = pd.to_numeric(df["Left_Knee"], errors="coerce")
    out["LKneeAngles_y"] = np.zeros(n, dtype=float)
    out["LKneeAngles_z"] = np.zeros(n, dtype=float)
    out["LAnkleAngles_x"] = pd.to_numeric(df["Left_Ankle"], errors="coerce")
    out["LAnkleAngles_y"] = np.zeros(n, dtype=float)
    out["LAnkleAngles_z"] = np.zeros(n, dtype=float)

    out["RHipAngles_x"] = np.zeros(n, dtype=float)
    out["RHipAngles_y"] = np.zeros(n, dtype=float)
    out["RHipAngles_z"] = np.zeros(n, dtype=float)
    out["RKneeAngles_x"] = pd.to_numeric(df["Right_Knee"], errors="coerce")
    out["RKneeAngles_y"] = np.zeros(n, dtype=float)
    out["RKneeAngles_z"] = np.zeros(n, dtype=float)
    out["RAnkleAngles_x"] = pd.to_numeric(df["Right_Ankle"], errors="coerce")
    out["RAnkleAngles_y"] = np.zeros(n, dtype=float)
    out["RAnkleAngles_z"] = np.zeros(n, dtype=float)

    out["time"] = np.arange(n, dtype=float) / float(sample_rate_hz)
    return out


def _load_subject_metadata_from_dir(subject_dir: Path) -> dict[str, dict]:
    subject = subject_dir.name
    metadata_path = subject_dir / f"{subject}_Metadata.csv"
    if not metadata_path.exists():
        return {}

    df = pd.read_csv(metadata_path)
    if "Filename" not in df.columns:
        return {}

    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        key = str(row.get("Filename", "")).strip()
        if not key:
            continue
        out[key] = row.to_dict()
    return out


def _load_subject_metadata_from_zip(zf: zipfile.ZipFile, subject: str) -> dict[str, dict]:
    member = f"{subject}/{subject}_Metadata.csv"
    if member not in zf.namelist():
        return {}

    with zf.open(member, "r") as f:
        df = pd.read_csv(f)
    if "Filename" not in df.columns:
        return {}

    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        key = str(row.get("Filename", "")).strip()
        if not key:
            continue
        out[key] = row.to_dict()
    return out


def _read_trial_from_dir(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, usecols=lambda c: c in WANTED_COLUMNS)


def _read_trial_from_zip(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    with zf.open(member, "r") as f:
        return pd.read_csv(f, usecols=lambda c: c in WANTED_COLUMNS)


def _validate_columns(df: pd.DataFrame) -> list[str]:
    required = ["Right_Ankle", "Right_Knee", "Left_Ankle", "Left_Knee"]
    missing = [c for c in required if c not in df.columns]
    return missing


def _trial_files_from_dir(subject_dir: Path) -> list[Path]:
    processed = subject_dir / "Processed"
    files = sorted(processed.glob("*_post.csv"))

    def _k(p: Path):
        m = TRIAL_RE.match(p.name)
        if m:
            return int(m.group("run"))
        return 10**9

    return sorted(files, key=_k)


def _trial_members_from_zip(zf: zipfile.ZipFile, subject: str) -> list[str]:
    prefix = f"{subject}/Processed/"
    names = [n for n in zf.namelist() if n.startswith(prefix) and n.endswith("_post.csv")]

    def _k(name: str):
        m = TRIAL_RE.match(Path(name).name)
        if m:
            return int(m.group("run"))
        return 10**9

    return sorted(names, key=_k)


def _discover_sources(raw_root: Path) -> list[tuple[str, str, Path]]:
    dir_map: dict[str, Path] = {}
    zip_map: dict[str, Path] = {}

    for p in raw_root.glob("AB*"):
        if p.is_dir() and (p / "Processed").exists():
            dir_map[p.name] = p

    for p in raw_root.glob("AB*.zip"):
        zip_map[p.stem] = p

    subjects = sorted(set(dir_map.keys()) | set(zip_map.keys()), key=lambda s: int(re.search(r"\d+", s).group(0)))
    out: list[tuple[str, str, Path]] = []
    for subject in subjects:
        if subject in dir_map:
            out.append((subject, "dir", dir_map[subject]))
        else:
            out.append((subject, "zip", zip_map[subject]))
    return out


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")


def convert_subject_from_dir(
    subject: str,
    subject_dir: Path,
    out_root: Path,
    sample_rate_hz: float,
    dataset_id: str,
    overwrite: bool,
) -> list[dict]:
    out_dir = out_root / subject
    out_dir.mkdir(parents=True, exist_ok=True)

    trial_files = _trial_files_from_dir(subject_dir)
    metadata_by_trial = _load_subject_metadata_from_dir(subject_dir)

    rows: list[dict] = []
    for csv_path in trial_files:
        row = {
            "subject": subject,
            "trial": csv_path.name,
            "status": "ok",
            "error": "",
            "source_kind": "dir",
            "source_file": str(csv_path.resolve()),
            "n_samples": 0,
            "n_event_types": 0,
            "n_events_total": 0,
            "out_jointAngles": "",
            "out_events": "",
            "out_info": "",
        }

        try:
            m = TRIAL_RE.match(csv_path.name)
            if m is None:
                raise ValueError(f"Unexpected trial filename: {csv_path.name}")
            run = m.group("run")
            base = f"{subject}_Circuit_{run}"

            out_joint = out_dir / f"{base}_jointAngles.csv"
            out_events = out_dir / f"{base}_gaitEvents.yaml"
            out_info = out_dir / f"{base}_info.yaml"

            if not overwrite and out_joint.exists() and out_events.exists() and out_info.exists():
                row["status"] = "skip_exists"
                row["out_jointAngles"] = str(out_joint)
                row["out_events"] = str(out_events)
                row["out_info"] = str(out_info)
                rows.append(row)
                continue

            df = _read_trial_from_dir(csv_path)
            missing = _validate_columns(df)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            joint_df = _build_joint_angles(df=df, sample_rate_hz=sample_rate_hz)
            n_samples = len(joint_df)
            if n_samples < 2:
                raise ValueError("Trial has fewer than 2 samples")

            events: dict[str, list[float]] = {}
            for src_col, out_key in EVENT_COLUMN_MAP.items():
                if src_col not in df.columns:
                    continue
                ev = _extract_event_times(df[src_col], n_samples=n_samples, sample_rate_hz=sample_rate_hz)
                if ev:
                    events[out_key] = ev

            mode_seq_codes: list[int] = []
            mode_seq_labels: list[str] = []
            if "Mode" in df.columns:
                mode_seq_codes = _compress_mode_sequence(df["Mode"])
                mode_seq_labels = [MODE_MAP.get(v, f"mode_{v}") for v in mode_seq_codes]

            info = {
                "dataset": DATASET_TAG,
                "source_dataset_id": dataset_id,
                "subject": subject,
                "participant": subject,
                "condition": "Circuit",
                "run": str(run),
                "feature": "jointAngles",
                "sample_rate_hz": float(sample_rate_hz),
                "n_samples": int(n_samples),
                "time_start_s": 0.0,
                "time_end_s": round((n_samples - 1) / float(sample_rate_hz), 6),
                "angle_unit": "deg",
                "source_file": str(csv_path.resolve()),
                "source_type": "processed_csv",
                "mode_sequence_codes": mode_seq_codes,
                "mode_sequence_labels": mode_seq_labels,
                "mode_legend": {str(k): v for k, v in MODE_MAP.items()},
                "notes": "Hip and non-sagittal axes are unavailable in source data and were filled with 0.0.",
            }

            trial_meta = metadata_by_trial.get(base, {})
            for k, v in trial_meta.items():
                info[f"trial_meta_{_sanitize_key(k)}"] = _to_builtin(v)

            joint_df.to_csv(out_joint, index=False)
            _write_yaml(out_events, events)
            _write_yaml(out_info, info)

            row["n_samples"] = int(n_samples)
            row["n_event_types"] = int(len(events))
            row["n_events_total"] = int(sum(len(v) for v in events.values()))
            row["out_jointAngles"] = str(out_joint)
            row["out_events"] = str(out_events)
            row["out_info"] = str(out_info)
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            row["error"] = str(exc)

        rows.append(row)

    return rows


def convert_subject_from_zip(
    subject: str,
    zip_path: Path,
    out_root: Path,
    sample_rate_hz: float,
    dataset_id: str,
    overwrite: bool,
) -> list[dict]:
    out_dir = out_root / subject
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        trial_members = _trial_members_from_zip(zf, subject)
        metadata_by_trial = _load_subject_metadata_from_zip(zf, subject)

        for member in trial_members:
            trial_name = Path(member).name
            row = {
                "subject": subject,
                "trial": trial_name,
                "status": "ok",
                "error": "",
                "source_kind": "zip",
                "source_file": f"{zip_path.resolve()}::{member}",
                "n_samples": 0,
                "n_event_types": 0,
                "n_events_total": 0,
                "out_jointAngles": "",
                "out_events": "",
                "out_info": "",
            }

            try:
                m = TRIAL_RE.match(trial_name)
                if m is None:
                    raise ValueError(f"Unexpected trial filename: {trial_name}")
                run = m.group("run")
                base = f"{subject}_Circuit_{run}"

                out_joint = out_dir / f"{base}_jointAngles.csv"
                out_events = out_dir / f"{base}_gaitEvents.yaml"
                out_info = out_dir / f"{base}_info.yaml"

                if not overwrite and out_joint.exists() and out_events.exists() and out_info.exists():
                    row["status"] = "skip_exists"
                    row["out_jointAngles"] = str(out_joint)
                    row["out_events"] = str(out_events)
                    row["out_info"] = str(out_info)
                    rows.append(row)
                    continue

                df = _read_trial_from_zip(zf, member)
                missing = _validate_columns(df)
                if missing:
                    raise ValueError(f"Missing required columns: {missing}")

                joint_df = _build_joint_angles(df=df, sample_rate_hz=sample_rate_hz)
                n_samples = len(joint_df)
                if n_samples < 2:
                    raise ValueError("Trial has fewer than 2 samples")

                events: dict[str, list[float]] = {}
                for src_col, out_key in EVENT_COLUMN_MAP.items():
                    if src_col not in df.columns:
                        continue
                    ev = _extract_event_times(df[src_col], n_samples=n_samples, sample_rate_hz=sample_rate_hz)
                    if ev:
                        events[out_key] = ev

                mode_seq_codes: list[int] = []
                mode_seq_labels: list[str] = []
                if "Mode" in df.columns:
                    mode_seq_codes = _compress_mode_sequence(df["Mode"])
                    mode_seq_labels = [MODE_MAP.get(v, f"mode_{v}") for v in mode_seq_codes]

                info = {
                    "dataset": DATASET_TAG,
                    "source_dataset_id": dataset_id,
                    "subject": subject,
                    "participant": subject,
                    "condition": "Circuit",
                    "run": str(run),
                    "feature": "jointAngles",
                    "sample_rate_hz": float(sample_rate_hz),
                    "n_samples": int(n_samples),
                    "time_start_s": 0.0,
                    "time_end_s": round((n_samples - 1) / float(sample_rate_hz), 6),
                    "angle_unit": "deg",
                    "source_file": f"{zip_path.resolve()}::{member}",
                    "source_type": "processed_csv_inside_zip",
                    "mode_sequence_codes": mode_seq_codes,
                    "mode_sequence_labels": mode_seq_labels,
                    "mode_legend": {str(k): v for k, v in MODE_MAP.items()},
                    "notes": "Hip and non-sagittal axes are unavailable in source data and were filled with 0.0.",
                }

                trial_meta = metadata_by_trial.get(base, {})
                for k, v in trial_meta.items():
                    info[f"trial_meta_{_sanitize_key(k)}"] = _to_builtin(v)

                joint_df.to_csv(out_joint, index=False)
                _write_yaml(out_events, events)
                _write_yaml(out_info, info)

                row["n_samples"] = int(n_samples)
                row["n_event_types"] = int(len(events))
                row["n_events_total"] = int(sum(len(v) for v in events.values()))
                row["out_jointAngles"] = str(out_joint)
                row["out_events"] = str(out_events)
                row["out_info"] = str(out_info)
            except Exception as exc:  # noqa: BLE001
                row["status"] = "error"
                row["error"] = str(exc)

            rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert dataset 5362627 (benchmark bilateral lower limb) to Eurobench jointAngles/events/info files."
    )
    parser.add_argument(
        "--raw-root",
        default="data/benchmark_datasets_for_bilateral_lower_limb/raw/5362627",
        help="Root folder containing ABxxx subject folders and/or ABxxx.zip files.",
    )
    parser.add_argument(
        "--out-root",
        default="data/benchmark_datasets_for_bilateral_lower_limb/eurobench",
        help="Output Eurobench root folder.",
    )
    parser.add_argument(
        "--dataset-id",
        default="5362627",
        help="Dataset source identifier to store in info YAML.",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=1000.0,
        help="Sample rate (Hz) used to build time and convert gait event indices to seconds.",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Optional single subject filter (e.g., AB156).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    all_sources = _discover_sources(raw_root)
    if args.subject:
        target = args.subject.strip()
        all_sources = [row for row in all_sources if row[0].lower() == target.lower()]

    if not all_sources:
        raise FileNotFoundError(f"No subject sources found in {raw_root}")

    rows: list[dict] = []
    for subject, source_kind, source_path in all_sources:
        if source_kind == "dir":
            rows.extend(
                convert_subject_from_dir(
                    subject=subject,
                    subject_dir=source_path,
                    out_root=out_root,
                    sample_rate_hz=args.sample_rate_hz,
                    dataset_id=args.dataset_id,
                    overwrite=args.overwrite,
                )
            )
        else:
            rows.extend(
                convert_subject_from_zip(
                    subject=subject,
                    zip_path=source_path,
                    out_root=out_root,
                    sample_rate_hz=args.sample_rate_hz,
                    dataset_id=args.dataset_id,
                    overwrite=args.overwrite,
                )
            )

    log_df = pd.DataFrame(rows)
    log_path = out_root / "conversion_log.csv"
    log_df.to_csv(log_path, index=False)

    summary = {
        "subjects": len(sorted(set(log_df["subject"]))) if not log_df.empty else 0,
        "trials_total": int(len(log_df)),
        "trials_ok": int((log_df["status"] == "ok").sum()) if not log_df.empty else 0,
        "trials_error": int((log_df["status"] == "error").sum()) if not log_df.empty else 0,
        "trials_skip_exists": int((log_df["status"] == "skip_exists").sum()) if not log_df.empty else 0,
        "sample_rate_hz": float(args.sample_rate_hz),
        "log_file": str(log_path),
    }

    summary_path = out_root / "conversion_summary.yaml"
    _write_yaml(summary_path, summary)

    print(log_path)
    print(summary_path)


if __name__ == "__main__":
    main()
