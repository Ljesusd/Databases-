import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mh_toolbox.conversion.json import JsonTrajectorySpec, convert_json_payload_to_eurobench_dataframe


DATASET_TAG = "running_injury_clinic_kinematic"
MODE_CONFIG = {
    "walking": {
        "condition": "WALK",
        "hz_key": "hz_w",
        "meta_csv": "walk_data_meta_healthy.csv",
        "speed_col": "speed_w",
    },
    "running": {
        "condition": "RUN",
        "hz_key": "hz_r",
        "meta_csv": "run_data_meta_healthy.csv",
        "speed_col": "speed_r",
    },
}
MODE_TRAJECTORY_SPEC: dict[str, JsonTrajectorySpec] = {
    "walking": JsonTrajectorySpec(marker_path="walking", sample_rate_path="hz_w"),
    "running": JsonTrajectorySpec(marker_path="running", sample_rate_path="hz_r"),
}


def _sanitize_key(key: str) -> str:
    key = re.sub(r"[^0-9a-zA-Z]+", "_", str(key)).strip("_").lower()
    return key or "field"


def _to_builtin(value):
    if isinstance(value, np.generic):
        value = value.item()
    if pd.isna(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        return value if value else None
    return value


def _load_mode_metadata(healthy_root: Path, mode: str) -> pd.DataFrame:
    csv_name = MODE_CONFIG[mode]["meta_csv"]
    csv_path = healthy_root / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing healthy metadata file: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"sub_id", "filename", "datestring"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")
    if df["filename"].duplicated().any():
        raise ValueError(f"{csv_path} has duplicated filename rows.")

    df = df.copy()
    df["sub_id"] = pd.to_numeric(df["sub_id"], errors="coerce").astype("Int64")
    df = df.sort_values(["sub_id", "datestring", "filename"], kind="mergesort").reset_index(drop=True)
    df["run_idx"] = df.groupby("sub_id").cumcount() + 1
    df["run"] = df["run_idx"].map(lambda x: f"{int(x):03d}")
    return df


def _build_metadata_maps(
    healthy_root: Path,
) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for mode in ["walking", "running"]:
        df = _load_mode_metadata(healthy_root=healthy_root, mode=mode)
        row_map = {str(row["filename"]): row.to_dict() for _, row in df.iterrows()}
        out[mode] = row_map
    return out


def _build_info_payload(
    row: dict,
    source_json: Path,
    mode: str,
    run: str,
    n_frames: int,
    sample_rate: float,
    labels: list[str],
) -> dict:
    conf = MODE_CONFIG[mode]
    speed_col = conf["speed_col"]

    sub_id = int(row["sub_id"])
    info = {
        "dataset": DATASET_TAG,
        "subject": f"Subject{sub_id}",
        "condition": conf["condition"],
        "run": str(run),
        "feature": "Trajectories",
        "sample_rate": float(sample_rate),
        "n_frames": int(n_frames),
        "units": "mm",
        "labels": labels,
        "n_markers": int(len(labels)),
        "mode": mode,
        "speed_mps": _to_builtin(row.get(speed_col)),
        "collection_datetime": _to_builtin(row.get("datestring")),
        "source_file": str(source_json.resolve()),
        "source_filename": source_json.name,
    }

    # Keep original metadata fields with normalized names.
    for k, v in row.items():
        key = f"meta_{_sanitize_key(k)}"
        info[key] = _to_builtin(v)
    return info


def _mode_list(user_mode: str) -> list[str]:
    if user_mode == "both":
        return ["walking", "running"]
    if user_mode == "walk":
        return ["walking"]
    if user_mode == "run":
        return ["running"]
    raise ValueError(f"Unsupported mode selector: {user_mode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert healthy Running Injury Clinic JSON files to Eurobench trajectories."
    )
    parser.add_argument(
        "--healthy-root",
        default="data/running_injury_clinic_kinematic/healthy",
        help="Folder with healthy metadata CSV files and extracted JSON files.",
    )
    parser.add_argument(
        "--json-root",
        default="data/running_injury_clinic_kinematic/healthy/json_raw/reformat_data",
        help="Root folder with extracted JSON files (reformat_data/*/*.json).",
    )
    parser.add_argument(
        "--out-root",
        default="data/running_injury_clinic_kinematic/healthy/eurobench",
        help="Output folder for Eurobench files.",
    )
    parser.add_argument(
        "--mode",
        choices=["both", "walk", "run"],
        default="both",
        help="Convert walking, running, or both trajectories.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of JSON files (debug usage).",
    )
    args = parser.parse_args()

    healthy_root = Path(args.healthy_root)
    json_root = Path(args.json_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    metadata_maps = _build_metadata_maps(healthy_root=healthy_root)
    wanted_modes = _mode_list(args.mode)

    json_files = sorted(json_root.glob("*/*.json"))
    if args.limit is not None:
        json_files = json_files[: max(0, int(args.limit))]
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {json_root}")

    rows = []
    n_written = 0
    n_errors = 0
    n_skipped = 0

    for idx, json_path in enumerate(json_files, start=1):
        subject_folder = json_path.parent.name
        try:
            subject_id = int(subject_folder)
        except ValueError:
            subject_id = None

        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "json_file": str(json_path),
                    "mode": "n/a",
                    "status": "error",
                    "error": f"json_read_error: {exc}",
                    "subject": "",
                    "condition": "",
                    "run": "",
                    "n_frames": 0,
                    "n_markers": 0,
                    "invalid_markers": 0,
                    "sample_rate": np.nan,
                    "out_csv": "",
                    "out_info": "",
                }
            )
            n_errors += 1
            continue

        filename = json_path.name

        for mode in wanted_modes:
            conf = MODE_CONFIG[mode]
            row_meta = metadata_maps[mode].get(filename)
            if row_meta is None:
                rows.append(
                    {
                        "json_file": str(json_path),
                        "mode": mode,
                        "status": "skip_not_in_healthy_meta",
                        "error": "",
                        "subject": "",
                        "condition": conf["condition"],
                        "run": "",
                        "n_frames": 0,
                        "n_markers": 0,
                        "invalid_markers": 0,
                        "sample_rate": np.nan,
                        "out_csv": "",
                        "out_info": "",
                    }
                )
                n_skipped += 1
                continue

            hz_value = payload.get(conf["hz_key"])
            sample_rate_for_error = (
                float(hz_value) if isinstance(hz_value, (int, float)) and float(hz_value) > 0 else np.nan
            )
            try:
                df, conversion_meta = convert_json_payload_to_eurobench_dataframe(
                    payload=payload,
                    spec=MODE_TRAJECTORY_SPEC[mode],
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "json_file": str(json_path),
                        "mode": mode,
                        "status": "error",
                        "error": f"section_parse_error: {exc}",
                        "subject": "",
                        "condition": conf["condition"],
                        "run": "",
                        "n_frames": 0,
                        "n_markers": 0,
                        "invalid_markers": 0,
                        "sample_rate": sample_rate_for_error,
                        "out_csv": "",
                        "out_info": "",
                    }
                )
                n_errors += 1
                continue

            labels = conversion_meta["labels"]
            n_frames = int(conversion_meta["n_frames"])
            invalid_markers = int(conversion_meta["invalid_markers"])
            sample_rate = conversion_meta.get("sample_rate")
            if sample_rate is None:
                sample_rate = sample_rate_for_error

            meta_sub_id = _to_builtin(row_meta.get("sub_id"))
            if subject_id is None:
                subject_id = int(meta_sub_id)
            if meta_sub_id is not None and subject_id != int(meta_sub_id):
                rows.append(
                    {
                        "json_file": str(json_path),
                        "mode": mode,
                        "status": "error",
                        "error": f"subject_mismatch_json={subject_id}_meta={meta_sub_id}",
                        "subject": "",
                        "condition": conf["condition"],
                        "run": "",
                        "n_frames": 0,
                        "n_markers": 0,
                        "invalid_markers": invalid_markers,
                        "sample_rate": sample_rate,
                        "out_csv": "",
                        "out_info": "",
                    }
                )
                n_errors += 1
                continue

            run = str(row_meta["run"])
            subject_tag = f"Subject{int(subject_id)}"
            file_stem = f"{subject_tag}_{conf['condition']}_{run}"
            subject_out = out_root / subject_tag
            subject_out.mkdir(parents=True, exist_ok=True)

            out_csv = subject_out / f"{file_stem}_Trajectories.csv"
            out_info = subject_out / f"{file_stem}_info.yaml"

            df.to_csv(out_csv, index=False)
            info_payload = _build_info_payload(
                row=row_meta,
                source_json=json_path,
                mode=mode,
                run=run,
                n_frames=n_frames,
                sample_rate=float(sample_rate),
                labels=labels,
            )
            out_info.write_text(yaml.safe_dump(info_payload, sort_keys=False), encoding="utf-8")

            rows.append(
                {
                    "json_file": str(json_path),
                    "mode": mode,
                    "status": "ok",
                    "error": "",
                    "subject": subject_tag,
                    "condition": conf["condition"],
                    "run": run,
                    "n_frames": n_frames,
                    "n_markers": len(labels),
                    "invalid_markers": invalid_markers,
                    "sample_rate": float(sample_rate),
                    "out_csv": str(out_csv),
                    "out_info": str(out_info),
                }
            )
            n_written += 1

        if idx % 50 == 0:
            print(f"processed_json={idx}/{len(json_files)} written_trials={n_written} errors={n_errors}")

    log_df = pd.DataFrame(rows)
    log_csv = out_root / "conversion_log.csv"
    log_df.to_csv(log_csv, index=False)

    ok_df = log_df[log_df["status"] == "ok"] if not log_df.empty else pd.DataFrame()
    summary = {
        "json_files_seen": int(len(json_files)),
        "modes_requested": wanted_modes,
        "trials_written": int(n_written),
        "errors": int(n_errors),
        "skipped": int(n_skipped),
        "subjects_written": int(ok_df["subject"].nunique()) if not ok_df.empty else 0,
        "conditions_written": sorted(ok_df["condition"].dropna().unique().tolist()) if not ok_df.empty else [],
        "sample_rate_hz": {
            "min": float(ok_df["sample_rate"].min()) if not ok_df.empty else None,
            "max": float(ok_df["sample_rate"].max()) if not ok_df.empty else None,
        },
        "log_file": str(log_csv),
    }
    summary_yaml = out_root / "conversion_summary.yaml"
    summary_yaml.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")

    print(f"json_files_seen={summary['json_files_seen']}")
    print(f"trials_written={summary['trials_written']}")
    print(f"errors={summary['errors']}")
    print(f"skipped={summary['skipped']}")
    print(f"subjects_written={summary['subjects_written']}")
    print(f"log={log_csv}")
    print(f"summary={summary_yaml}")


if __name__ == "__main__":
    main()
