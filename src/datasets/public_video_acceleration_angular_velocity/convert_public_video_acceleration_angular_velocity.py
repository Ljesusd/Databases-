import argparse
import re
import shutil
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mh_toolbox.conversion.common import format_subject_info_name, format_trial_basename, write_semicolon_csv, write_yaml


DATASET_ROOT = Path("data/A public dataset of video, acceleration, angular velocity")
IMU_ROOT = DATASET_ROOT / "IMU" / "IMU"
VIDEO_ROOT = DATASET_ROOT / "Videos" / "Videos"
METADATA_CSV = DATASET_ROOT / "PDFEinfo.csv"


IMU_COLUMN_MAP = {
    "Frame #": "frame",
    "Time [s]": "time",
    "ACC ML [g]": "acc_ml_g",
    "ACC AP [g]": "acc_ap_g",
    "ACC SI [g]": "acc_si_g",
    "GYR ML [deg/s]": "gyr_ml_deg_s",
    "GYR AP [deg/s]": "gyr_ap_deg_s",
    "GYR SI [deg/s]": "gyr_si_deg_s",
    "Freezing event [flag]": "freezing_event_flag",
}


def _sanitize_key(value: str) -> str:
    clean = value.replace("\n", " ").replace("\r", " ").strip().lower()
    clean = re.sub(r"[^a-z0-9]+", "_", clean)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean


def _clean_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    return value


def _load_metadata(path: Path) -> tuple[pd.DataFrame, dict[str, dict]]:
    df = pd.read_csv(path, sep=";", encoding="latin1")
    records: dict[str, dict] = {}
    for _, row in df.iterrows():
        row_dict = {str(col): _clean_value(row[col]) for col in df.columns}
        subject_id = str(row_dict.get("ID") or "").strip()
        match = re.search(r"(\d+)$", subject_id)
        if not match:
            continue
        num = match.group(1).zfill(2)
        records[num] = row_dict
    return df, records


def _copy_raw(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _load_imu_txt(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", encoding="latin1")
    df = df.rename(columns=IMU_COLUMN_MAP)
    required = [col for col in IMU_COLUMN_MAP.values() if col != "freezing_event_flag"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing IMU columns in {path.name}: {missing}")
    if "freezing_event_flag" not in df.columns:
        df["freezing_event_flag"] = None
    ordered_cols = [IMU_COLUMN_MAP[src] for src in IMU_COLUMN_MAP]
    df = df[ordered_cols].copy()
    return df


def _sample_rate_hz(df: pd.DataFrame) -> float | None:
    if "time" not in df.columns or len(df) < 2:
        return None
    dt = pd.to_numeric(df["time"], errors="coerce").diff().dropna()
    dt = dt[dt > 0]
    if dt.empty:
        return None
    return float(1.0 / dt.median())


def _duration_s(df: pd.DataFrame) -> float | None:
    if "time" not in df.columns or df.empty:
        return None
    time = pd.to_numeric(df["time"], errors="coerce").dropna()
    if time.empty:
        return None
    return float(time.iloc[-1] - time.iloc[0])


def _metadata_payload(raw_row: dict | None) -> dict:
    if raw_row is None:
        return {}
    payload = {}
    for key, value in raw_row.items():
        payload[_sanitize_key(str(key))] = _clean_value(value)
    return payload


def _subject_info_payload(pdfe_id: str, sub_id: str, metadata_row: dict | None) -> dict:
    meta = _metadata_payload(metadata_row)
    info = {
        "subject_id": pdfe_id,
        "source_ids": {
            "imu_subject": sub_id,
            "video_subject": pdfe_id,
            "metadata_subject": pdfe_id if metadata_row is not None else None,
        },
        "modalities": ["imu", "video"],
        "dataset_name": "A public dataset of video, acceleration, angular velocity",
        "metadata": meta,
    }
    return info


def _trial_condition_and_run(run_label: str) -> tuple[str, str]:
    if run_label == "standing":
        return "standing", "001"
    return "walk", str(run_label).zfill(3)


def _testbed_payload(
    pdfe_id: str,
    condition: str,
    run: str,
    imu_df: pd.DataFrame | None,
    raw_outputs: list[str],
    processed_outputs: list[str],
    video_available: bool,
    notes: list[str] | None = None,
) -> dict:
    payload = {
        "subject_id": pdfe_id,
        "condition": condition,
        "run": run,
        "dataset_name": "A public dataset of video, acceleration, angular velocity",
        "task": condition,
        "modalities": {
            "imu": imu_df is not None,
            "video": video_available,
        },
        "raw_outputs": raw_outputs,
        "processed_outputs": processed_outputs,
        "notes": notes or [],
    }
    if imu_df is not None:
        payload["imu"] = {
            "sample_rate_hz": _sample_rate_hz(imu_df),
            "duration_s": _duration_s(imu_df),
            "channels": [col for col in imu_df.columns if col not in {"frame", "time"}],
            "units": {
                "acc": "g",
                "gyr": "deg/s",
            },
            "freezing_event_flag_available": "freezing_event_flag" in imu_df.columns,
        }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert A public dataset of video, acceleration, angular velocity into a EUROBENCH-style layout."
    )
    parser.add_argument(
        "--dataset-root",
        default=str(DATASET_ROOT),
        help="Root folder of the dataset.",
    )
    parser.add_argument(
        "--eurobench-root",
        default=str(DATASET_ROOT / "eurobench"),
        help="Output folder for EUROBENCH-style files.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    eurobench_root = Path(args.eurobench_root)
    raw_root = eurobench_root / "raw_data"
    imu_root = dataset_root / "IMU" / "IMU"
    video_root = dataset_root / "Videos" / "Videos"
    metadata_csv = dataset_root / "PDFEinfo.csv"

    eurobench_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    metadata_df, metadata_by_num = _load_metadata(metadata_csv)
    _copy_raw(metadata_csv, raw_root / "PDFEinfo.csv")

    imu_txt_by_key = {}
    imu_bin_by_key = {}
    video_by_key = {}
    subject_nums = set()

    for txt_path in sorted(imu_root.glob("*.txt")):
        subj, run_label = txt_path.stem.split("_", 1)
        num = subj.replace("SUB", "").zfill(2)
        imu_txt_by_key[(num, run_label)] = txt_path
        subject_nums.add(num)

    for csv_path in sorted(imu_root.glob("*.csv")):
        subj, run_label = csv_path.stem.split("_", 1)
        num = subj.replace("SUB", "").zfill(2)
        imu_bin_by_key[(num, run_label)] = csv_path
        subject_nums.add(num)

    for mp4_path in sorted(video_root.glob("*.mp4")):
        subj, run_label = mp4_path.stem.split("_", 1)
        num = subj.replace("PDFE", "").zfill(2)
        video_by_key[(num, run_label)] = mp4_path
        subject_nums.add(num)

    subject_nums |= set(metadata_by_num.keys())

    counts = {
        "subject_info": 0,
        "processed_imu": 0,
        "testbedLabel": 0,
        "raw_imu_txt": 0,
        "raw_imu_spreadsheet": 0,
        "raw_video": 0,
    }
    log_rows: list[list[object]] = []
    mapping_rows: list[dict[str, object]] = []

    present_subjects = sorted(
        num for num in subject_nums if any(k[0] == num for k in imu_txt_by_key) or any(k[0] == num for k in video_by_key)
    )

    for num in present_subjects:
        pdfe_id = metadata_by_num.get(num, {}).get("ID") if num in metadata_by_num else None
        pdfe_id = str(pdfe_id).strip() if pdfe_id else f"PDFE{num}"
        sub_id = f"SUB{num}"

        subject_info = _subject_info_payload(pdfe_id, sub_id, metadata_by_num.get(num))
        info_path = eurobench_root / format_subject_info_name(pdfe_id)
        write_yaml(subject_info, info_path)
        counts["subject_info"] += 1
        log_rows.append(["subject_info", sub_id, info_path.name, "", "ok"])

        run_labels = sorted(
            {run for subj_num, run in imu_txt_by_key.keys() if subj_num == num}
            | {run for subj_num, run in video_by_key.keys() if subj_num == num}
        )

        for run_label in run_labels:
            condition, run = _trial_condition_and_run(run_label)
            base_name = format_trial_basename(pdfe_id, condition, run)
            raw_outputs: list[str] = []
            processed_outputs: list[str] = []
            notes: list[str] = []

            imu_txt = imu_txt_by_key.get((num, run_label))
            imu_bin = imu_bin_by_key.get((num, run_label))
            video = video_by_key.get((num, run_label))

            imu_df = None
            if imu_txt is not None:
                imu_df = _load_imu_txt(imu_txt)
                imu_csv_out = eurobench_root / f"{base_name}_imu.csv"
                write_semicolon_csv(
                    list(imu_df.columns),
                    imu_df.values.tolist(),
                    imu_csv_out,
                )
                processed_outputs.append(imu_csv_out.name)
                counts["processed_imu"] += 1
                log_rows.append(["processed_imu", str(imu_txt), imu_csv_out.name, "", "ok"])

                raw_txt_out = raw_root / f"{base_name}_imu.txt"
                _copy_raw(imu_txt, raw_txt_out)
                raw_outputs.append(raw_txt_out.relative_to(eurobench_root).as_posix())
                counts["raw_imu_txt"] += 1
                log_rows.append(["raw_imu_txt", str(imu_txt), raw_txt_out.relative_to(eurobench_root).as_posix(), "", "ok"])
            else:
                notes.append("missing_imu_txt")

            if imu_bin is not None:
                raw_bin_out = raw_root / f"{base_name}_imu_spreadsheet.csv"
                _copy_raw(imu_bin, raw_bin_out)
                raw_outputs.append(raw_bin_out.relative_to(eurobench_root).as_posix())
                counts["raw_imu_spreadsheet"] += 1
                log_rows.append(["raw_imu_spreadsheet", str(imu_bin), raw_bin_out.relative_to(eurobench_root).as_posix(), "", "ok"])
            else:
                notes.append("missing_imu_spreadsheet")

            if video is not None:
                raw_video_out = raw_root / f"{base_name}_video.mp4"
                _copy_raw(video, raw_video_out)
                raw_outputs.append(raw_video_out.relative_to(eurobench_root).as_posix())
                counts["raw_video"] += 1
                log_rows.append(["raw_video", str(video), raw_video_out.relative_to(eurobench_root).as_posix(), "", "ok"])
            elif condition != "standing":
                notes.append("missing_video")

            testbed = _testbed_payload(
                pdfe_id=pdfe_id,
                condition=condition,
                run=run,
                imu_df=imu_df,
                raw_outputs=raw_outputs,
                processed_outputs=processed_outputs,
                video_available=video is not None,
                notes=notes,
            )
            testbed_path = eurobench_root / f"{base_name}_testbedLabel.yaml"
            write_yaml(testbed, testbed_path)
            counts["testbedLabel"] += 1
            log_rows.append(["testbedLabel", base_name, testbed_path.name, "", "ok"])

            mapping_rows.append(
                {
                    "subject_num": num,
                    "subject_id": pdfe_id,
                    "imu_subject_id": sub_id,
                    "run_label_source": run_label,
                    "condition": condition,
                    "run": run,
                    "imu_txt_exists": imu_txt is not None,
                    "imu_spreadsheet_exists": imu_bin is not None,
                    "video_exists": video is not None,
                    "metadata_exists": num in metadata_by_num,
                    "processed_imu_csv": f"{base_name}_imu.csv" if imu_txt is not None else None,
                    "testbed_label": testbed_path.name,
                    "notes": ",".join(notes) if notes else "",
                }
            )

    mapping_df = pd.DataFrame(mapping_rows).sort_values(["subject_num", "condition", "run"])
    mapping_path = eurobench_root / "subject_trial_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False)

    log_path = eurobench_root / "conversion_log.csv"
    pd.DataFrame(log_rows, columns=["kind", "source", "output", "detail", "status"]).to_csv(log_path, index=False)

    summary = {
        "dataset_root": str(dataset_root),
        "eurobench_root": str(eurobench_root),
        "n_metadata_rows": int(len(metadata_df)),
        "n_subjects_with_files": int(len(present_subjects)),
        "counts": counts,
        "mapping_summary_file": str(mapping_path),
        "notes": [
            "IMU text files were converted into semicolon-delimited *_imu.csv files.",
            "Original binary spreadsheet files (misnamed .csv) were copied to raw_data as *_imu_spreadsheet.csv.",
            "Videos without matching IMU (e.g. PDFE10_3, PDFE30_3) were still copied to raw_data and recorded in subject_trial_mapping.csv.",
        ],
    }
    write_yaml(summary, eurobench_root / "conversion_summary.yaml")

    print(f"subjects={len(present_subjects)}")
    print(f"processed_imu={counts['processed_imu']}")
    print(f"testbed_labels={counts['testbedLabel']}")
    print(f"raw_video={counts['raw_video']}")
    print(f"mapping={mapping_path}")


if __name__ == "__main__":
    main()
