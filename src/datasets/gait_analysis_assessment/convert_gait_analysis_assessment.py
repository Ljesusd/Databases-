import argparse
from pathlib import Path
import re
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

from mh_toolbox.base.base_eurobench import DataBase  # noqa: E402
from mh_toolbox.base.eurobench_element import EurobenchElement  # noqa: E402


SUBJECT_FILE_RE = re.compile(r"^Subject(?P<subject>\d+)_(?P<run>\d+)\.(?P<ext>csv|c3d)$", re.IGNORECASE)


def _parse_subject_id(value: str) -> str:
    value = value.strip()
    match = re.fullmatch(r"Subject(?P<id>\d+)", value, flags=re.IGNORECASE)
    if match:
        return match.group("id").zfill(2)
    if re.fullmatch(r"\d+", value):
        return value.zfill(2)
    raise ValueError("Subject must look like '01' or 'Subject01'.")


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _to_float(value: str) -> float:
    value = value.strip()
    if not value:
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _sanitize_label(label: str) -> str:
    label = label.strip()
    if ":" in label:
        label = label.split(":")[-1]
    label = re.sub(r"\s+", "_", label)
    label = re.sub(r"[^A-Za-z0-9_]", "", label)
    return label


def discover_subject_ids(raw_root: Path) -> list[str]:
    subjects = set()
    for path in raw_root.glob("Subject*_*.csv"):
        match = SUBJECT_FILE_RE.match(path.name)
        if match:
            subjects.add(match.group("subject").zfill(2))
    return sorted(subjects)


def _find_line(lines: list[str], exact_text: str) -> int:
    for idx, line in enumerate(lines):
        if line.strip() == exact_text:
            return idx
    raise ValueError(f"Section '{exact_text}' not found")


def _build_unique_names(pairs: list[tuple[int, str]]) -> list[tuple[int, str]]:
    counts: dict[str, int] = {}
    out = []
    for idx, name in pairs:
        if name not in counts:
            counts[name] = 1
            out.append((idx, name))
            continue
        counts[name] += 1
        out.append((idx, f"{name}_{counts[name]}"))
    return out


def _parse_events(lines: list[str]) -> dict:
    events_idx = _find_line(lines, "Events")
    header_idx = None
    for idx in range(events_idx + 1, len(lines)):
        if lines[idx].startswith("Subject\tContext\tName\tTime (s)"):
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError("Events header not found")

    events = {
        "l_heel_strike": [],
        "l_toe_off": [],
        "r_heel_strike": [],
        "r_toe_off": [],
    }

    for line in lines[header_idx + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if "\t" not in line:
            break
        tokens = line.rstrip("\r\n").split("\t")
        if len(tokens) < 4:
            continue
        if not _is_number(tokens[3].strip()):
            continue
        time_val = float(tokens[3].strip())
        context = tokens[1].strip().lower()
        name = tokens[2].strip().lower()

        if "left" in context and "strike" in name:
            events["l_heel_strike"].append(time_val)
        elif "left" in context and "off" in name:
            events["l_toe_off"].append(time_val)
        elif "right" in context and "strike" in name:
            events["r_heel_strike"].append(time_val)
        elif "right" in context and "off" in name:
            events["r_toe_off"].append(time_val)

    clean_events = {}
    for key, vals in events.items():
        if vals:
            clean_events[key] = sorted(vals)
    return clean_events


def _parse_trajectories(lines: list[str]) -> tuple[pd.DataFrame, float]:
    traj_idx = _find_line(lines, "Trajectories")

    sample_rate = float(lines[traj_idx + 1].strip())

    header_idx = None
    for idx in range(traj_idx + 1, len(lines)):
        if lines[idx].startswith("Frame\tSub Frame\t"):
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError("Trajectories header not found")

    labels = lines[header_idx - 1].rstrip("\r\n").split("\t")
    headers = lines[header_idx].rstrip("\r\n").split("\t")

    cols: list[tuple[int, str]] = []
    current_marker = ""
    for idx, axis_raw in enumerate(headers):
        axis = axis_raw.strip()
        label = _sanitize_label(labels[idx] if idx < len(labels) else "")
        if idx < 2:
            continue
        if label:
            current_marker = label
        if axis in {"X", "Y", "Z"} and current_marker:
            cols.append((idx, f"{current_marker}_{axis.lower()}"))
        elif axis == "Count":
            cols.append((idx, "trajectory_count"))
    cols = _build_unique_names(cols)

    rows = []
    first_frame = None
    for line in lines[header_idx + 2 :]:
        stripped = line.strip()
        if not stripped:
            continue
        if "\t" not in line:
            break
        tokens = line.rstrip("\r\n").split("\t")
        if len(tokens) < len(headers):
            tokens += [""] * (len(headers) - len(tokens))
        frame_raw = tokens[0].strip()
        if not _is_number(frame_raw):
            continue
        frame = float(frame_raw)
        if first_frame is None:
            first_frame = frame

        row = {}
        for col_idx, col_name in cols:
            row[col_name] = _to_float(tokens[col_idx] if col_idx < len(tokens) else "")
        row["time"] = (frame - first_frame) / sample_rate
        rows.append(row)

    if not rows:
        raise ValueError("No trajectory rows found")

    return pd.DataFrame(rows), sample_rate


def convert_trial(raw_csv: Path, out_dir: Path) -> dict:
    lines = raw_csv.read_text(encoding="utf-8", errors="ignore").splitlines()
    trajectories_df, sample_rate = _parse_trajectories(lines)
    events = _parse_events(lines)

    match = SUBJECT_FILE_RE.match(raw_csv.name)
    if not match:
        raise ValueError(f"Unexpected filename: {raw_csv.name}")
    subject = match.group("subject").zfill(2)
    run = match.group("run").zfill(2)

    info = {
        "subject": subject,
        "condition": "Subject",
        "run": run,
        "feature": "Trajectories",
        "sample_rate": sample_rate,
        "source_file": str(raw_csv),
    }

    element = EurobenchElement(key_id=raw_csv.stem, load_files=False)
    element.add_info(info=info, load_file=False)
    element.add_events(events=events, key_events="point", load_file=False)
    element.add_data(data=DataBase(trajectories_df, time_key="time"), load_file=False)
    element.save_eurobench_elements(
        filename_out=raw_csv.name,
        dir_out=out_dir,
        info_suffix="info",
        events_suffix="gaitEvents",
        filename_suffix="Trajectories",
        writing_mode="w",
        save_events=True,
        save_info=True,
        save_data=True,
    )

    return {
        "trial": raw_csv.stem,
        "status": "ok",
        "n_samples": len(trajectories_df),
        "n_event_types": len(events),
    }


def convert_subject(raw_root: Path, eurobench_root: Path, subject_id: str) -> dict:
    subject_tag = f"Subject{subject_id}"
    out_dir = eurobench_root / subject_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(raw_root.glob(f"{subject_tag}_*.csv"))
    result: dict = {
        "subject": subject_tag,
        "n_csv_in": len(csv_files),
        "status": "ok",
        "error": "",
    }
    if not csv_files:
        result["status"] = "no_csv"
        return result

    errors = []
    ok_trials = 0
    for csv_file in csv_files:
        try:
            convert_trial(csv_file, out_dir)
            ok_trials += 1
        except Exception as exc:
            errors.append(f"{csv_file.name}: {exc}")

    if errors and ok_trials == 0:
        result["status"] = "error"
    elif errors:
        result["status"] = "partial"
    result["error"] = " | ".join(errors)
    result["n_trials_ok"] = ok_trials
    result["n_traj_out"] = len(list(out_dir.glob("*_Trajectories.csv")))
    result["n_events_out"] = len(list(out_dir.glob("*_point_gaitEvents.yaml")))
    result["n_info_out"] = len(list(out_dir.glob("*_info.yaml")))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        default="data/gait_analysis_assessment/raw",
        help="Folder with SubjectXX_YY raw files.",
    )
    parser.add_argument(
        "--eurobench-root",
        default="data/gait_analysis_assessment/eurobench",
        help="Output Eurobench root.",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Process one subject, e.g. '01' or 'Subject01'.",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    eurobench_root = Path(args.eurobench_root)
    eurobench_root.mkdir(parents=True, exist_ok=True)

    if args.subject:
        subject_ids = [_parse_subject_id(args.subject)]
    else:
        subject_ids = discover_subject_ids(raw_root)

    rows = []
    for subject_id in subject_ids:
        rows.append(convert_subject(raw_root, eurobench_root, subject_id))

    log_path = eurobench_root / "conversion_log.csv"
    pd.DataFrame(rows).to_csv(log_path, index=False)
    print(log_path)


if __name__ == "__main__":
    main()
