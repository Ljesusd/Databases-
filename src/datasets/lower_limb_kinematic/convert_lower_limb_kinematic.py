import argparse
import os
from pathlib import Path
import re
import site
import sys
import tempfile

import pandas as pd
import yaml


def _ensure_ezc3d_dylib_or_reexec() -> None:
    if "DYLD_LIBRARY_PATH" in os.environ and "ezc3d" in os.environ["DYLD_LIBRARY_PATH"]:
        return
    if os.environ.get("EZC3D_DYLD_REEXEC") == "1":
        return
    candidates = [site.getusersitepackages()] + site.getsitepackages()
    for base in candidates:
        ez_dir = Path(base) / "ezc3d"
        if (ez_dir / "libezc3d.dylib").exists():
            current = os.environ.get("DYLD_LIBRARY_PATH", "")
            os.environ["DYLD_LIBRARY_PATH"] = f"{ez_dir}{os.pathsep}{current}" if current else str(ez_dir)
            os.environ["EZC3D_DYLD_REEXEC"] = "1"
            os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_ezc3d_dylib_or_reexec()

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

from mh_toolbox.conversion.c3d.c3d_eurobench import (  # noqa: E402
    convert_dir_c3d_to_eurobench_using_predefined_types,
)
try:
    from src.datasets.marker_standardization import (  # noqa: E402
        TRAJECTORY_MARKER_STANDARDIZATION,
    )
except ModuleNotFoundError:
    from datasets.marker_standardization import (  # noqa: E402
        TRAJECTORY_MARKER_STANDARDIZATION,
    )


PATTERN_SUBJECT_CONDITION_RUN = r"(?P<subject>Subject\d+)_(?P<condition>V\d+)_(?P<run>\d+)"
GROUPS = ["subject", "condition", "run"]

PARTICIPANT_RE = re.compile(r"^Participant(?P<id>\d+)$", re.IGNORECASE)
TRIAL_RE = re.compile(r"^T(?P<trial>\d+)\.c3d$", re.IGNORECASE)
SPEED_RE = re.compile(r"^V(?P<speed>\d+)$", re.IGNORECASE)
INFO_RE = re.compile(r"^(?P<subject>Subject\d+)_(?P<speed>V\d+)_(?P<run>\d+)_info\.yaml$", re.IGNORECASE)


def _speed_key(speed_name: str) -> tuple[int, str]:
    m = SPEED_RE.match(speed_name)
    if m:
        return int(m.group("speed")), speed_name
    return 10**9, speed_name


def _parse_number(text: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if m is None:
        return None
    return float(m.group(0))


def _parse_metadata(metadata_file: Path) -> dict:
    out: dict = {}
    if not metadata_file.exists():
        return out

    rows = {}
    for line in metadata_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        rows[key.strip().lower()] = value.strip()

    out["participant_original_id"] = rows.get("id")
    out["age_years"] = _parse_number(rows.get("age", ""))
    out["gender"] = rows.get("gender")
    out["body_height_m"] = _parse_number(rows.get("body height", ""))
    out["body_mass_kg"] = _parse_number(rows.get("body mass", ""))
    out["leg_length_m"] = _parse_number(rows.get("leg length", ""))
    out["foot_length_m"] = _parse_number(rows.get("foot length", ""))
    return {k: v for k, v in out.items() if v is not None and v != ""}


def _participant_tag(value: str) -> str:
    value = value.strip()
    if PARTICIPANT_RE.match(value):
        return f"Participant{int(PARTICIPANT_RE.match(value).group('id'))}"
    if re.fullmatch(r"\d+", value):
        return f"Participant{int(value)}"
    if re.fullmatch(r"Subject\d+", value, flags=re.IGNORECASE):
        digits = re.search(r"\d+", value).group(0)
        return f"Participant{int(digits)}"
    raise ValueError("Participant must look like 'Participant1', '1' or 'Subject1'.")


def _subject_from_participant(participant_tag: str) -> str:
    m = PARTICIPANT_RE.match(participant_tag)
    if m is None:
        raise ValueError(f"Invalid participant tag: {participant_tag}")
    return f"Subject{int(m.group('id')):02d}"


def _trial_number(c3d_file: Path) -> int:
    m = TRIAL_RE.match(c3d_file.name)
    if m is None:
        raise ValueError(f"Unexpected trial filename: {c3d_file.name}")
    return int(m.group("trial"))


def _discover_participants(raw_root: Path) -> list[Path]:
    participants = []
    for p in sorted(raw_root.glob("Participant*"), key=lambda x: int(re.search(r"\d+", x.name).group(0))):
        if p.is_dir() and (p / "Raw_Data").exists():
            participants.append(p)
    return participants


def _discover_speed_dirs(raw_participant_dir: Path, speed: str | None) -> list[Path]:
    speed_dirs = sorted([p for p in raw_participant_dir.glob("V*") if p.is_dir()], key=lambda p: _speed_key(p.name))
    if speed is None:
        return speed_dirs
    speed = speed.upper()
    return [p for p in speed_dirs if p.name.upper() == speed]


def _make_links_for_speed(speed_dir: Path, subject_tag: str) -> tuple[tempfile.TemporaryDirectory, list[dict]]:
    tmp_ctx = tempfile.TemporaryDirectory(prefix=f"{subject_tag}_{speed_dir.name}_")
    tmp_path = Path(tmp_ctx.name)
    manifest: list[dict] = []
    for c3d_file in sorted(speed_dir.glob("T*.c3d"), key=_trial_number):
        trial = _trial_number(c3d_file)
        run = f"{trial:02d}"
        link_name = f"{subject_tag}_{speed_dir.name}_{run}.c3d"
        link_path = tmp_path / link_name
        link_path.symlink_to(c3d_file.resolve())
        manifest.append(
            {
                "speed": speed_dir.name,
                "run": run,
                "trial": f"T{trial}",
                "source_c3d_file": str(c3d_file.resolve()),
                "link_filename": link_name,
            }
        )
    return tmp_ctx, manifest


def _convert_speed_to_eurobench(
    speed_dir: Path,
    out_dir: Path,
    subject_tag: str,
    save_analogs: bool,
) -> tuple[dict, list[dict]]:
    row = {
        "subject": subject_tag,
        "speed": speed_dir.name,
        "status": "ok",
        "error": "",
        "n_c3d_in": 0,
        "n_traj_out": 0,
        "n_events_out": 0,
        "n_info_out": 0,
        "n_analogs_out": 0,
    }

    tmp_ctx, manifest = _make_links_for_speed(speed_dir=speed_dir, subject_tag=subject_tag)
    row["n_c3d_in"] = len(manifest)
    if not manifest:
        row["status"] = "no_c3d"
        return row, manifest

    try:
        with tmp_ctx as tmp_name:
            convert_dir_c3d_to_eurobench_using_predefined_types(
                dir_in=str(tmp_name),
                dir_out=str(out_dir),
                predefined_types="TRAJECTORY",
                b_save_data=True,
                b_save_analogs=save_analogs,
                b_save_events=True,
                b_save_info=True,
                pattern_c3d="*.c3d",
                pattern_subject_condition_run=PATTERN_SUBJECT_CONDITION_RUN,
                group_names=GROUPS,
                filename_suffix_data=None,
                filename_suffix_analog="analogs",
                info_suffix="info",
                events_suffix="gaitEvents",
                writing_mode="w",
                **TRAJECTORY_MARKER_STANDARDIZATION,
            )
    except Exception as exc:  # noqa: BLE001
        row["status"] = "error"
        row["error"] = str(exc)

    row["n_traj_out"] = len(list(out_dir.glob(f"{subject_tag}_{speed_dir.name}_*_Trajectories.csv")))
    row["n_events_out"] = len(list(out_dir.glob(f"{subject_tag}_{speed_dir.name}_*_gaitEvents.yaml")))
    row["n_info_out"] = len(list(out_dir.glob(f"{subject_tag}_{speed_dir.name}_*_info.yaml")))
    row["n_analogs_out"] = len(list(out_dir.glob(f"{subject_tag}_{speed_dir.name}_*_analogs.csv")))
    return row, manifest


def _enrich_info_files(
    out_dir: Path,
    subject_tag: str,
    participant_tag: str,
    metadata: dict,
    source_by_speed_run: dict[tuple[str, str], str],
) -> int:
    updated = 0
    for info_path in sorted(out_dir.glob(f"{subject_tag}_*_info.yaml")):
        m = INFO_RE.match(info_path.name)
        if m is None:
            continue
        speed = m.group("speed")
        run = m.group("run")
        source_file = source_by_speed_run.get((speed, run))

        payload = yaml.safe_load(info_path.read_text(encoding="utf-8", errors="ignore")) or {}
        if not isinstance(payload, dict):
            payload = {}

        payload.setdefault("dataset", "lower_limb_kinematic")
        payload["participant"] = participant_tag
        payload["subject"] = subject_tag

        for key, value in metadata.items():
            payload[key] = value

        if source_file:
            old_source = payload.get("source_file")
            if old_source and old_source != source_file:
                payload["source_file_symlink"] = old_source
            payload["source_file"] = source_file

        info_path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
        updated += 1
    return updated


def _convert_participant(
    participant_dir: Path,
    ascii_root: Path,
    eurobench_root: Path,
    speed: str | None,
    save_analogs: bool,
) -> tuple[dict, list[dict]]:
    participant_tag = participant_dir.name
    subject_tag = _subject_from_participant(participant_tag)
    out_dir = eurobench_root / subject_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = ascii_root / participant_tag / "Metadata.txt"
    if not metadata_path.exists():
        metadata_path = participant_dir / "Metadata.txt"
    metadata = _parse_metadata(metadata_path)

    speed_rows: list[dict] = []
    source_by_speed_run: dict[tuple[str, str], str] = {}

    raw_data_dir = participant_dir / "Raw_Data"
    speed_dirs = _discover_speed_dirs(raw_data_dir, speed=speed)

    for speed_dir in speed_dirs:
        row, manifest = _convert_speed_to_eurobench(
            speed_dir=speed_dir,
            out_dir=out_dir,
            subject_tag=subject_tag,
            save_analogs=save_analogs,
        )
        speed_rows.append(row)
        for item in manifest:
            source_by_speed_run[(item["speed"], item["run"])] = item["source_c3d_file"]

    n_info_enriched = _enrich_info_files(
        out_dir=out_dir,
        subject_tag=subject_tag,
        participant_tag=participant_tag,
        metadata=metadata,
        source_by_speed_run=source_by_speed_run,
    )

    status = "ok"
    if speed_rows and any(r["status"] == "error" for r in speed_rows):
        status = "partial_error"
    elif not speed_rows:
        status = "no_speed_dirs"

    participant_row = {
        "participant": participant_tag,
        "subject": subject_tag,
        "status": status,
        "n_speeds_processed": len(speed_rows),
        "n_c3d_in": int(sum(r["n_c3d_in"] for r in speed_rows)),
        "n_traj_out": int(sum(r["n_traj_out"] for r in speed_rows)),
        "n_events_out": int(sum(r["n_events_out"] for r in speed_rows)),
        "n_info_out": int(sum(r["n_info_out"] for r in speed_rows)),
        "n_analogs_out": int(sum(r["n_analogs_out"] for r in speed_rows)),
        "n_info_enriched": n_info_enriched,
        "metadata_file": str(metadata_path) if metadata_path.exists() else "",
    }
    return participant_row, speed_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Lower limb kinematic C3D files to Eurobench format.")
    parser.add_argument(
        "--raw-root",
        default="data/lower_limb_kinematic/c3d files",
        help="Root containing Participant*/Raw_Data/V*/T*.c3d",
    )
    parser.add_argument(
        "--ascii-root",
        default="data/lower_limb_kinematic/ASCII files",
        help="Root containing Participant*/Metadata.txt",
    )
    parser.add_argument(
        "--eurobench-root",
        default="data/lower_limb_kinematic/eurobench",
        help="Output Eurobench root.",
    )
    parser.add_argument(
        "--participant",
        default=None,
        help="Optional participant selector (e.g., Participant1, 1, Subject1).",
    )
    parser.add_argument(
        "--speed",
        default=None,
        help="Optional speed selector (e.g., V3).",
    )
    parser.add_argument(
        "--save-analogs",
        action="store_true",
        help="Also save *_analogs.csv (large output).",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    ascii_root = Path(args.ascii_root)
    eurobench_root = Path(args.eurobench_root)
    eurobench_root.mkdir(parents=True, exist_ok=True)

    if args.speed is not None and SPEED_RE.match(args.speed.upper()) is None:
        raise ValueError("--speed must look like V1, V15, V25, V35, etc.")

    if args.participant:
        participant_dir = raw_root / _participant_tag(args.participant)
        participant_dirs = [participant_dir]
    else:
        participant_dirs = _discover_participants(raw_root)

    if not participant_dirs:
        raise FileNotFoundError(f"No participant folders found under {raw_root}")

    participant_rows = []
    speed_rows = []
    for participant_dir in participant_dirs:
        if not participant_dir.exists():
            participant_rows.append(
                {
                    "participant": participant_dir.name,
                    "subject": "",
                    "status": "missing_participant_dir",
                    "n_speeds_processed": 0,
                    "n_c3d_in": 0,
                    "n_traj_out": 0,
                    "n_events_out": 0,
                    "n_info_out": 0,
                    "n_analogs_out": 0,
                    "n_info_enriched": 0,
                    "metadata_file": "",
                }
            )
            continue

        participant_row, per_speed = _convert_participant(
            participant_dir=participant_dir,
            ascii_root=ascii_root,
            eurobench_root=eurobench_root,
            speed=args.speed.upper() if args.speed else None,
            save_analogs=args.save_analogs,
        )
        participant_rows.append(participant_row)
        speed_rows.extend(per_speed)

    participant_log = eurobench_root / "conversion_participant_log.csv"
    speed_log = eurobench_root / "conversion_log.csv"
    pd.DataFrame(participant_rows).to_csv(participant_log, index=False)
    pd.DataFrame(speed_rows).to_csv(speed_log, index=False)

    print(f"participants={len(participant_rows)}")
    print(f"speeds={len(speed_rows)}")
    print(f"log_participants={participant_log}")
    print(f"log_speeds={speed_log}")


if __name__ == "__main__":
    main()
