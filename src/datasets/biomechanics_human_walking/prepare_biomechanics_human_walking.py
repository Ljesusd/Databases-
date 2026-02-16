import argparse
import csv
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


TRIAL_RE = re.compile(
    r"^p(?P<subject>\d+)_trial(?P<trial>\d+)(?:_(?P<run>\d+))?\.c3d$",
    re.IGNORECASE,
)
STANDING_RE = re.compile(
    r"^p(?P<subject>\d+)_standing(?:_(?P<run>\d+))?\.c3d$",
    re.IGNORECASE,
)
SUBJECT_DIR_RE = re.compile(r"^p(?P<subject>\d+)_c3dfiles$", re.IGNORECASE)

NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "p": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def _ensure_7z_available() -> None:
    if shutil.which("7z") is None:
        raise RuntimeError("7z command not found. Install p7zip/7z to continue.")


def _extract_level1(source_root: Path, raw_root: Path) -> Path:
    zip_path = source_root / "Level 1 - C3D files.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing zip: {zip_path}")
    _ensure_7z_available()

    raw_root.mkdir(parents=True, exist_ok=True)
    cmd = ["7z", "x", "-y", str(zip_path), f"-o{raw_root}"]
    subprocess.run(cmd, check=True)

    extracted_dir = raw_root / "Level 1 - C3D files"
    if not extracted_dir.exists():
        raise FileNotFoundError(f"Expected extracted folder not found: {extracted_dir}")
    return extracted_dir


def _read_trial_lookup_xlsx(xlsx_path: Path) -> list[dict]:
    with zipfile.ZipFile(xlsx_path) as zf:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root_shared = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root_shared.findall("a:si", NS):
                text = "".join(node.text or "" for node in si.findall(".//a:t", NS))
                shared_strings.append(text)

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {
            rel.attrib["Id"]: rel.attrib["Target"] for rel in rels.findall("p:Relationship", NS)
        }

        first_sheet = workbook.find("a:sheets/a:sheet", NS)
        if first_sheet is None:
            return []
        rel_id = first_sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        if rel_id is None or rel_id not in rel_map:
            return []

        target = rel_map[rel_id]
        sheet_path = "xl/" + target if not target.startswith("xl/") else target
        worksheet = ET.fromstring(zf.read(sheet_path))

        rows: list[list[str]] = []
        for row in worksheet.findall("a:sheetData/a:row", NS):
            values: dict[int, str] = {}
            for cell in row.findall("a:c", NS):
                cell_ref = cell.attrib.get("r", "")
                col = _column_index(cell_ref)
                ctype = cell.attrib.get("t")
                value_node = cell.find("a:v", NS)
                inline_node = cell.find("a:is", NS)

                value = ""
                if ctype == "s" and value_node is not None and value_node.text is not None:
                    value = shared_strings[int(value_node.text)]
                elif ctype == "inlineStr" and inline_node is not None:
                    value = "".join(node.text or "" for node in inline_node.findall(".//a:t", NS))
                elif value_node is not None and value_node.text is not None:
                    value = value_node.text
                values[col] = value

            if values:
                last_col = max(values.keys())
                rows.append([values.get(i, "") for i in range(1, last_col + 1)])

        if not rows:
            return []

        headers = [(h or "").strip().lower().replace(" ", "_") for h in rows[0]]
        records: list[dict] = []
        for row in rows[1:]:
            if not any((c or "").strip() for c in row):
                continue
            rec = {}
            for idx, value in enumerate(row):
                key = headers[idx] if idx < len(headers) and headers[idx] else f"col_{idx+1}"
                rec[key] = value
            records.append(rec)
        return records


def _column_index(cell_ref: str) -> int:
    letters = []
    for ch in cell_ref:
        if "A" <= ch <= "Z":
            letters.append(ch)
        else:
            break
    val = 0
    for ch in letters:
        val = val * 26 + (ord(ch) - ord("A") + 1)
    return val


def _write_trial_lookup_csvs(source_root: Path, metadata_root: Path) -> tuple[Path, Path]:
    metadata_root.mkdir(parents=True, exist_ok=True)

    xlsx_path = source_root / "trial_look_up.xlsx"
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {xlsx_path}")

    xlsx_copy_path = metadata_root / "trial_look_up.xlsx"
    shutil.copy2(xlsx_path, xlsx_copy_path)

    records = _read_trial_lookup_xlsx(xlsx_path)
    out_csv = metadata_root / "trial_lookup.csv"
    out_cond_csv = metadata_root / "condition_map.csv"

    rows: list[dict] = []
    for rec in records:
        trial_raw = str(rec.get("trial_number", "")).strip()
        if not trial_raw:
            continue
        try:
            trial_num = int(float(trial_raw))
        except ValueError:
            continue

        speed_raw = str(rec.get("speed_(m/s)", "")).strip()
        speed_value = ""
        if speed_raw:
            try:
                speed_value = f"{float(speed_raw):.6g}"
            except ValueError:
                speed_value = speed_raw

        experiment = str(rec.get("walking_experiment", "")).strip()
        rows.append(
            {
                "trial_number": trial_num,
                "condition_code": f"C{trial_num:02d}",
                "speed_m_s": speed_value,
                "walking_experiment": experiment,
            }
        )

    rows = sorted(rows, key=lambda r: int(r["trial_number"]))

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["trial_number", "condition_code", "speed_m_s", "walking_experiment"],
        )
        writer.writeheader()
        writer.writerows(rows)

    with out_cond_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["condition_code", "trial_number", "speed_m_s", "walking_experiment"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "condition_code": row["condition_code"],
                    "trial_number": row["trial_number"],
                    "speed_m_s": row["speed_m_s"],
                    "walking_experiment": row["walking_experiment"],
                }
            )

    return out_csv, out_cond_csv


def _safe_unlink(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        raise RuntimeError(f"Path exists and is not a file/symlink: {path}")


def _materialize(source: Path, dest: Path, link_mode: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        _safe_unlink(dest)

    if link_mode == "symlink":
        dest.symlink_to(source.resolve())
    elif link_mode == "copy":
        shutil.copy2(source, dest)
    else:
        raise ValueError(f"Unsupported link mode: {link_mode}")


def _organize_standardized(
    extracted_level1_root: Path,
    standardized_root: Path,
    link_mode: str = "symlink",
) -> tuple[Path, dict]:
    if not extracted_level1_root.exists():
        raise FileNotFoundError(f"Missing extracted folder: {extracted_level1_root}")

    standardized_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    counters = {
        "subject_count": 0,
        "dynamic_count": 0,
        "standing_count": 0,
        "skipped_count": 0,
    }

    subject_dirs = sorted([p for p in extracted_level1_root.glob("p*_c3dfiles") if p.is_dir()])
    for subject_dir in subject_dirs:
        m_subj = SUBJECT_DIR_RE.match(subject_dir.name)
        if m_subj is None:
            counters["skipped_count"] += 1
            continue
        subject_int = int(m_subj.group("subject"))
        subject_code = f"{subject_int:02d}"
        counters["subject_count"] += 1

        dst_subject_dir = standardized_root / subject_code
        dst_subject_dir.mkdir(parents=True, exist_ok=True)

        for c3d in sorted(subject_dir.glob("*.c3d")):
            name = c3d.name
            m_trial = TRIAL_RE.match(name)
            m_stand = STANDING_RE.match(name)

            if m_trial:
                trial_num = int(m_trial.group("trial"))
                condition = f"C{trial_num:02d}"
                run_num = int(m_trial.group("run") or "1")
                run = f"{run_num:02d}"
                dst_name = f"{subject_code}_{condition}_{run}.c3d"
                dst_path = dst_subject_dir / dst_name
                _materialize(c3d, dst_path, link_mode=link_mode)
                counters["dynamic_count"] += 1
                summary_rows.append(
                    {
                        "subject": subject_code,
                        "source_file": str(c3d),
                        "standardized_file": str(dst_path),
                        "type": "dynamic",
                        "condition": condition,
                        "run": run,
                        "trial_number": trial_num,
                    }
                )
                continue

            if m_stand:
                run_num = int(m_stand.group("run") or "1")
                dst_name = f"{subject_code}_ST.c3d" if run_num == 1 else f"{subject_code}_ST{run_num:02d}.c3d"
                dst_path = dst_subject_dir / dst_name
                _materialize(c3d, dst_path, link_mode=link_mode)
                counters["standing_count"] += 1
                summary_rows.append(
                    {
                        "subject": subject_code,
                        "source_file": str(c3d),
                        "standardized_file": str(dst_path),
                        "type": "standing",
                        "condition": "ST",
                        "run": f"{run_num:02d}",
                        "trial_number": "",
                    }
                )
                continue

            counters["skipped_count"] += 1

    summary_csv = standardized_root.parent / "standardization_log.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "subject",
                "source_file",
                "standardized_file",
                "type",
                "condition",
                "run",
                "trial_number",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    return summary_csv, counters


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and organize 'A biomechanics dataset of healthy human walking' for Eurobench conversion."
    )
    parser.add_argument(
        "--source-root",
        default="/Volumes/Trabajo/A biomechanics dataset of healthy human walking",
        help="Folder containing Level 1/2/3 zip files and trial_look_up.xlsx.",
    )
    parser.add_argument(
        "--dataset-root",
        default="data/biomechanics_human_walking",
        help="Target dataset root under this repo.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["symlink", "copy"],
        default="symlink",
        help="How to populate standardized files.",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip zip extraction and reuse existing raw extraction.",
    )
    parser.add_argument(
        "--clean-standardized",
        action="store_true",
        help="Delete standardized folder before recreating it.",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root)
    dataset_root = Path(args.dataset_root)
    raw_root = dataset_root / "raw"
    metadata_root = dataset_root / "metadata"
    standardized_root = raw_root / "standardized"

    if args.clean_standardized and standardized_root.exists():
        shutil.rmtree(standardized_root)

    if args.skip_extract:
        extracted_level1_root = raw_root / "Level 1 - C3D files"
        if not extracted_level1_root.exists():
            raise FileNotFoundError(
                f"--skip-extract was set but extracted folder does not exist: {extracted_level1_root}"
            )
    else:
        extracted_level1_root = _extract_level1(source_root, raw_root)

    lookup_csv, condition_map_csv = _write_trial_lookup_csvs(source_root, metadata_root)
    standardization_log_csv, counters = _organize_standardized(
        extracted_level1_root=extracted_level1_root,
        standardized_root=standardized_root,
        link_mode=args.link_mode,
    )

    print(f"dataset_root={dataset_root}")
    print(f"extracted_level1={extracted_level1_root}")
    print(f"standardized_root={standardized_root}")
    print(f"trial_lookup_csv={lookup_csv}")
    print(f"condition_map_csv={condition_map_csv}")
    print(f"standardization_log_csv={standardization_log_csv}")
    print(
        "counts="
        f"subjects:{counters['subject_count']},"
        f"dynamic:{counters['dynamic_count']},"
        f"standing:{counters['standing_count']},"
        f"skipped:{counters['skipped_count']}"
    )


if __name__ == "__main__":
    main()
