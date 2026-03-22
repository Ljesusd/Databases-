import argparse
import os
from pathlib import Path
import site
import sys


def _ensure_ezc3d_dylib_or_reexec():
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

from mh_toolbox.conversion.c3d.c3d_eurobench import (
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


# Filenames look like P01_S01_2minWalk_01.c3d.
# Keep the task name inside condition to avoid duplicating it in ProcessFilenames.
DEFAULT_PATTERN = r"(?P<subject>P\d+)_S(?P<condition>\d+_[^_]+)_(?P<run>\d+)"
DEFAULT_GROUPS = ["subject", "condition", "run"]


def convert_subject(raw_root: Path, eurobench_root: Path, subject_dir: Path, pattern: str):
    raw_data = subject_dir / "RAW_DATA"
    if not raw_data.exists():
        return {"subject": subject_dir.name, "status": "no_raw_data"}
    c3d_files = list(raw_data.glob("*.c3d"))
    if not c3d_files:
        return {"subject": subject_dir.name, "status": "no_c3d"}

    out_dir = eurobench_root / subject_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    convert_dir_c3d_to_eurobench_using_predefined_types(
        dir_in=str(raw_data),
        dir_out=str(out_dir),
        predefined_types="TRAJECTORY",
        b_save_data=True,
        b_save_analogs=True,
        b_save_events=True,
        b_save_info=True,
        pattern_c3d="*.c3d",
        pattern_subject_condition_run=pattern,
        group_names=DEFAULT_GROUPS,
        filename_suffix_data=None,
        filename_suffix_analog="analogs",
        info_suffix="info",
        events_suffix="gaitEvents",
        writing_mode="w",
        **TRAJECTORY_MARKER_STANDARDIZATION,
    )

    return {"subject": subject_dir.name, "status": "ok", "n_c3d": len(c3d_files)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        default="data/human_gait/raw/researchdata",
        help="Root with subject folders containing RAW_DATA",
    )
    parser.add_argument(
        "--eurobench-root",
        default="data/human_gait/eurobench",
        help="Output eurobench root",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="Regex with subject/condition/feature/run groups",
    )
    parser.add_argument("--subject", help="Process a single subject folder, e.g. P01_S01")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    eurobench_root = Path(args.eurobench_root)
    eurobench_root.mkdir(parents=True, exist_ok=True)

    if args.subject:
        subject_dirs = [raw_root / args.subject]
    else:
        subject_dirs = sorted(p for p in raw_root.glob("P*_S*") if p.is_dir())

    results = []
    for subj_dir in subject_dirs:
        results.append(convert_subject(raw_root, eurobench_root, subj_dir, args.pattern))

    # Save a simple log
    log_path = eurobench_root / "conversion_log.csv"
    import pandas as pd

    pd.DataFrame(results).to_csv(log_path, index=False)
    print(log_path)


if __name__ == "__main__":
    main()
