import argparse
import os
from pathlib import Path
import site
import sys
import tempfile

import pandas as pd


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


DYNAMIC_PATTERN = r"(?P<subject>\d+)_(?P<condition>C\d+)_(?P<run>\d+)"
STATIC_PATTERN = r"(?P<subject>\d+)_(?P<condition>ST)_(?P<run>\d+)"
GROUPS = ["subject", "condition", "run"]


def _convert_dir(
    dir_in: Path,
    dir_out: Path,
    pattern_c3d: str,
    pattern_subject_condition_run: str,
    writing_mode: str = "w",
    save_analogs: bool = False,
    save_info: bool = False,
) -> None:
    convert_dir_c3d_to_eurobench_using_predefined_types(
        dir_in=str(dir_in),
        dir_out=str(dir_out),
        predefined_types="TRAJECTORY",
        b_save_data=True,
        b_save_analogs=save_analogs,
        b_save_events=True,
        b_save_info=save_info,
        pattern_c3d=pattern_c3d,
        pattern_subject_condition_run=pattern_subject_condition_run,
        group_names=GROUPS,
        filename_suffix_data=None,
        filename_suffix_analog="analogs",
        info_suffix="info",
        events_suffix="gaitEvents",
        writing_mode=writing_mode,
    )


def _prepare_static_tmp_dir(subject_dir: Path) -> tuple[tempfile.TemporaryDirectory, int]:
    tmp = tempfile.TemporaryDirectory(prefix=f"{subject_dir.name}_st_", dir=str(subject_dir))
    tmp_path = Path(tmp.name)
    st_files = sorted(subject_dir.glob("*_ST.c3d"))
    for st in st_files:
        link_name = tmp_path / f"{st.stem}_00.c3d"
        link_name.symlink_to(st.resolve())
    return tmp, len(st_files)


def convert_subject(
    subject_dir: Path,
    eurobench_root: Path,
    include_static: bool = True,
    save_analogs: bool = False,
    save_info: bool = False,
) -> dict:
    out_dir = eurobench_root / subject_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    n_dynamic = len(list(subject_dir.glob("*_C*_*.c3d")))
    n_static = len(list(subject_dir.glob("*_ST.c3d")))

    result: dict = {
        "subject": subject_dir.name,
        "n_dynamic_c3d": n_dynamic,
        "n_static_c3d": n_static,
        "status": "ok",
        "error": "",
    }

    try:
        if n_dynamic > 0:
            _convert_dir(
                dir_in=subject_dir,
                dir_out=out_dir,
                pattern_c3d="*_C*_*.c3d",
                pattern_subject_condition_run=DYNAMIC_PATTERN,
                writing_mode="w",
                save_analogs=save_analogs,
                save_info=save_info,
            )

        if include_static and n_static > 0:
            tmp_ctx, _ = _prepare_static_tmp_dir(subject_dir)
            with tmp_ctx as tmp_name:
                _convert_dir(
                    dir_in=Path(tmp_name),
                    dir_out=out_dir,
                    pattern_c3d="*.c3d",
                    pattern_subject_condition_run=STATIC_PATTERN,
                    writing_mode="a",
                    save_analogs=save_analogs,
                    save_info=save_info,
                )
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)

    result["n_traj_out"] = len(list(out_dir.glob("*_Trajectories.csv")))
    result["n_events_out"] = len(list(out_dir.glob("*_gaitEvents.yaml")))
    result["n_info_out"] = len(list(out_dir.glob("*_info.yaml")))
    result["n_analogs_out"] = len(list(out_dir.glob("*_analogs.csv")))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        default="data/biomechanics_human_walking/raw/standardized",
        help="Root with standardized subject folders and C3D files.",
    )
    parser.add_argument(
        "--eurobench-root",
        default="data/biomechanics_human_walking/eurobench",
        help="Output Eurobench root.",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Process one subject folder (e.g., 01).",
    )
    parser.add_argument(
        "--no-static",
        action="store_true",
        help="Skip *_ST.c3d files.",
    )
    parser.add_argument(
        "--save-analogs",
        action="store_true",
        help="Save analog CSVs (large output).",
    )
    parser.add_argument(
        "--save-info",
        action="store_true",
        help="Save info YAML files.",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    eurobench_root = Path(args.eurobench_root)
    eurobench_root.mkdir(parents=True, exist_ok=True)

    if args.subject:
        subject_dirs = [raw_root / args.subject]
    else:
        subject_dirs = sorted(p for p in raw_root.glob("*") if p.is_dir() and p.name.isdigit())

    rows = []
    for subject_dir in subject_dirs:
        rows.append(
            convert_subject(
                subject_dir=subject_dir,
                eurobench_root=eurobench_root,
                include_static=not args.no_static,
                save_analogs=args.save_analogs,
                save_info=args.save_info,
            )
        )

    log_path = eurobench_root / "conversion_log.csv"
    pd.DataFrame(rows).to_csv(log_path, index=False)
    print(log_path)


if __name__ == "__main__":
    main()

