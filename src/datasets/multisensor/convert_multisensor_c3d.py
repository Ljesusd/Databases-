import argparse
import os
from pathlib import Path
import site
import sys

import numpy as np
import pandas as pd


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


def _get_marker(points: np.ndarray, labels: list[str], name: str) -> np.ndarray:
    if name not in labels:
        raise KeyError(f"Missing marker '{name}' in C3D")
    idx = labels.index(name)
    return points[:3, idx, :].T


def _mean_markers(*markers: np.ndarray) -> np.ndarray:
    return np.mean(np.stack(markers, axis=0), axis=0)


def convert_c3d_to_marker_csv(c3d_path: Path, out_csv: Path):
    import ezc3d

    c3d = ezc3d.c3d(str(c3d_path))
    labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
    points = c3d["data"]["points"]
    n_frames = points.shape[2]
    rate = float(c3d["parameters"]["POINT"]["RATE"]["value"][0])
    time = np.arange(n_frames) / rate

    ias_r = _get_marker(points, labels, "ias_R")
    ias_l = _get_marker(points, labels, "ias_L")
    ips_r = _get_marker(points, labels, "ips_R")
    ips_l = _get_marker(points, labels, "ips_L")
    fme_r = _get_marker(points, labels, "fme_R")
    fle_r = _get_marker(points, labels, "fle_R")
    tam_r = _get_marker(points, labels, "tam_R")
    fal_r = _get_marker(points, labels, "fal_R")
    fm1_r = _get_marker(points, labels, "fm1_R")
    fm5_r = _get_marker(points, labels, "fm5_R")
    heel_r = _get_marker(points, labels, "heel_R")

    pelvis = _mean_markers(ias_r, ias_l, ips_r, ips_l)
    rthigh = _mean_markers(ias_r, ips_r)
    rknee = _mean_markers(fme_r, fle_r)
    rank = _mean_markers(tam_r, fal_r)
    rtoe = _mean_markers(fm1_r, fm5_r)

    df = pd.DataFrame(
        {
            "time": time,
            "PELV_x": pelvis[:, 0],
            "PELV_y": pelvis[:, 1],
            "PELV_z": pelvis[:, 2],
            "RTHI_x": rthigh[:, 0],
            "RTHI_y": rthigh[:, 1],
            "RTHI_z": rthigh[:, 2],
            "RKNE_x": rknee[:, 0],
            "RKNE_y": rknee[:, 1],
            "RKNE_z": rknee[:, 2],
            "RANK_x": rank[:, 0],
            "RANK_y": rank[:, 1],
            "RANK_z": rank[:, 2],
            "RHEE_x": heel_r[:, 0],
            "RHEE_y": heel_r[:, 1],
            "RHEE_z": heel_r[:, 2],
            "RTOE_x": rtoe[:, 0],
            "RTOE_y": rtoe[:, 1],
            "RTOE_z": rtoe[:, 2],
        }
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--c3d-path",
        required=True,
        help="Path to a multisensor C3D file",
    )
    parser.add_argument(
        "--out-csv",
        required=True,
        help="Output CSV path",
    )
    args = parser.parse_args()

    convert_c3d_to_marker_csv(Path(args.c3d_path), Path(args.out_csv))
    print(args.out_csv)


if __name__ == "__main__":
    _ensure_ezc3d_dylib_or_reexec()
    main()
