from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mh_toolbox.utils.yaml_utils import save_dict_as_yaml


def _find_local_extrema(signal: np.ndarray, kind: str) -> np.ndarray:
    if signal.size < 3:
        return np.array([], dtype=int)

    diff = np.diff(signal)
    if kind == "max":
        idx = np.where((diff[:-1] > 0) & (diff[1:] <= 0))[0] + 1
    elif kind == "min":
        idx = np.where((diff[:-1] < 0) & (diff[1:] >= 0))[0] + 1
    else:
        raise ValueError("kind must be 'max' or 'min'")
    return idx


def _enforce_min_interval(indices: np.ndarray, time: np.ndarray, min_interval_s: float) -> np.ndarray:
    if indices.size == 0:
        return indices
    sorted_idx = indices[np.argsort(time[indices])]
    kept = []
    last_t = -np.inf
    for idx in sorted_idx:
        if time[idx] - last_t >= min_interval_s:
            kept.append(idx)
            last_t = time[idx]
    return np.array(kept, dtype=int)


def detect_gait_events_markers(
    csv_path: str,
    side: str = "R",
    pelvis_marker: str = "SACR",
    heel_marker: str = "HEE",
    toe_marker: str = "TOE",
    axis_mode: str = "auto",
    axis_override: str | None = None,
    vertical_axis: str = "z",
    out_yaml: str | None = None,
):
    df = pd.read_csv(csv_path)
    time = df["time"].values

    pelvis_candidates = [pelvis_marker, "SACR", "RPSI", "LPSI", "RASI", "LASI"]
    pelvis_candidates = [m for i, m in enumerate(pelvis_candidates) if m not in pelvis_candidates[:i]]

    def pelvis_ref(axis: str) -> pd.Series:
        cols = [f"{marker}_{axis}" for marker in pelvis_candidates if f"{marker}_{axis}" in df.columns]
        if not cols:
            raise KeyError(
                f"No pelvis marker columns found for axis '{axis}' in {csv_path}. "
                f"Tried: {', '.join(pelvis_candidates)}"
            )
        if len(cols) == 1:
            return df[cols[0]]
        return df[cols].mean(axis=1)

    def require_column(name: str) -> pd.Series:
        if name not in df.columns:
            raise KeyError(f"Missing column '{name}' in {csv_path}")
        return df[name]

    axes = ["x", "y", "z"]
    if axis_override:
        axis = axis_override
    elif axis_mode == "vertical":
        axis = vertical_axis
    else:
        heel_ranges = {}
        for ax in axes:
            heel_rel = require_column(f"{side}{heel_marker}_{ax}") - pelvis_ref(ax)
            heel_ranges[ax] = heel_rel.max() - heel_rel.min()
        axis = max(heel_ranges, key=heel_ranges.get)

    heel_rel = require_column(f"{side}{heel_marker}_{axis}") - pelvis_ref(axis)
    toe_rel = require_column(f"{side}{toe_marker}_{axis}") - pelvis_ref(axis)

    if axis_mode == "vertical":
        hs_idx = _find_local_extrema(heel_rel.values, "min")
        to_idx = _find_local_extrema(toe_rel.values, "min")

        if hs_idx.size:
            hs_thresh = np.percentile(heel_rel.values, 30)
            hs_idx = hs_idx[heel_rel.values[hs_idx] <= hs_thresh]
            hs_idx = _enforce_min_interval(hs_idx, time, min_interval_s=0.3)

        if to_idx.size:
            to_thresh = np.percentile(toe_rel.values, 30)
            to_idx = to_idx[toe_rel.values[to_idx] <= to_thresh]
            to_idx = _enforce_min_interval(to_idx, time, min_interval_s=0.3)
    else:
        hs_idx = _find_local_extrema(heel_rel.values, "max")
        to_idx = _find_local_extrema(toe_rel.values, "min")

        if hs_idx.size:
            hs_thresh = np.percentile(heel_rel.values, 70)
            hs_idx = hs_idx[heel_rel.values[hs_idx] >= hs_thresh]
            hs_idx = _enforce_min_interval(hs_idx, time, min_interval_s=0.3)

        if to_idx.size:
            to_thresh = np.percentile(toe_rel.values, 30)
            to_idx = to_idx[toe_rel.values[to_idx] <= to_thresh]
            to_idx = _enforce_min_interval(to_idx, time, min_interval_s=0.3)

    events = {
        f"{side.lower()}_heel_strike": time[hs_idx].tolist(),
        f"{side.lower()}_toe_off": time[to_idx].tolist(),
    }

    if out_yaml:
        out_path = Path(out_yaml)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict_as_yaml(events, out_path, writing_mode="w")

    return events, axis


def main():
    csv_path = "data/_test_eurobench/SUBJ1 (0)_Trajectories.csv"
    out_yaml = "data/_test_eurobench/SUBJ1 (0)_gaitEvents.yaml"
    events, axis = detect_gait_events_markers(csv_path, out_yaml=out_yaml)
    print("Axis:", axis)
    print(events)


if __name__ == "__main__":
    main()
