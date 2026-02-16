import argparse
import csv
from pathlib import Path
import sys
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "src"))

from detect_gait_events_markers import detect_gait_events_markers  # noqa: E402


def _as_sorted_floats(values: Any) -> list[float]:
    if values is None:
        return []
    if isinstance(values, list):
        out = [float(v) for v in values]
    else:
        out = [float(values)]
    return sorted(out)


def _pick_best_hs_pair(
    hs: list[float],
    to: list[float],
    t_min: float,
    t_max: float,
    min_stride_s: float,
    max_stride_s: float,
) -> tuple[float, float] | None:
    if len(hs) < 2:
        return None

    mid_t = 0.5 * (t_min + t_max)
    candidates: list[tuple[float, float, float]] = []
    for i in range(len(hs) - 1):
        hs1 = hs[i]
        hs2 = hs[i + 1]
        stride = hs2 - hs1
        if stride < min_stride_s or stride > max_stride_s:
            continue
        to_inside = [x for x in to if hs1 < x < hs2]
        if not to_inside:
            continue
        center = 0.5 * (hs1 + hs2)
        score = abs(center - mid_t)
        candidates.append((score, hs1, hs2))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1], candidates[0][2]


def _estimate_for_side(
    trajectories_csv: Path,
    side: str,
    heel_marker: str,
    toe_marker: str,
    axis_mode: str,
    axis_override: str | None,
    vertical_axis: str,
    min_stride_s: float,
    max_stride_s: float,
) -> tuple[dict[str, list[float]], str]:
    events, axis = detect_gait_events_markers(
        csv_path=str(trajectories_csv),
        side=side,
        heel_marker=heel_marker,
        toe_marker=toe_marker,
        axis_mode=axis_mode,
        axis_override=axis_override,
        vertical_axis=vertical_axis,
        out_yaml=None,
    )

    side_prefix = side.lower()
    hs_key = f"{side_prefix}_heel_strike"
    to_key = f"{side_prefix}_toe_off"
    hs = _as_sorted_floats(events.get(hs_key))
    to = _as_sorted_floats(events.get(to_key))

    if not hs:
        return {hs_key: hs, to_key: to}, axis

    t_min = hs[0]
    t_max = hs[-1]
    best_pair = _pick_best_hs_pair(
        hs=hs,
        to=to,
        t_min=t_min,
        t_max=t_max,
        min_stride_s=min_stride_s,
        max_stride_s=max_stride_s,
    )

    out = {hs_key: hs, to_key: to}
    if best_pair is not None:
        out[f"{side_prefix}_heel_strike1"] = [float(best_pair[0])]
        out[f"{side_prefix}_heel_strike2"] = [float(best_pair[1])]
    elif len(hs) >= 2:
        out[f"{side_prefix}_heel_strike1"] = [float(hs[0])]
        out[f"{side_prefix}_heel_strike2"] = [float(hs[1])]
    return out, axis


def _save_yaml(path: Path, payload: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate gait events from marker trajectories in batch.")
    parser.add_argument(
        "--eurobench-root",
        default="data/biomechanics_human_walking/eurobench",
        help="Eurobench root containing *_Trajectories.csv files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *_point_gaitEvents.yaml files.",
    )
    parser.add_argument(
        "--axis-mode",
        choices=["auto", "vertical"],
        default="auto",
        help="Axis selection strategy for extrema detection.",
    )
    parser.add_argument(
        "--axis-override",
        choices=["x", "y", "z"],
        default=None,
        help="Use fixed axis for both sides (overrides axis-mode auto).",
    )
    parser.add_argument(
        "--vertical-axis",
        choices=["x", "y", "z"],
        default="z",
        help="Axis used when axis-mode=vertical.",
    )
    parser.add_argument(
        "--heel-marker",
        default="CAL",
        help="Heel marker suffix after side prefix (e.g., CAL -> RCAL/LCAL).",
    )
    parser.add_argument(
        "--toe-marker",
        default="5TH",
        help="Toe marker suffix after side prefix (e.g., 5TH -> R5TH/L5TH).",
    )
    parser.add_argument(
        "--min-stride-s",
        type=float,
        default=0.5,
        help="Minimum HS-HS duration for selecting heel_strike1/2.",
    )
    parser.add_argument(
        "--max-stride-s",
        type=float,
        default=2.0,
        help="Maximum HS-HS duration for selecting heel_strike1/2.",
    )
    parser.add_argument(
        "--summary-csv",
        default="data/biomechanics_human_walking/eurobench/event_estimation_summary.csv",
        help="Where to save summary table.",
    )
    args = parser.parse_args()

    eurobench_root = Path(args.eurobench_root)
    trajectories = sorted(eurobench_root.rglob("*_Trajectories.csv"))

    rows: list[dict[str, Any]] = []
    created = 0
    skipped = 0
    failed = 0

    for traj in trajectories:
        out_yaml = traj.with_name(traj.name.replace("_Trajectories.csv", "_point_gaitEvents.yaml"))
        if out_yaml.exists() and not args.overwrite:
            skipped += 1
            rows.append(
                {
                    "file": str(traj),
                    "status": "skipped_exists",
                    "axis_right": "",
                    "axis_left": "",
                    "n_r_hs": "",
                    "n_r_to": "",
                    "n_l_hs": "",
                    "n_l_to": "",
                    "has_r_pair": "",
                    "has_l_pair": "",
                    "error": "",
                }
            )
            continue

        try:
            right_events, axis_r = _estimate_for_side(
                trajectories_csv=traj,
                side="R",
                heel_marker=args.heel_marker,
                toe_marker=args.toe_marker,
                axis_mode=args.axis_mode,
                axis_override=args.axis_override,
                vertical_axis=args.vertical_axis,
                min_stride_s=args.min_stride_s,
                max_stride_s=args.max_stride_s,
            )
            left_events, axis_l = _estimate_for_side(
                trajectories_csv=traj,
                side="L",
                heel_marker=args.heel_marker,
                toe_marker=args.toe_marker,
                axis_mode=args.axis_mode,
                axis_override=args.axis_override,
                vertical_axis=args.vertical_axis,
                min_stride_s=args.min_stride_s,
                max_stride_s=args.max_stride_s,
            )

            merged = {}
            merged.update(right_events)
            merged.update(left_events)
            _save_yaml(out_yaml, merged)
            created += 1

            rows.append(
                {
                    "file": str(traj),
                    "status": "ok",
                    "axis_right": axis_r,
                    "axis_left": axis_l,
                    "n_r_hs": len(right_events.get("r_heel_strike", [])),
                    "n_r_to": len(right_events.get("r_toe_off", [])),
                    "n_l_hs": len(left_events.get("l_heel_strike", [])),
                    "n_l_to": len(left_events.get("l_toe_off", [])),
                    "has_r_pair": int("r_heel_strike1" in right_events and "r_heel_strike2" in right_events),
                    "has_l_pair": int("l_heel_strike1" in left_events and "l_heel_strike2" in left_events),
                    "error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            rows.append(
                {
                    "file": str(traj),
                    "status": "error",
                    "axis_right": "",
                    "axis_left": "",
                    "n_r_hs": "",
                    "n_r_to": "",
                    "n_l_hs": "",
                    "n_l_to": "",
                    "has_r_pair": "",
                    "has_l_pair": "",
                    "error": str(exc),
                }
            )

    summary_path = Path(args.summary_csv)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "file",
                "status",
                "axis_right",
                "axis_left",
                "n_r_hs",
                "n_r_to",
                "n_l_hs",
                "n_l_to",
                "has_r_pair",
                "has_l_pair",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"total_files={len(trajectories)}")
    print(f"created={created}")
    print(f"skipped={skipped}")
    print(f"failed={failed}")
    print(f"summary_csv={summary_path}")


if __name__ == "__main__":
    main()

