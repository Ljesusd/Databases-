import argparse
from pathlib import Path

import pandas as pd
import yaml


NO_INJURY_TOKENS = {
    "no injury",
    "noinjury",
    "no injury,no injury",
}
EMPTY_TOKENS = {
    "",
    "nan",
    "na",
    "n/a",
    "null",
    "none",
}


def _norm_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )


def _is_no_injury_or_empty(series: pd.Series) -> pd.Series:
    s = _norm_text(series)
    return s.isin(NO_INJURY_TOKENS) | s.isin(EMPTY_TOKENS)


def _healthy_mask(df: pd.DataFrame) -> pd.Series:
    if "InjDefn" not in df.columns:
        raise ValueError("Metadata file missing required column: InjDefn")

    inj_defn = _norm_text(df["InjDefn"])
    mask = inj_defn.isin({"no injury", "noinjury"})

    for col in ["InjJoint", "SpecInjury", "InjJoint2", "SpecInjury2"]:
        if col in df.columns:
            mask = mask & _is_no_injury_or_empty(df[col])
    return mask


def _load_and_filter(path_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path_csv)
    mask = _healthy_mask(df)
    out = df.loc[mask].copy()
    out["is_healthy"] = True
    return df, out


def _session_union(run_healthy: pd.DataFrame, walk_healthy: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["sub_id", "filename"]
    run = run_healthy[key_cols + ["datestring"]].copy()
    run["has_run_meta"] = True
    run["has_walk_meta"] = False
    run["source_meta"] = "run"

    walk = walk_healthy[key_cols + ["datestring"]].copy()
    walk["has_run_meta"] = False
    walk["has_walk_meta"] = True
    walk["source_meta"] = "walk"

    both = pd.concat([run, walk], axis=0, ignore_index=True)
    union = (
        both.groupby(key_cols, dropna=False)
        .agg(
            datestring_first=("datestring", "min"),
            has_run_meta=("has_run_meta", "max"),
            has_walk_meta=("has_walk_meta", "max"),
            source_meta=("source_meta", lambda x: ",".join(sorted(set(x)))),
        )
        .reset_index()
    )
    union["sub_id"] = union["sub_id"].astype(int)
    union["json_archive_path"] = union.apply(
        lambda r: f"reformat_data/{int(r['sub_id'])}/{r['filename']}",
        axis=1,
    )
    return union.sort_values(["sub_id", "filename"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create healthy-only subset metadata for running_injury_clinic_kinematic."
    )
    parser.add_argument(
        "--data-root",
        default="data/running_injury_clinic_kinematic",
        help="Folder containing run_data_meta.csv and walk_data_meta.csv.",
    )
    parser.add_argument(
        "--out-root",
        default="data/running_injury_clinic_kinematic/healthy",
        help="Output folder for healthy-only metadata and manifests.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    run_path = data_root / "run_data_meta.csv"
    walk_path = data_root / "walk_data_meta.csv"
    if not run_path.exists() or not walk_path.exists():
        raise FileNotFoundError("Missing run_data_meta.csv or walk_data_meta.csv in --data-root")

    run_all, run_healthy = _load_and_filter(run_path)
    walk_all, walk_healthy = _load_and_filter(walk_path)
    union = _session_union(run_healthy, walk_healthy)

    run_out = out_root / "run_data_meta_healthy.csv"
    walk_out = out_root / "walk_data_meta_healthy.csv"
    union_out = out_root / "sessions_healthy_union.csv"
    manifest_out = out_root / "healthy_json_manifest.txt"
    summary_out = out_root / "healthy_subset_summary.yaml"

    run_healthy.to_csv(run_out, index=False)
    walk_healthy.to_csv(walk_out, index=False)
    union.to_csv(union_out, index=False)
    manifest_out.write_text("\n".join(union["json_archive_path"].tolist()) + "\n", encoding="utf-8")

    summary = {
        "run": {
            "rows_all": int(len(run_all)),
            "rows_healthy": int(len(run_healthy)),
            "subjects_all": int(run_all["sub_id"].nunique()),
            "subjects_healthy": int(run_healthy["sub_id"].nunique()),
        },
        "walk": {
            "rows_all": int(len(walk_all)),
            "rows_healthy": int(len(walk_healthy)),
            "subjects_all": int(walk_all["sub_id"].nunique()),
            "subjects_healthy": int(walk_healthy["sub_id"].nunique()),
        },
        "session_union": {
            "rows_unique": int(len(union)),
            "subjects_unique": int(union["sub_id"].nunique()),
            "has_run_and_walk": int(
                (union["has_run_meta"] & union["has_walk_meta"]).sum()
            ),
            "has_run_only": int(
                (union["has_run_meta"] & ~union["has_walk_meta"]).sum()
            ),
            "has_walk_only": int(
                (~union["has_run_meta"] & union["has_walk_meta"]).sum()
            ),
        },
        "criterion": (
            "healthy if InjDefn is 'No injury' and all injury fields "
            "(InjJoint, SpecInjury, InjJoint2, SpecInjury2) "
            "are either empty or no-injury labels."
        ),
    }
    summary_out.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")

    print(f"run_all={len(run_all)} run_healthy={len(run_healthy)}")
    print(f"walk_all={len(walk_all)} walk_healthy={len(walk_healthy)}")
    print(f"sessions_healthy_union={len(union)}")
    print(f"run_out={run_out}")
    print(f"walk_out={walk_out}")
    print(f"union_out={union_out}")
    print(f"manifest_out={manifest_out}")
    print(f"summary_out={summary_out}")


if __name__ == "__main__":
    main()
