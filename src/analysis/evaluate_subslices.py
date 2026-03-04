import pandas as pd
from pathlib import Path
from typing import Any, Dict

from config import PATHS
from evaluation.reporting import extract_run_base
from analysis.create_subslices import create_subslices
from utils.helper import read_csv_data


def build_instance_table_for_run(pred_csv: Path, analysis_df: pd.DataFrame, base: Dict[str, Any]) -> pd.DataFrame:
    """Merge exisitng tables"""

    preds_df = read_csv_data(pred_csv)
    preds_df["id"] = preds_df["id"].astype(str)

    analysis_df = analysis_df.copy()
    analysis_df["ID"] = analysis_df["ID"].astype(str)
    analysis_df = analysis_df[[
        "ID",
        "MWE",
        "seen_mwe_type",
        "train_mwe_freq_bin",
        "is_ambiguous_mwe",
        "slice_minority_instance",
        "slice_ambiguous",
    ]]

    analysis = (
        preds_df[["id"]]
        .merge(analysis_df, left_on="id", right_on="ID", how="left")
        .drop(columns=["ID", "id"])
    )

    n_rows = len(preds_df)  # one row per test instance in this run
    base_cols_df = pd.DataFrame({key: [value] * n_rows for key, value in base.items()})

    return pd.concat([base_cols_df, preds_df, analysis], axis=1)


def evaluate_all_runs(runs_root: Path, results_root: Path, split_type: str = "test") -> pd.DataFrame:
    """Evaluates all available experiments on the same set of subslices and saves a master table"""

    rows = []

    for experiment_dir in sorted(runs_root.iterdir()):
        if not experiment_dir.is_dir():
            continue

        pred_csv = experiment_dir / "test_predictions.csv"
      
        base = extract_run_base(experiment_dir)
        setting = base.get("setting")

        analysis_path = PATHS.data_analysis / f"{setting}_{split_type}_analysis.csv"
        if not analysis_path.exists():
            raise FileNotFoundError(f"Missing analysis table: {analysis_path}")

        analysis_df = read_csv_data(analysis_path)

        df = build_instance_table_for_run(pred_csv, analysis_df, base)
        df.to_csv(experiment_dir / "instance_overview.csv", index=False)
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    overview_df = pd.concat(rows, ignore_index=True)
    overview_df.to_csv(results_root / "slices_overview.csv", index=False)

    return overview_df


def evaluate_subslices(project_paths = PATHS, split_type: str = "test") -> None:
    """Subslice evaluation: Runs subslice evaluation over all experiments and stores the aggregated results"""

    create_subslices(project_paths)
    evaluate_all_runs(runs_root=PATHS.runs, results_root=PATHS.results, split_type=split_type)