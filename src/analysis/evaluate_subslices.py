import json
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from pathlib import Path

from evaluation.metrics import compute_metrics
from config import PATHS
from evaluation.reporting import extract_run_base
from utils.helper import write_json


def evaluate_slices_for_run(
    pred_csv: Path,
    slice_ids: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Computes per-slice performance for one experiment run"""
    
    pred_df = pd.read_csv(pred_csv)

    pred_df["id"] = pred_df["id"].astype(str)
    pred_df["label"] = pred_df["label"].astype(int)
    pred_df["test_pred"] = pred_df["test_pred"].astype(int)

    pred_by_id = pred_df.set_index("id", drop=False)

    out: Dict[str, Any] = {}
    for slice_name, ids in slice_ids.items():
        ids_set = set(map(str, ids))
        sub = pred_by_id.loc[pred_by_id.index.intersection(ids_set)]

        if len(sub) == 0:
            out[slice_name] = {"n": 0}
            continue

        y = sub["label"].to_numpy()
        preds = sub["test_pred"].to_numpy()

        metrics = compute_metrics(y, preds)

        out[slice_name] = {"n": int(len(sub)), **metrics}

    return out


def flatten_slice_metrics(
    slice_metrics: Dict[str, Any],
    base: Dict[str, Any],       
) -> pd.DataFrame:
    """Converts per-slice metrics into a single long table with run metadata"""
    rows = []
    for slice_name, slice_result in slice_metrics.items():
        cm = slice_result.get("confusion_matrix_values", {}) if isinstance(m, dict) else {}

        rows.append({
            **base,                 
            "slice": slice_name,
            "n_samples": slice_result.get("n_samples", 0),

            "accuracy": slice_result.get("accuracy"),
            "macro_precision": slice_result.get("macro_precision"),
            "macro_recall": slice_result.get("macro_recall"),
            "macro_f1": slice_result.get("macro_f1"),

            "tp": cm.get("tp"),
            "tn": cm.get("tn"),
            "fp": cm.get("fp"),
            "fn": cm.get("fn"),
        })
    return pd.DataFrame(rows)


def evaluate_all_runs(
    runs_root: Path,
    save_dir: Path,
    split_type: str,
    include_all_reference: bool = True,
    all_slice_name: str = "ALL",
) -> pd.DataFrame:
    """Evaluates all available experiments on the same set of subslices and saves a master table"""

    rows_all = []

    for exp_dir in sorted(runs_root.iterdir()):
        if not exp_dir.is_dir():
            continue

        pred_csv = exp_dir / "test_predictions.csv"
        if not pred_csv.exists():
            continue

        pred_df = pd.read_csv(pred_csv)
        pred_df["id"] = pred_df["id"].astype(str)

        # load run config to know setting
        exp_cfg_path = exp_dir / "experiment_config.json"
        if not exp_cfg_path.exists():
            continue

        with open(exp_cfg_path, "r") as f:
            exp_cfg = json.load(f)

        setting = exp_cfg.get("setting")
        if setting not in {"one_shot", "zero_shot"}:
            continue

        slice_ids_path = PATHS.data_analysis / f"{setting}_{split_type}_slice_ids.json"

        if slice_ids_path.exists():
            with open(slice_ids_path, "r") as f:
                run_slice_ids = json.load(f)
        else:
            run_slice_ids = {}

        if include_all_reference and all_slice_name not in run_slice_ids:
            run_slice_ids[all_slice_name] = pred_df["id"].tolist()

        slice_metrics = evaluate_slices_for_run(pred_csv, run_slice_ids)
        write_json(exp_dir / "slice_metrics.json", slice_metrics)

        base = extract_run_base(exp_dir)
        df_long = flatten_slice_metrics(slice_metrics, base=base)
        rows_all.append(df_long)

    if not rows_all:
        return pd.DataFrame()

    df_long_all = pd.concat(rows_all, ignore_index=True)

    # add slice_group
    df_long_all["slice_group"] = np.where(
        df_long_all["slice"].astype(str).str.startswith("freqbin="),
        "freqbin",
        "ambiguous",
    )

    # sort
    df_long_all = df_long_all.sort_values(
        by=["run_dir", "slice_group", "slice"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


    df_long_all.to_csv(save_dir / "slice_metrics_long.csv", index=False)

    return df_long_all


def evaluate_subslices(split_type: str="test"):
    """Subslice evaluation: Runs subslice evaluation over all experiments and stores the aggregated results"""
    runs_root = PATHS.runs
    save_dir = PATHS.results
    evaluate_all_runs(runs_root=runs_root, save_dir=save_dir, split_type=split_type)