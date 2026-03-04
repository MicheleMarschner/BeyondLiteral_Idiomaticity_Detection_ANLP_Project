import json
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from pathlib import Path

from evaluation.metrics import compute_metrics
from config import PATHS
from evaluation.reporting import extract_run_base
from utils.helper import read_csv_data, write_json


def add_deltas_vs_reference(
    df_long: pd.DataFrame,
    ref_slice: str = "ALL",
    metrics: List[str] = None,
) -> pd.DataFrame:
    '''
    Adds delta columns per run: metric - metric(ref_slice).
    Requires df_long to include the reference slice row per run.
    '''
    if metrics is None:
        metrics = ["macro_f1", "accuracy", "log_loss", "mean_pred_conf"]

    out = df_long.copy()
    ref = out[out["slice"] == ref_slice][["run_dir"] + metrics].copy()
    ref = ref.rename(columns={m: f"{m}__ref" for m in metrics})

    out = out.merge(ref, on="run_dir", how="left")

    for m in metrics:
        out[f"delta_{m}_vs_{ref_slice}"] = out[m] - out[f"{m}__ref"]

    return out


def add_deltas_between_two_slices(
    df_long: pd.DataFrame,
    *,
    slice_a: str,
    slice_b: str,
    metrics: List[str] = None,
) -> pd.DataFrame:
    '''
    Produces one row per run: metric(slice_a) - metric(slice_b).
    Useful for hard vs control, minority vs control, etc.
    '''
    if metrics is None:
        metrics = ["macro_f1", "accuracy", "log_loss", "mean_pred_conf"]

    a = df_long[df_long["slice"] == slice_a][["run_dir"] + metrics].copy()
    b = df_long[df_long["slice"] == slice_b][["run_dir"] + metrics].copy()

    a = a.rename(columns={m: f"{m}__{slice_a}" for m in metrics})
    b = b.rename(columns={m: f"{m}__{slice_b}" for m in metrics})

    merged = a.merge(b, on="run_dir", how="inner")

    for m in metrics:
        merged[f"delta_{m}__{slice_a}_minus_{slice_b}"] = merged[f"{m}__{slice_a}"] - merged[f"{m}__{slice_b}"]

    return merged

def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    '''
    y in {0,1}, p = P(y=1 | x)
    '''
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def mean_pred_confidence(p: np.ndarray) -> float:
    '''
    confidence of predicted class given p(y=1): max(p, 1-p)
    '''
    p = np.asarray(p, dtype=float)
    conf = np.maximum(p, 1.0 - p)
    return float(np.mean(conf))



def evaluate_slices_for_run(
    pred_csv: Path,
    slice_ids: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Computes per-slice performance for one experiment run"""
    
    pred_df = read_csv_data(pred_csv)

    pred_df["id"] = pred_df["id"].astype(str)
    pred_df["label"] = pred_df["label"].astype(int)
    pred_df["test_pred"] = pred_df["test_pred"].astype(int)
    pred_df["test_proba_literal"] = pred_df["test_proba_literal"].astype(float)

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
        proba = sub["test_proba_literal"].to_numpy()

        metrics = compute_metrics(y, preds)

        proba_stats = {
            "log_loss": log_loss(y, p_literal),
            "mean_pred_conf": mean_pred_confidence(p_literal),
            "mean_p_literal": float(np.mean(p_literal)),
            "std_p_literal": float(np.std(proba)),
        }

        out[slice_name] = {
            "n_samples": int(len(sub)),
            **metrics,
            "proba_stats": proba_stats,
        }

    return out