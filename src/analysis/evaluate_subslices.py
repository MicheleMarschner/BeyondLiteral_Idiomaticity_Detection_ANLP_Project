from typing import Any, Dict, List

import numpy as np
import pandas as pd
from pathlib import Path

from evaluation.metrics import compute_metrics
from utils.helper import write_json


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    """
    y in {0,1}, p = P(y=1 | x)
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def mean_pred_confidence(p: np.ndarray) -> float:
    """
    confidence of predicted class given p(y=1): max(p, 1-p)
    """
    p = np.asarray(p, dtype=float)
    conf = np.maximum(p, 1.0 - p)
    return float(np.mean(conf))


# ----------------------------
# Core: slice evaluation for one run
# ----------------------------
def evaluate_slices_for_run(
    pred_csv: Path,
    slice_ids: Dict[str, List[str]],
) -> Dict[str, Any]:
    """
    pred_csv must contain: id, label, test_pred, test_proba_literal
    Returns:
      slice_name -> {n, macro_f1, ..., proba_stats{log_loss, mean_pred_conf, ...}}
    """
    pred_df = pd.read_csv(pred_csv)

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
        p = sub["test_proba_literal"].to_numpy()

        metrics = compute_metrics(y, preds)

        proba_stats = {
            "log_loss": log_loss(y, p),
            "mean_pred_conf": mean_pred_confidence(p),
            "mean_p_literal": float(np.mean(p)),
            "std_p_literal": float(np.std(p)),
        }

        out[slice_name] = {"n": int(len(sub)), **metrics, "proba_stats": proba_stats}

    return out


# ----------------------------
# Flatten + deltas
# ----------------------------
def flatten_slice_metrics(
    run_dir: str,
    slice_metrics: Dict[str, Any],
) -> pd.DataFrame:
    """
    slice_metrics: output of evaluate_slices_for_run
    Returns long DF with one row per slice.
    """
    rows = []
    for slice_name, m in slice_metrics.items():
        proba = m.get("proba_stats", {}) if isinstance(m, dict) else {}
        cm = m.get("confusion_matrix_values", {}) if isinstance(m, dict) else {}

        rows.append({
            "run_dir": run_dir,
            "slice": slice_name,
            "n": m.get("n", 0),

            "accuracy": m.get("accuracy"),
            "macro_precision": m.get("macro_precision"),
            "macro_recall": m.get("macro_recall"),
            "macro_f1": m.get("macro_f1"),

            "tp": cm.get("tp"),
            "tn": cm.get("tn"),
            "fp": cm.get("fp"),
            "fn": cm.get("fn"),

            "log_loss": proba.get("log_loss"),
            "mean_pred_conf": proba.get("mean_pred_conf"),
            "mean_p_literal": proba.get("mean_p_literal"),
            "std_p_literal": proba.get("std_p_literal"),
        })

    return pd.DataFrame(rows)


def add_deltas_vs_reference(
    df_long: pd.DataFrame,
    *,
    ref_slice: str = "ALL",
    metrics: List[str] = None,
) -> pd.DataFrame:
    """
    Adds delta columns per run: metric - metric(ref_slice).
    Requires df_long to include the reference slice row per run.
    """
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
    """
    Produces one row per run: metric(slice_a) - metric(slice_b).
    Useful for hard vs control, minority vs control, etc.
    """
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


# ----------------------------
# Batch: all runs
# ----------------------------
def evaluate_all_runs(
    runs_root: Path,
    slice_ids_path: Path,
    out_dir: Path,
    *,
    include_all_reference: bool = True,
    all_slice_name: str = "ALL",
) -> pd.DataFrame:
    """
    For each run folder containing test_predictions.csv:
      - compute slice metrics
      - write run_dir/slice_metrics.json
    Also writes:
      - out_dir/slice_metrics_long.csv
      - out_dir/slice_metrics_with_deltas_vs_ALL.csv
      - (optional) out_dir/deltas__hard_minus_control.csv if those slices exist
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    slice_ids = pd.read_json(slice_ids_path)

    # optionally add ALL slice ids = all ids in a run (computed per run)
    rows_all = []

    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        pred_csv = run_dir / "test_predictions.csv"
        if not pred_csv.exists():
            continue

        # load predictions once
        pred_df = pd.read_csv(pred_csv)
        pred_df["id"] = pred_df["id"].astype(str)

        # build per-run slice ids dict
        run_slice_ids = dict(slice_ids)

        if include_all_reference and all_slice_name not in run_slice_ids:
            run_slice_ids[all_slice_name] = pred_df["id"].tolist()

        # compute
        slice_metrics = evaluate_slices_for_run(pred_csv, run_slice_ids)

        # save per-run json
        write_json(run_dir / "slice_metrics.json", slice_metrics)

        # flatten rows
        df_long = flatten_slice_metrics(run_dir.name, slice_metrics)
        rows_all.append(df_long)

    if not rows_all:
        return pd.DataFrame()

    df_long_all = pd.concat(rows_all, ignore_index=True)
    df_long_all.to_csv(out_dir / "slice_metrics_long.csv", index=False)

    # deltas vs ALL
    if include_all_reference:
        df_with_deltas = add_deltas_vs_reference(df_long_all, ref_slice=all_slice_name)
        df_with_deltas.to_csv(out_dir / f"slice_metrics_with_deltas_vs_{all_slice_name}.csv", index=False)
    else:
        df_with_deltas = df_long_all

    # optional: hard vs control deltas (if you have these slice names)
    # common names from your pipeline:
    #   slice_ambiguous == "hard"/"control" would need to be turned into ID lists if you want them here.
    # If you stored them as ID lists in slice_ids.json, then these will exist.
    if "ambiguous_mwe_ids" in slice_ids and "control_ids" in slice_ids:
        # you can create these keys in your slice-ids creation step if desired
        pass

    return df_with_deltas





def subslice_evaluation():
    pass