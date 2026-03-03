import pandas as pd
from pathlib import Path
from typing import Dict, Sequence, Any

from utils.helper import ensure_dir, read_json, write_json


def build_test_predictions(
    ids: Sequence[str],
    preds: Sequence[int],
    gold_labels: Sequence[int],
    proba: Sequence[float]
) -> Dict[str, Any]:
    """Create a record per test example with gold label, prediction, probability, and the MWE string"""
    
    rows = [
        {
            "id": str(id),
            "label": int(y),
            "test_pred": int(preds),
            "test_proba_literal": float(proba)
        }
        for id, y, preds, proba in zip(ids, gold_labels, preds, proba)
    ]
    return rows


def flatten_metrics(metrics):
    """Convert metrics into a flat row to log results and combine them across experiments for later analysis."""
    out = {}

    if "overall" in metrics:
        blocks = [("overall_", metrics["overall"])] + [
            (f"{lang}_", metric) for lang, metric in metrics.get("per_language", {}).items()
        ]
    else:
        blocks = [("", metrics)]

    for prefix, metric in blocks:
        cm = metric.get("confusion_matrix_values", {})
        out.update({
            f"{prefix}accuracy": metric.get("accuracy"),
            f"{prefix}macro_precision": metric.get("macro_precision"),
            f"{prefix}macro_recall": metric.get("macro_recall"),
            f"{prefix}macro_f1": metric.get("macro_f1"),
            f"{prefix}tp": cm.get("tp"),
            f"{prefix}tn": cm.get("tn"),
            f"{prefix}fp": cm.get("fp"),
            f"{prefix}fn": cm.get("fn"),
        })

    return out


def save_artifacts(
    run_dir: Path,
    split_stats: Dict[str, Any],
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    test_predictions: Dict[str, Any]
) -> None:
    """Create the experiment folder and save the results such as metrics and per-example test outputs as JSON or CSV files"""
    
    ensure_dir(run_dir)
    write_json(run_dir / "experiment_config.json", config)
    write_json(run_dir / "metrics.json", metrics)
    pd.DataFrame(split_stats).to_csv(run_dir / "split_stats.csv", index=False)
    pd.DataFrame(test_predictions).to_csv(run_dir / "test_predictions.csv", index=False)
    pd.DataFrame([flatten_metrics(metrics)]).to_csv(run_dir / "metrics.csv", index=False)

    print(f"All files successfully written to {run_dir}")


