import pandas as pd
from pathlib import Path
from typing import Dict, Sequence, Any

from src.utils.helper import ensure_dir, write_json


def build_test_predictions(
    ids: Sequence[str],
    preds: Sequence[int],
    gold_labels: Sequence[int],
    proba: Sequence[float],
    mwe: Sequence[str],
) -> Dict[str, Any]:
    """Create a record per test example with gold label, prediction, probability, and the MWE string"""
    
    rows = [
        {
            "id": str(id),
            "label": int(y),
            "test_pred": int(preds),
            "test_proba": float(proba),
            "mwe": str(mwes),
        }
        for id, y, preds, proba, mwes in zip(ids, gold_labels, preds, proba, mwe)
    ]
    return rows


def flatten_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Convert metrics into a flat row to log results and combine them across experiments for later analysis."""
    cm = metrics.get("confusion_matrix_values", {})
    return {
        "accuracy": metrics.get("accuracy"),
        "macro_precision": metrics.get("macro_precision"),
        "macro_recall": metrics.get("macro_recall"),
        "macro_f1": metrics.get("macro_f1"),
        "tp": cm.get("tp"),
        "tn": cm.get("tn"),
        "fp": cm.get("fp"),
        "fn": cm.get("fn"),
    }


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

