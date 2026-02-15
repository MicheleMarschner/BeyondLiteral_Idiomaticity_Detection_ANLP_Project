import numpy as np
import pandas as pd
import matplotlib as plt
from pathlib import Path
from typing import Dict, Sequence, Tuple, Any

from utils.helper import ensure_dir, write_json


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
            "test_proba": float(proba)
        }
        for id, y, preds, proba, mwes in zip(ids, gold_labels, preds, proba)
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


def plot_confusion_matrix_from_counts(
    cm_vals: Dict[str, int],
    save_path: str,
    labels: Tuple[str, str] = ("Idiom", "Literal"),
    title: str = "Confusion Matrix",
    ax=None,
):
    """
    Creates a binary confusion matrix plot:
      [[TN, FP],
       [FN, TP]]
    """

    cm = np.array([[cm_vals["tn"], cm_vals["fp"]],
                   [cm_vals["fn"], cm_vals["tp"]]], dtype=float)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_row = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(cm_row, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(title=title + " (row % + counts)", xlabel="Predicted", ylabel="True")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels([labels[0], labels[1]])
    ax.set_yticklabels([labels[0], labels[1]])


    threshold = cm_row.max() / 2.0 if cm_row.size else 0
    for i in range(2):
        for j in range(2):
            count = int(cm[i, j])
            perc = cm_row[i, j] * 100
            color = "white" if cm_row[i, j] > threshold else "black"
            ax.text(j, i, f"{count}\n{perc:.1f}%", ha="center", va="center", color=color)

    ax.set_ylim(1.5, -0.5)
    plt.tight_layout()

    ax.figure.savefig(save_path, dpi=200, bbox_inches="tight")

    return ax