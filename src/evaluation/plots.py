from typing import Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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