from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import PATHS
from utils.helper import ensure_dir


def _short(s: str) -> str:
    return str(s).replace("previous_target_next", "full").replace("target-only", "target")

def _filter(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    out = df.copy()
    for k, v in kwargs.items():
        if v is None:
            continue
        out = out[out[k] == v]
    return out

def _save(fig, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ----------------------------
# 1) Heatmap: context × signal for one slice
# ----------------------------
def plot_heatmap_context_signal(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    setting: str,
    model_family: str,
    eval_language: str,           # "EN"/"PT"/"GL"/"overall"
    language_mode: str | None = None,
    train_language: str | None = None,  # for per_language runs
    contexts: list[str] | None = None,
    signals: list[str] | None = None,
    metric: str = "macro_f1",
) -> None:
    ensure_dir(out_dir)

    q = df.copy()
    q = q[q["eval_language"] == eval_language]
    q = _filter(q, setting=setting, model_family=model_family)
    if language_mode is not None:
        q = q[q["language_mode"] == language_mode]
    if train_language is not None:
        q = q[q["language"] == train_language]

    if contexts is not None:
        q = q[q["context"].isin(contexts)]
    if signals is not None:
        q = q[q["features"].isin(signals)]

    if q.empty:
        return

    # pivot: rows=context, cols=features
    pivot = q.pivot_table(index="context", columns="features", values=metric, aggfunc="mean")

    # order nicely if provided
    if contexts is not None:
        pivot = pivot.reindex(index=contexts)
    if signals is not None:
        pivot = pivot.reindex(columns=signals)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(f"{metric} | {setting} | {model_family} | eval={eval_language}")
    ax.set_xlabel("signal / features")
    ax.set_ylabel("context")

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=30, ha="right")
    ax.set_yticklabels([_short(r) for r in pivot.index])

    # annotate values
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if pd.isna(v):
                continue
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9)

    fname = f"heatmap__{setting}__{model_family}__eval-{eval_language}"
    if language_mode: fname += f"__mode-{language_mode}"
    if train_language: fname += f"__train-{train_language}"
    _save(fig, out_dir / f"{fname}.png")

# ----------------------------
# 2) Barplot: Δ highlight vs none
# ----------------------------
def plot_delta_bar(
    delta_df: pd.DataFrame,
    out_dir: Path,
    *,
    title: str,
    out_name: str,
    eval_language: str,
    group_by: list[str],   # e.g. ["setting","model_family","context","language_mode","language"]
) -> None:
    ensure_dir(out_dir)
    q = delta_df[delta_df["eval_language"] == eval_language].copy()
    if q.empty:
        return

    # build labels
    q["label"] = q[group_by].astype(str).agg(" | ".join, axis=1)
    q = q.sort_values("delta", ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(q))))
    ax.barh(q["label"], q["delta"])
    ax.axvline(0.0, linewidth=1)
    ax.set_title(title + f" (eval={eval_language})")
    ax.set_xlabel("Δ macro_f1")

    _save(fig, out_dir / f"{out_name}__eval-{eval_language}.png")

# ----------------------------
# 3) One-shot gain plot: Δ(one_shot - zero_shot)
# ----------------------------
def plot_one_shot_gain(
    one_shot_delta: pd.DataFrame,
    out_dir: Path,
    *,
    eval_language: str,
    facet_cols: list[str] = ("model_family",),
) -> None:
    ensure_dir(out_dir)
    q = one_shot_delta[one_shot_delta["eval_language"] == eval_language].copy()
    if q.empty:
        return

    # collapse into a readable label
    keep = ["language_mode","language","model_family","context","features","transform","include_mwe_segment"]
    keep = [c for c in keep if c in q.columns]
    q["label"] = q[keep].astype(str).agg(" | ".join, axis=1)
    q = q.sort_values("delta", ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(q))))
    ax.barh(q["label"], q["delta"])
    ax.axvline(0.0, linewidth=1)
    ax.set_title(f"One-shot gain: one_shot − zero_shot (eval={eval_language})")
    ax.set_xlabel("Δ macro_f1")

    _save(fig, out_dir / f"delta__one_shot_gain__eval-{eval_language}.png")

# ----------------------------
# 4) Scatter: EN vs PT tradeoff for multilingual models
# ----------------------------
def plot_en_pt_scatter(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    setting: str,
    model_family: str,
    language_mode: str = "multilingual",
    metric: str = "macro_f1",
) -> None:
    ensure_dir(out_dir)
    q = df[(df["setting"] == setting) & (df["model_family"] == model_family) & (df["language_mode"] == language_mode)].copy()
    if q.empty:
        return

    en = q[q["eval_language"] == "EN"].copy()
    pt = q[q["eval_language"] == "PT"].copy()

    join_cols = ["run_dir","context","features","transform","include_mwe_segment","language","seed"]
    join_cols = [c for c in join_cols if c in q.columns]

    merged = en.merge(pt, on=join_cols, suffixes=("_EN","_PT"))
    if merged.empty:
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(merged[f"{metric}_EN"], merged[f"{metric}_PT"])

    # annotate a few points (optional)
    for _, r in merged.iterrows():
        ax.text(r[f"{metric}_EN"], r[f"{metric}_PT"], _short(r["context"]), fontsize=7, alpha=0.7)

    ax.set_xlabel(f"EN {metric}")
    ax.set_ylabel(f"PT {metric}")
    ax.set_title(f"EN vs PT tradeoff | {setting} | {model_family} | {language_mode}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    _save(fig, out_dir / f"scatter__EN_vs_PT__{setting}__{model_family}.png")




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