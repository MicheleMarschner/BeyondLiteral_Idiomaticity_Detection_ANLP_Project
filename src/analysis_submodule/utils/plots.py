# utils/plots.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.helper import ensure_dir, read_csv_data


# ----------------------------
# Small helpers
# ----------------------------
def _short(s: str) -> str:
    return (
        str(s)
        .replace("previous_target_next", "full")
        .replace("target-only", "target")
    )


def _save_mpl(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_grid(grid, out_path: Path) -> None:
    """Save seaborn FacetGrid/catplot."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.fig.tight_layout()
    grid.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(grid.fig)


def _normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "features" in out.columns:
        out["features"] = out["features"].fillna("").astype(str).replace({"": "empty"})
    return out


def add_signal_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds `signal` that combines `transform` and `features`.

    Examples:
      - transform=none,     features=empty       -> signal="empty"
      - transform=highlight,features=empty       -> signal="highlight"
      - transform=none,     features=ner         -> signal="ner"
      - transform=highlight,features=ner         -> signal="highlight+ner"
      - transform=none,     features=glosses,ner -> signal="glosses,ner"
      - transform=highlight,features=glosses,ner -> signal="highlight+glosses,ner"
    """
    out = _normalize_features(df)

    feats = out["features"].astype(str)
    is_hl = out["transform"].astype(str).eq("highlight") if "transform" in out.columns else False

    signal = feats.copy()
    if isinstance(is_hl, pd.Series):
        signal = signal.where(~is_hl, "highlight+" + signal)
        signal = signal.replace({"highlight+empty": "highlight"})
    out["signal"] = signal
    return out


def load_base_from_master(master: pd.DataFrame) -> pd.DataFrame:
    """One row per run_dir with run metadata used for joins."""
    base_cols = [
        "run_dir",
        "setting",
        "language_mode",
        "language",
        "model_family",
        "seed",
        "context",
        "features",
        "include_mwe_segment",
        "transform",
    ]
    base = master[base_cols].drop_duplicates(subset=["run_dir"]).copy()
    base = _normalize_features(base)
    return base


# ----------------------------
# 1) Heatmap: context × signal (signal = transform + features)
# ----------------------------
def plot_heatmap_context_signal(
    df_master: pd.DataFrame,
    out_dir: Path,
    *,
    setting: str,
    model_family: str,
    eval_language: str,                # "EN"/"PT"/"GL"/"overall"
    language_mode: Optional[str] = None,
    train_language: Optional[str] = None,
    contexts: Optional[list[str]] = None,
    signals: Optional[list[str]] = None,   # uses df["signal"]
    metric: str = "macro_f1",
) -> None:
    ensure_dir(out_dir)

    q = df_master.copy()
    q = q[q["eval_language"] == eval_language]
    q = q[(q["setting"] == setting) & (q["model_family"] == model_family)]

    if language_mode is not None:
        q = q[q["language_mode"] == language_mode]
    if train_language is not None:
        q = q[q["language"] == train_language]

    if q.empty:
        return

    q = add_signal_col(q)

    if contexts is not None:
        q = q[q["context"].isin(contexts)]
    if signals is not None:
        q = q[q["signal"].isin(signals)]

    if q.empty:
        return

    pivot = q.pivot_table(index="context", columns="signal", values=metric, aggfunc="mean")

    if contexts is not None:
        pivot = pivot.reindex(index=contexts)
    if signals is not None:
        pivot = pivot.reindex(columns=signals)

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(f"{metric} | {setting} | {model_family} | eval={eval_language}")
    ax.set_xlabel("signal")
    ax.set_ylabel("context")

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=30, ha="right")
    ax.set_yticklabels([_short(r) for r in pivot.index])

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if pd.isna(v):
                continue
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9)

    fname = f"heatmap__{setting}__{model_family}__eval-{eval_language}"
    if language_mode:
        fname += f"__mode-{language_mode}"
    if train_language:
        fname += f"__train-{train_language}"
    _save_mpl(fig, out_dir / f"{fname}.png")


# ----------------------------
# 2) Barplot: Δ highlight vs none (if you have delta__highlight_vs_none.csv)
# ----------------------------
def plot_delta_bar(
    delta_df: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    eval_language: str,
    group_by: list[str],
    delta_col: str = "delta",
) -> None:
    q = delta_df[delta_df["eval_language"] == eval_language].copy()
    if q.empty:
        return

    q["label"] = q[group_by].astype(str).agg(" | ".join, axis=1)
    q = q.sort_values(delta_col, ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(q))))
    ax.barh(q["label"], q[delta_col])
    ax.axvline(0.0, linewidth=1)
    ax.set_title(title + f" (eval={eval_language})")
    ax.set_xlabel("Δ macro_f1")

    _save_mpl(fig, out_path)


# ----------------------------
# 3) One-shot gain plot
# ----------------------------
def plot_one_shot_gain(
    one_shot_delta: pd.DataFrame,
    out_path: Path,
    *,
    eval_language: str,
) -> None:
    q = one_shot_delta[one_shot_delta["eval_language"] == eval_language].copy()
    if q.empty:
        return

    keep = ["language_mode", "language", "model_family", "context", "features", "transform", "include_mwe_segment"]
    keep = [c for c in keep if c in q.columns]
    q["label"] = q[keep].astype(str).agg(" | ".join, axis=1)
    q = q.sort_values("delta", ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(q))))
    ax.barh(q["label"], q["delta"])
    ax.axvline(0.0, linewidth=1)
    ax.set_title(f"One-shot gain: one_shot − zero_shot (eval={eval_language})")
    ax.set_xlabel("Δ macro_f1")

    _save_mpl(fig, out_path)


# ----------------------------
# 4) Scatter: EN vs PT tradeoff for multilingual models
# ----------------------------
def plot_en_pt_scatter(
    df_master: pd.DataFrame,
    out_path: Path,
    *,
    setting: str,
    model_family: str,
    language_mode: str = "multilingual",
    metric: str = "macro_f1",
) -> None:
    q = df_master[
        (df_master["setting"] == setting)
        & (df_master["model_family"] == model_family)
        & (df_master["language_mode"] == language_mode)
    ].copy()
    if q.empty:
        return

    en = q[q["eval_language"] == "EN"].copy()
    pt = q[q["eval_language"] == "PT"].copy()

    join_cols = ["run_dir", "context", "features", "transform", "include_mwe_segment", "language", "seed"]
    join_cols = [c for c in join_cols if c in q.columns]

    merged = en.merge(pt, on=join_cols, suffixes=("_EN", "_PT"))
    if merged.empty:
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(merged[f"{metric}_EN"], merged[f"{metric}_PT"])
    ax.set_xlabel(f"EN {metric}")
    ax.set_ylabel(f"PT {metric}")
    ax.set_title(f"EN vs PT | {setting} | {model_family} | {language_mode}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    _save_mpl(fig, out_path)


def rq2_plot_masking_df_scatter(
    masking_df: pd.DataFrame,
    base: pd.DataFrame,
    out_path: Path,
) -> None:
    # join context/features/transform for styling (masking_df already has setting/language/etc.)
    join_base = base[["run_dir", "context", "features", "transform"]].drop_duplicates(subset=["run_dir"]).copy()
    join_base["features"] = join_base["features"].fillna("").replace({"": "empty"})

    df = masking_df.merge(join_base, on="run_dir", how="left")
    df["features"] = df["features"].fillna("").replace({"": "empty"})

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 6))

    sns.scatterplot(
        data=df,
        x="macro_f1_normal",
        y="macro_f1_both",
        hue="setting",
        style="features",
        ax=ax,
        alpha=0.9,
    )

    lo = float(min(df["macro_f1_normal"].min(), df["macro_f1_both"].min()))
    hi = float(max(df["macro_f1_normal"].max(), df["macro_f1_both"].max()))
    ax.plot([lo, hi], [lo, hi], linewidth=1)

    ax.set_xlabel("macro-F1 (normal)")
    ax.set_ylabel("macro-F1 (mask both)")
    ax.set_title("RQ2: normal vs mask_both (stress masking)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

# ----------------------------
# Confusion matrix from counts (optional utility)
# ----------------------------
def plot_confusion_matrix_from_counts(
    cm_vals: Dict[str, int],
    out_path: Path,
    labels: Tuple[str, str] = ("Idiom", "Literal"),
    title: str = "Confusion Matrix",
) -> None:
    cm = np.array([[cm_vals["tn"], cm_vals["fp"]],
                   [cm_vals["fn"], cm_vals["tp"]]], dtype=float)

    row_sums = cm.sum(axis=1, keepdims=True)
    cm_row = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_row, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(title=title + " (row % + counts)", xlabel="Predicted", ylabel="True")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels([labels[0], labels[1]])
    ax.set_yticklabels([labels[0], labels[1]])

    thr = cm_row.max() / 2.0 if cm_row.size else 0
    for i in range(2):
        for j in range(2):
            count = int(cm[i, j])
            perc = cm_row[i, j] * 100
            color = "white" if cm_row[i, j] > thr else "black"
            ax.text(j, i, f"{count}\n{perc:.1f}%", ha="center", va="center", color=color)

    ax.set_ylim(1.5, -0.5)
    _save_mpl(fig, out_path)


# ----------------------------
# RQ1: stress masking deltas barplot (first/last/both)
# ----------------------------
def rq1_plot_surface_reliance(
    df_stress: pd.DataFrame,
    df_base: pd.DataFrame,
    out_path: Path,
) -> None:
    # stress already contains setting/language/model_family/seed; only join missing run fields
    join_base = df_base[["run_dir", "context", "features", "transform", "include_mwe_segment"]].copy()
    join_base = _normalize_features(join_base)

    df = df_stress.merge(join_base, on="run_dir", how="left")
    df = _normalize_features(df)

    long = df.melt(
        id_vars=["run_dir", "setting", "language", "context", "transform", "features"],
        value_vars=["delta_macro_f1_first", "delta_macro_f1_last", "delta_macro_f1_both"],
        var_name="mask_variant",
        value_name="delta_macro_f1",
    )
    long["mask_variant"] = long["mask_variant"].map({
        "delta_macro_f1_first": "mask_first",
        "delta_macro_f1_last": "mask_last",
        "delta_macro_f1_both": "mask_both",
    })

    long["signal"] = add_signal_col(long)["signal"]
    long["x"] = long["context"].astype(str) + " | " + long["signal"].astype(str)

    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=long,
        kind="bar",
        x="x",
        y="delta_macro_f1",
        hue="mask_variant",
        col="setting",
        estimator="mean",
        errorbar="se",
        height=5,
        aspect=1.5,
        sharey=True,
    )
    g.set_axis_labels("", "Δ macro-F1 (masked − normal)")
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.axhline(0.0, linewidth=1)
        ax.tick_params(axis="x", rotation=45)

    _save_grid(g, out_path)


# ----------------------------
# RQ2: stress masking scatter (normal vs both)
# ----------------------------
def rq2_plot_stress_scatter(
    df_stress: pd.DataFrame,
    df_base: pd.DataFrame,
    out_path: Path,
) -> None:
    join_base = df_base[["run_dir", "context", "features", "transform"]].copy()
    join_base = _normalize_features(join_base)

    df = df_stress.merge(join_base, on="run_dir", how="left")
    df = _normalize_features(df)
    df = add_signal_col(df)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 6))

    sns.scatterplot(
        data=df,
        x="macro_f1_normal",
        y="macro_f1_both",
        hue="setting",
        style="signal",
        ax=ax,
        alpha=0.9,
    )

    lo = float(min(df["macro_f1_normal"].min(), df["macro_f1_both"].min()))
    hi = float(max(df["macro_f1_normal"].max(), df["macro_f1_both"].max()))
    ax.plot([lo, hi], [lo, hi], linewidth=1)

    ax.set_xlabel("macro-F1 (normal)")
    ax.set_ylabel("macro-F1 (mask both)")
    ax.set_title("Stress masking: normal vs mask_both")

    _save_mpl(fig, out_path)


# ----------------------------
# RQ2 support: hard vs control delta (from slice_metrics_long.csv)
# ----------------------------
def rq2_plot_hard_control_delta(
    df_slices: pd.DataFrame,
    df_base: pd.DataFrame,
    out_path: Path,
    *,
    hard_label: str = "hard",
    control_label: str = "control",
    aggregate: bool = True,
) -> None:
    need = df_slices[df_slices["slice"].isin([hard_label, control_label])].copy()
    if need.empty:
        raise ValueError(f"No rows for slices '{hard_label}'/'{control_label}' in slice_metrics_long.csv")

    wide = (
        need.pivot_table(index="run_dir", columns="slice", values="macro_f1", aggfunc="first")
        .reset_index()
    )
    if hard_label not in wide.columns or control_label not in wide.columns:
        raise ValueError("After pivot, 'hard'/'control' columns are missing. Check slice names.")

    wide["delta_hard_minus_control"] = wide[hard_label] - wide[control_label]

    join_base = df_base[["run_dir", "setting", "context", "features", "transform"]].copy()
    join_base = _normalize_features(join_base)

    df = wide.merge(join_base, on="run_dir", how="left")
    df = _normalize_features(df)
    df = add_signal_col(df)

    df["x"] = df["context"].astype(str) + " | " + df["signal"].astype(str)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))

    if aggregate:
        plot_df = (
            df.groupby(["setting", "x"], dropna=False)["delta_hard_minus_control"]
            .mean()
            .reset_index()
        )
        sns.barplot(
            data=plot_df,
            x="x",
            y="delta_hard_minus_control",
            hue="setting",
            errorbar="se",
            ax=ax,
        )
        ax.set_title("Ambiguity gap: hard − control (mean over runs)")
        ax.tick_params(axis="x", rotation=45)
    else:
        sns.barplot(
            data=df.sort_values("delta_hard_minus_control"),
            x="run_dir",
            y="delta_hard_minus_control",
            hue="setting",
            errorbar=None,
            ax=ax,
        )
        ax.set_title("Ambiguity gap: hard − control (per run)")
        ax.tick_params(axis="x", rotation=45)

    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("")
    ax.set_ylabel("Δ macro-F1 (hard − control)")

    _save_mpl(fig, out_path)