from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from analysis_submodule.analysis_Michele.utils.data_views import filter_baseline, get_data_for_setup, normalize_context, normalize_variant
from analysis_submodule.analysis_Michele.utils.helper_analysis import assert_unique, save_plot, CLASS_RECALL_ORDER, PT_TRANSFER_ORDER
from analysis_submodule.analysis_Michele.utils.plots import add_y_margin_for_annotations, annotate_bar_values


# -----------------------------------------------------------------------------
# RQ5: Cross lingual transfer
# -----------------------------------------------------------------------------
def plot_ordered_bars(
    df: pd.DataFrame,
    x: str,
    y: str,
    save_path: Path,
    title: str,
    ylabel: str,
    xlabel: str = "",
    order: list[str] | None = None,
    figsize: tuple[float, float] = (5.0, 3.2),
    x_rotation: int = 0,
    y_lim: tuple[float, float] | None = None,
    decimals: int = 3,
    annotation_pad: float = 0.002,
    fontsize: int = 9,
    show_zero: bool = False,
    hline_zero: bool = False,
) -> None:
    """Simple ordered bar plot with value annotations."""
    plot_df = df.copy()

    if order is not None:
        plot_df[x] = pd.Categorical(plot_df[x], categories=order, ordered=True)
        plot_df = plot_df.sort_values(x)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(plot_df[x].astype(str), plot_df[y].astype(float))

    if hline_zero:
        ax.axhline(0, color="black", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.tick_params(axis="x", rotation=x_rotation, labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)

    if y_lim is not None:
        ax.set_ylim(*y_lim)

    add_y_margin_for_annotations(
        ax,
        top_frac=0.12,
        bottom_frac=0.06 if hline_zero else 0.02,
    )
    annotate_bar_values(
        ax,
        decimals=decimals,
        pad=annotation_pad,
        fontsize=fontsize,
        show_zero=show_zero,
    )

    save_plot(fig, save_path)



def get_crosslingual_baseline_row(
    master_df: pd.DataFrame,
    model_family: str = "mBERT",
    setting: str = "zero_shot",
    include_mwe_segment: bool = True,
    train_language: str = "EN_PT",
    eval_language: str = "PT",
) -> pd.Series:
    """
    Return the single baseline cross-lingual row for EN_PT -> PT.

    Assumes the cross-lingual run is stored in master_df and marked via:
    - language_mode == cross_lingual
    - language == train_language
    - eval_language == eval_language
    """
    df = master_df.copy()
    for c in ["setting", "language_mode", "language", "eval_language", "model_family", "run_dir"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    df = df[
        (df["setting"] == setting) &
        (df["include_mwe_segment"] == include_mwe_segment)
    ].copy()

    if "run_dir" in df.columns:
        df = df[~df["run_dir"].str.contains("probe", case=False, na=False)].copy()

    df = normalize_variant(df)
    df = normalize_context(df)
    df = filter_baseline(df)

    df = df[
        (df["language_mode"].astype(str).str.lower() == "cross_lingual") &
        (df["language"].astype(str) == train_language) &
        (df["eval_language"].astype(str) == eval_language) &
        (df["model_family"].astype(str) == model_family)
    ].copy()

    if df.empty:
        raise ValueError(
            f"No cross-lingual baseline row found for model_family={model_family}, "
            f"train_language={train_language}, eval_language={eval_language}."
        )

    assert_unique(
        df,
        keys=["model_family", "language", "eval_language", "variant", "context_label"],
        what="get_crosslingual_baseline_row",
    )

    return df.iloc[0]


def build_pt_transfer_comparison_df(
    master_df: pd.DataFrame,
    model_family: str = "mBERT",
    metric: str = "macro_f1",
    include_mwe_segment: bool = True,
) -> pd.DataFrame:
    """
    Build the 3-condition PT comparison for the baseline input:
    - PT monolingual
    - EN+PT multilingual
    - EN→PT cross-lingual
    """
    mono_df = filter_baseline(
        get_data_for_setup(
            master_df,
            setup="monolingual",
            setting="zero_shot",
            include_mwe_segment=include_mwe_segment,
        )
    )
    mono_df = mono_df[
        (mono_df["model_family"].astype(str) == model_family) &
        (mono_df["eval_language"].astype(str) == "PT")
    ].copy()

    if mono_df.empty:
        raise ValueError(f"No PT monolingual baseline row found for model_family={model_family}.")

    assert_unique(
        mono_df,
        keys=["model_family", "eval_language", "variant", "context_label"],
        what="build_pt_transfer_comparison_df monolingual PT baseline",
    )
    pt_mono_value = float(mono_df.iloc[0][metric])

    multi_df = filter_baseline(
        get_data_for_setup(
            master_df,
            setup="multilingual",
            setting="zero_shot",
            include_mwe_segment=include_mwe_segment,
        )
    )
    multi_df = multi_df[
        (multi_df["model_family"].astype(str) == model_family) &
        (multi_df["eval_language"].astype(str) == "PT")
    ].copy()

    if multi_df.empty:
        raise ValueError(f"No PT multilingual baseline row found for model_family={model_family}.")

    assert_unique(
        multi_df,
        keys=["model_family", "eval_language", "variant", "context_label"],
        what="build_pt_transfer_comparison_df multilingual PT baseline",
    )
    pt_multi_value = float(multi_df.iloc[0][metric])

    cross_row = get_crosslingual_baseline_row(
        master_df,
        model_family=model_family,
        setting="zero_shot",
        include_mwe_segment=include_mwe_segment,
        train_language="EN_PT",
        eval_language="PT",
    )
    cross_value = float(cross_row[metric])

    out = pd.DataFrame(
        {
            "condition": ["PT mono", "EN+PT multi", "EN→PT"],
            metric: [pt_mono_value, pt_multi_value, cross_value],
        }
    )
    out["condition"] = pd.Categorical(
        out["condition"],
        categories=PT_TRANSFER_ORDER,
        ordered=True,
    )
    return out.sort_values("condition").reset_index(drop=True)


def plot_pt_transfer_comparison(
    master_df: pd.DataFrame,
    save_path: Path,
    *,
    title: str = "Portuguese test performance across training conditions",
    model_family: str = "mBERT",
    metric: str = "macro_f1",
    include_mwe_segment: bool = True,
) -> pd.DataFrame:
    """
    Main cross-lingual plot:
    PT mono vs EN+PT multi vs EN→PT cross-lingual
    """
    plot_df = build_pt_transfer_comparison_df(
        master_df=master_df,
        model_family=model_family,
        metric=metric,
        include_mwe_segment=include_mwe_segment,
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7.0, 3.9))

    x = np.array([0.00, 0.30, 0.50])
    width = 0.1

    bars = ax.bar(
        x,
        plot_df[metric].astype(float).to_numpy(),
        width=width,
    )

    ax.set_title(title, fontsize=11, pad=12)
    ax.set_xlabel("Training condition", fontsize=10, labelpad=10)
    ax.set_ylabel("Macro-F1 on PT test set", fontsize=10, labelpad=12)
    ax.set_ylim(0.0, 1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(PT_TRANSFER_ORDER)

    ax.tick_params(axis="x", labelsize=9, pad=6)
    ax.tick_params(axis="y", labelsize=9, pad=4)

    ax.grid(axis="y", linewidth=0.6, alpha=0.30)
    ax.grid(axis="x", visible=False)

    add_y_margin_for_annotations(ax, top_frac=0.10, bottom_frac=0.02)
    annotate_bar_values(ax, decimals=3, fontsize=8, show_zero=False)

    fig.subplots_adjust(top=0.88)

    save_plot(fig, save_path)
    return plot_df


def _get_pt_baseline_row_for_setup(
    master_df: pd.DataFrame,
    *,
    setup: str,  # "monolingual" or "multilingual"
    model_family: str = "mBERT",
    setting: str = "zero_shot",
    include_mwe_segment: bool = True,
) -> pd.Series:
    """
    Return the single PT baseline row for the requested training setup.
    """
    df = filter_baseline(
        get_data_for_setup(
            master_df,
            setup=setup,
            setting=setting,
            include_mwe_segment=include_mwe_segment,
        )
    )
    df = df[
        (df["model_family"].astype(str) == model_family) &
        (df["eval_language"].astype(str) == "PT")
    ].copy()

    if df.empty:
        raise ValueError(
            f"No PT baseline row found for setup={setup}, model_family={model_family}."
        )

    assert_unique(
        df,
        keys=["model_family", "eval_language", "variant", "context_label"],
        what=f"_get_pt_baseline_row_for_setup ({setup})",
    )
    return df.iloc[0]


def _class_recalls_from_row(row: pd.Series) -> dict[str, float]:
    """
    Compute class-wise recall from one metrics row.

    Assumes:
    - tp / fn refer to the idiomatic class
    - tn / fp refer to the literal class
    """
    required = ["tp", "tn", "fp", "fn"]
    missing = [k for k in required if k not in row.index]
    if missing:
        raise KeyError(f"Metrics row missing keys: {missing}")

    tp = float(row["tp"])
    tn = float(row["tn"])
    fp = float(row["fp"])
    fn = float(row["fn"])

    idiomatic_recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    literal_recall = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return {
        "Idiomatic": idiomatic_recall,
        "Literal": literal_recall,
    }


def build_pt_transfer_class_recall_df(
    master_df: pd.DataFrame,
    model_family: str = "mBERT",
    include_mwe_segment: bool = True,
) -> pd.DataFrame:
    """
    Build grouped class-wise recall data for:
    - PT mono
    - EN+PT multi
    - EN→PT
    """
    mono_row = _get_pt_baseline_row_for_setup(
        master_df,
        setup="monolingual",
        model_family=model_family,
        setting="zero_shot",
        include_mwe_segment=include_mwe_segment,
    )

    multi_row = _get_pt_baseline_row_for_setup(
        master_df,
        setup="multilingual",
        model_family=model_family,
        setting="zero_shot",
        include_mwe_segment=include_mwe_segment,
    )

    cross_row = get_crosslingual_baseline_row(
        master_df,
        model_family=model_family,
        setting="zero_shot",
        include_mwe_segment=include_mwe_segment,
        train_language="EN_PT",
        eval_language="PT",
    )

    rows = []
    for condition, row in [
        ("PT mono", mono_row),
        ("EN+PT multi", multi_row),
        ("EN→PT", cross_row),
    ]:
        recalls = _class_recalls_from_row(row)
        for class_label, recall in recalls.items():
            rows.append(
                {
                    "condition": condition,
                    "class_label": class_label,
                    "recall": recall,
                }
            )

    out = pd.DataFrame(rows)
    out["condition"] = pd.Categorical(
        out["condition"],
        categories=PT_TRANSFER_ORDER,
        ordered=True,
    )
    out["class_label"] = pd.Categorical(
        out["class_label"],
        categories=CLASS_RECALL_ORDER,
        ordered=True,
    )
    return out.sort_values(["class_label", "condition"]).reset_index(drop=True)


def plot_pt_transfer_class_recall(
    master_df: pd.DataFrame,
    save_path: Path,
    title: str = "Portuguese test set: recall by class and training condition",
    model_family: str = "mBERT",
    include_mwe_segment: bool = True,
) -> pd.DataFrame:
    """
    Grouped comparison:
    class-wise recall for PT mono, EN+PT multi, and EN→PT.
    """
    plot_df = build_pt_transfer_class_recall_df(
        master_df=master_df,
        model_family=model_family,
        include_mwe_segment=include_mwe_segment,
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    # fixed order
    class_order = CLASS_RECALL_ORDER
    cond_order = PT_TRANSFER_ORDER

    # pivot into one row per class, one col per condition
    wide = (
        plot_df.pivot(index="class_label", columns="condition", values="recall")
        .reindex(index=class_order, columns=cond_order)
    )

    # closer class groups + slimmer bars
    x = np.array([0.00, 0.58])
    width = 0.14
    offsets = np.array([-0.15, 0.00, 0.15])

    palette = sns.color_palette(n_colors=len(cond_order))

    for i, condition in enumerate(cond_order):
        values = wide[condition].to_numpy(dtype=float)
        ax.bar(
            x + offsets[i],
            values,
            width=width,
            label=condition,
            color=palette[i],
        )

    ax.set_title(title, fontsize=11, pad=12)
    ax.set_xlabel("Class", fontsize=10, labelpad=10)
    ax.set_ylabel("Recall", fontsize=10, labelpad=12)
    ax.set_ylim(0.0, 1.0)

    ax.set_xticks(x)
    ax.set_xticklabels(class_order)

    ax.tick_params(axis="x", labelsize=9, pad=6)
    ax.tick_params(axis="y", labelsize=9, pad=4)

    ax.grid(axis="y", linewidth=0.6, alpha=0.30)
    ax.grid(axis="x", visible=False)

    legend = ax.legend(
        title="Training condition",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=3,
        fontsize=9,
        title_fontsize=9,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.6,
    )

    add_y_margin_for_annotations(ax, top_frac=0.10, bottom_frac=0.02)
    annotate_bar_values(ax, decimals=3, fontsize=8, show_zero=False)

    fig.subplots_adjust(bottom=0.24, top=0.88)

    save_plot(fig, save_path)
    return plot_df