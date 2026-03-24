import pandas as pd
import seaborn as sns
from pathlib import Path
import numpy as np

from analysis_submodule.analysis_Michele.utils.helper_analysis import CONTEXT_ORDER, VARIANT_ORDER, TRAINING_SETUP_ORDER, pivot_strict, save_plot
from analysis_submodule.analysis_Michele.utils.data_views import get_data_for_setup
from analysis_submodule.analysis_Michele.utils.plots import pretty_context_label, pretty_eval_language, pretty_metric, pretty_model_family, pretty_setting


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _get_mono_and_multilingual(
    master_df: pd.DataFrame,
    setting: str,
    include_mwe_segment: bool,
    train_lang_joint: str,
    eval_languages: tuple[str, ...],
    model_family: str | None = None,
    baseline_only: bool = False,  # Standard + Full
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (mono_df, joint_df) using get_data_for_setup.
    Optionally filters to model_family and baseline (Standard+Full), and restricts eval_languages.
    """
    mono = get_data_for_setup(
        master_df,
        setup="monolingual",
        setting=setting,
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        drop_probe_runs=True,
    )
    multilingual = get_data_for_setup(
        master_df,
        setup="multilingual",
        setting=setting,
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        drop_probe_runs=True,
    )

    if model_family is not None:
        mono = mono[mono["model_family"] == model_family].copy()
        multilingual = multilingual[multilingual["model_family"] == model_family].copy()

    if baseline_only:
        mono = mono[(mono["variant"] == "Standard") & (mono["context_label"] == "Full")].copy()
        multilingual = multilingual[(multilingual["variant"] == "Standard") & (multilingual["context_label"] == "Full")].copy()

    mono = mono[mono["eval_language"].astype(str).isin(eval_languages)].copy()
    multilingual = multilingual[multilingual["eval_language"].astype(str).isin(eval_languages)].copy()

    return mono, multilingual


def _delta_train_long(
    master_df: pd.DataFrame,
    setting: str,
    include_mwe_segment: bool,
    train_lang_joint: str,
    eval_languages: tuple[str, ...] = ("EN", "PT"),
    metric: str = "macro_f1",
) -> pd.DataFrame:
    """
    Returns long df with:
      model_family, context_label, variant, eval_language, delta_train
    where delta_train = joint - monolingual
    """
    mono, multilingual = _get_mono_and_multilingual(
        master_df,
        setting=setting,
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        eval_languages=eval_languages,
        model_family=None,
        baseline_only=False,
    )

    keys = ["model_family", "context_label", "variant", "eval_language"]
    for c in keys:
        mono[c] = mono[c].astype(str)
        multilingual[c] = multilingual[c].astype(str)

    mono_s = mono.set_index(keys)[metric].sort_index()
    multilingual_s = multilingual.set_index(keys)[metric].reindex(mono_s.index)

    delta = (multilingual_s - mono_s).rename("delta_train").reset_index()
    return delta


def plot_delta_train_over_variants_grid(
    master_df: pd.DataFrame,
    save_path: Path,
    title: str,
    setting: str = "zero_shot",
    include_mwe_segment: bool = True,
    train_lang_joint: str = "EN_PT_GL",
    eval_languages: tuple[str, ...] = ("EN", "PT"),
    metric: str = "macro_f1",
    height: float = 3.2,
    aspect: float = 1.2,
) -> pd.DataFrame:
    """
    Gemeinsame Figur (FacetGrid):
      x = variant
      y = Δtrain (multilingual − monolingual)
      hue = eval_language
      row = context_label
      col = model_family
    """
    df = _delta_train_long(
        master_df,
        setting=setting,
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        eval_languages=eval_languages,
        metric=metric,
    )
    df = df.dropna(subset=["delta_train"]).copy()
    if df.empty:
        return df

    # ordering
    df["variant"] = pd.Categorical(df["variant"], categories=VARIANT_ORDER, ordered=True)
    df["context_label"] = pd.Categorical(df["context_label"], categories=CONTEXT_ORDER, ordered=True)
    df["eval_language"] = pd.Categorical(df["eval_language"], categories=list(eval_languages), ordered=True)

    fams = sorted(df["model_family"].dropna().unique().tolist())
    df["model_family"] = pd.Categorical(df["model_family"], categories=fams, ordered=True)

    sns.set_theme(style="whitegrid")
    g = sns.relplot(
        data=df.sort_values("variant"),
        kind="line",
        x="variant",
        y="delta_train",
        hue="eval_language",
        row="context_label",
        col="model_family",
        height=height,
        aspect=aspect,
        markers=True,
        dashes=True,
        estimator=np.mean,   # no-op if unique
        errorbar=None,
        facet_kws={"margin_titles": True},
    )

    # overall title + spacing similar to your reference style
    g.fig.suptitle(title, fontsize=11, y=1.02)

    for ax in g.axes.flat:
        if ax is None:
            continue

        ax.axhline(0, color="0.55", linewidth=0.8)

        ax.set_xlabel("Input variant", fontsize=10, labelpad=10)
        ax.set_ylabel("Δ macro-F1 (multilingual − monolingual)", fontsize=10, labelpad=12)

        ax.tick_params(axis="x", rotation=25, labelsize=9, pad=6)
        ax.tick_params(axis="y", labelsize=9, pad=4)

        ax.grid(axis="y", linewidth=0.6, alpha=0.30)
        ax.grid(axis="x", visible=False)

    # cleaner facet titles
    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    # legend styling closer to reference
    if g._legend is not None:
        g._legend.set_title("Eval language")
        g._legend.get_title().set_fontsize(9)
        for text in g._legend.texts:
            text.set_fontsize(9)

        g._legend.set_bbox_to_anchor((1.03, 1))
        try:
            g._legend.set_loc("upper left")
        except AttributeError:
            pass

    g.fig.subplots_adjust(top=0.88, right=0.86, wspace=0.18, hspace=0.28)

    save_plot(g.fig, save_path)
    return df


def plot_multi_vs_mono_delta_bars_by_model_family(
    master_df: pd.DataFrame,
    save_dir: Path,
    setting: str = "zero_shot",
    include_mwe_segment: bool = True,
    train_lang_joint: str = "EN_PT_GL",
    eval_languages: tuple[str, ...] = ("EN", "PT"),
    metric: str = "macro_f1",
    height: float = 3.2,
    aspect: float = 1.6,
) -> pd.DataFrame:
    """
    Erstellt pro model_family eine Figure:
      row = context_label (Full/Target)
      x = variant
      hue = eval_language (EN/PT)
      y = Δtrain (multilingual − monolingual)
    """
    df = _delta_train_long(
        master_df,
        setting=setting,
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        eval_languages=eval_languages,
        metric=metric,
    )
    df = df.dropna(subset=["delta_train"]).copy()
    if df.empty:
        return df

    df["variant"] = pd.Categorical(df["variant"], categories=VARIANT_ORDER, ordered=True)
    df["context_label"] = pd.Categorical(df["context_label"], categories=CONTEXT_ORDER, ordered=True)
    df["eval_language"] = pd.Categorical(df["eval_language"], categories=list(eval_languages), ordered=True)

    sns.set_theme(style="whitegrid")

    for mf in sorted(df["model_family"].dropna().unique().tolist()):
        sub = df[df["model_family"] == mf].copy()
        if sub.empty:
            continue

        palette = {"EN": "#4C72B0", "PT": "#DD8452"}

        g = sns.catplot(
            data=sub.copy(),
            kind="bar",
            x="variant",
            y="delta_train",
            hue="eval_language",
            row="context_label",
            order=VARIANT_ORDER,
            row_order=CONTEXT_ORDER,
            hue_order=[l for l in eval_languages if l in ("EN", "PT")],
            palette=palette,
            height=height,
            aspect=aspect,
            errorbar=None,
        )

        for ax in g.axes.flat:
            if ax is None:
                continue

            ax.axhline(0, color="0.55", linewidth=0.8)

            ax.set_xlabel("Input variant", fontsize=10, labelpad=10)
            ax.set_ylabel(f"Δ {pretty_metric(metric)} (multi − mono)", fontsize=10, labelpad=10)

            ax.tick_params(axis="x", rotation=25, labelsize=9, pad=6)
            ax.tick_params(axis="y", labelsize=9, pad=4)

            ax.grid(axis="y", linewidth=0.6, alpha=0.30)
            ax.grid(axis="x", visible=False)

        g.set_titles("{row_name}")
        for ax in g.axes.flat:
            if ax is not None:
                ax.set_title(pretty_context_label(ax.get_title()), fontsize=11, pad=10)

        g.fig.suptitle(
            f"{pretty_model_family(mf)} | Δ {pretty_metric(metric)} (multi − mono) | {pretty_setting(setting)}",
            fontsize=11,
            y=0.975,
        )

        if g._legend is not None:
            g._legend.set_title("Eval language")
            g._legend.get_title().set_fontsize(9)
            for text in g._legend.texts:
                text.set_fontsize(9)
                text.set_text(pretty_eval_language(text.get_text()))

            g._legend.set_bbox_to_anchor((0.96, 0.96))
            try:
                g._legend.set_loc("upper left")
            except AttributeError:
                pass

        g.fig.subplots_adjust(top=0.88, right=0.82, hspace=0.18)

        save_plot(g.fig, save_dir / f"delta_multi_vs_mono_bars__{mf}__{setting}.png")

    return df



#######################################################
### !TODO fliegt eher raus oder appendix
def tables_train_delta_context_variant(
    master_df: pd.DataFrame,
    setting: str = "zero_shot",
    include_mwe_segment: bool = True,
    train_lang_joint: str = "EN_PT_GL",
    metric: str = "macro_f1",
    eval_languages: tuple[str, ...] = ("EN", "PT"),
) -> dict[str, pd.DataFrame]:
    """
    Per model_family:
      Δtrain = multilingual - monolingual
    for each (eval_language in EN/PT, context_label in Full/Target, variant).
    """
    mono, multilingual = _get_mono_and_multilingual(
        master_df,
        setting=setting,
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        eval_languages=eval_languages,
        model_family=None,
        baseline_only=False,
    )

    for c in ["model_family", "eval_language", "variant", "context_label"]:
        mono[c] = mono[c].astype(str)
        multilingual[c] = multilingual[c].astype(str)

    cell_keys = ["model_family", "eval_language", "variant", "context_label"]

    mono_s = mono.set_index(cell_keys)[metric].sort_index()
    multilingual_s = multilingual.set_index(cell_keys)[metric].reindex(mono_s.index)

    delta = (multilingual_s - mono_s).rename("Δ").reset_index()

    out: dict[str, pd.DataFrame] = {}
    for mf in sorted(delta["model_family"].unique()):
        sub = delta[delta["model_family"] == mf].copy()

        tab = pivot_strict(
            sub,
            index=["variant"],
            columns=["eval_language", "context_label"],
            values="Δ",
            what=f"ctx×variant delta pivot mf={mf}",
        )

        ordered_cols = []
        for lang in [l for l in eval_languages if l in tab.columns.levels[0]]:
            for ctx in CONTEXT_ORDER:
                if (lang, ctx) in tab.columns:
                    ordered_cols.append((lang, ctx))

        tab = tab.reindex(columns=pd.MultiIndex.from_tuples(ordered_cols, names=["eval_language", "context_label"]))
        out[mf] = tab

    return out


# -----------------------------------------------------------------------------
# PLOTS
# -----------------------------------------------------------------------------
### !TODO fliegt eher raus oder appendix
def plot_training_setup_connected_big_figure(
    master_df: pd.DataFrame,
    save_path: Path,
    setting: str = "zero_shot",
    model_family: str = "mBERT",
    include_mwe_segment: bool = True,
    train_lang_joint: str = "EN_PT_GL",
    eval_languages: tuple[str, ...] = ("EN", "PT"),
    metric: str = "macro_f1",
) -> pd.DataFrame:
    """
    Overview: Monolingual vs Multilingual training across eval_language × context_label with hue=variant.
    """
    mono, multilingual = _get_mono_and_multilingual(
        master_df,
        setting=setting,
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        eval_languages=eval_languages,
        model_family=model_family,
        baseline_only=False,
    )

    mono["training_setup"] = "monolingual"
    multilingual["training_setup"] = "multilingual"
    df = pd.concat([mono, multilingual], ignore_index=True)

    df = df.dropna(subset=["training_setup", "eval_language", "context_label", "variant", metric]).copy()

    df["training_setup"] = pd.Categorical(df["training_setup"], categories=TRAINING_SETUP_ORDER, ordered=True)
    df["eval_language"] = pd.Categorical(
        df["eval_language"].astype(str),
        categories=[l for l in eval_languages if l in set(df["eval_language"].astype(str))],
        ordered=True,
    )
    df["variant"] = pd.Categorical(df["variant"].astype(str), categories=VARIANT_ORDER, ordered=True)
    df["context_label"] = pd.Categorical(df["context_label"].astype(str), categories=CONTEXT_ORDER, ordered=True)

    df["training_setup"] = df["training_setup"].cat.remove_unused_categories()
    df["eval_language"] = df["eval_language"].cat.remove_unused_categories()
    df["variant"] = df["variant"].cat.remove_unused_categories()
    df["context_label"] = df["context_label"].cat.remove_unused_categories()

    grid = sns.catplot(
        data=df,
        kind="point",
        x="training_setup",
        y=metric,
        hue="variant",
        row="eval_language",
        col="context_label",
        order=TRAINING_SETUP_ORDER,
        hue_order=VARIANT_ORDER,
        dodge=True,
        markers="o",
        linestyles="-",
        height=3.2,
        aspect=1.2,
        estimator=np.mean,
        errorbar=None,
    )

    grid.set_axis_labels("", "Macro-F1")
    grid.set_titles("{row_name} | {col_name}")
    grid.fig.suptitle(f"Monolingual vs Multilingual — {model_family} | {setting}", y=1.02)

    save_plot(grid.fig, save_path)
    return df

