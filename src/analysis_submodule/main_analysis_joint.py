import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

from analysis_submodule.utils.helper import CONTEXT_ORDER, VARIANT_ORDER, REGIME_ORDER, save_plot, prepare_master_with_regime


#### 2 Tables #####
def table1_baseline_scoreboard_by_regime(df_reg: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline slice: Full + Standard.
    Returns MultiIndex columns so save_multicol_latex renders grouped headers.
    """
    df = df_reg.copy()

    # baseline: Full + Standard
    df = df[
        (df["context_label"] == "Full") &
        (df["variant"] == "Standard")
    ].copy()

    agg = (
        df.groupby(["model_family", "regime", "eval_language"], dropna=False)["macro_f1"]
        .mean()
        .reset_index()
    )

    tab = agg.pivot_table(
        index=["model_family", "regime"],
        columns="eval_language",
        values="macro_f1",
        aggfunc="mean",
    )

    # order language cols
    col_order = [c for c in ["EN", "PT", "GL", "Joint"] if c in tab.columns]
    tab = tab[col_order]

    tab.columns = pd.MultiIndex.from_product([["macro-F1"], tab.columns.tolist()])
    return tab


def table2_context_variant_by_language_joint(df_reg: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Per model_family table. Joint regime only.
    MultiIndex columns: (language, metric) where metric in [Full, Target, Δ].
    """
    df = df_reg[df_reg["regime"] == "joint"].copy()
    if df.empty:
        raise ValueError("No joint rows in df_reg.")

    agg = (
        df.groupby(["model_family", "eval_language", "variant", "context_label"], dropna=False)["macro_f1"]
        .mean()
        .reset_index()
    )

    out: dict[str, pd.DataFrame] = {}

    for mf in sorted(agg["model_family"].dropna().unique()):
        sub = agg[agg["model_family"] == mf].copy()

        piv = sub.pivot_table(
            index=["variant", "eval_language"],
            columns="context_label",
            values="macro_f1",
            aggfunc="first",
        ).reset_index()

        piv["Δ"] = piv["Full"] - piv["Target"]

        wide = piv.pivot_table(
            index="variant",
            columns="eval_language",
            values=["Full", "Target", "Δ"],
            aggfunc="first",
        )

        # convert to (language, metric)
        wide.columns = wide.columns.swaplevel(0, 1)
        ordered_cols = []
        for lang in [l for l in ["EN", "PT", "GL", "Joint"] if l in wide.columns.levels[0]]:
            for met in ["Full", "Target", "Δ"]:
                if (lang, met) in wide.columns:
                    ordered_cols.append((lang, met))
        wide = wide.reindex(columns=pd.MultiIndex.from_tuples(ordered_cols, names=["language", "metric"]))

        out[mf] = wide

    return out


def plot_context_effect_by_regime(
    master_df: pd.DataFrame,
    save_path: Path,
    setting: str = "zero_shot",
    model_family: str = "mBERT",
    include_mwe_segment: bool = True,
    train_lang_joint: str = "EN_PT_GL",
    eval_languages: tuple[str, ...] = ("EN", "PT", "GL"),
    metric: str = "macro_f1",
) -> pd.DataFrame:
    """
    Connected plot comparing Full vs Target, separately for isolated vs joint.
    Facets: rows=regime, cols=eval_language. Hue=variant.
    """
    df = prepare_master_with_regime(
        master_df,
        setting=setting,
        model_family=model_family,
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        eval_languages=eval_languages,
    )

    agg = (
        df.groupby(["regime","eval_language","variant","context_label"], dropna=False)[metric]
        .mean()
        .reset_index()
    )

    grid = sns.catplot(
        data=agg,
        kind="point",
        x="context_label",
        y=metric,
        hue="variant",
        row="regime",
        col="eval_language",
        order=CONTEXT_ORDER,
        hue_order=VARIANT_ORDER,
        dodge=True,
        markers="o",
        linestyles="-",
        height=3.0,
        aspect=1.1,
    )
    grid.set_axis_labels("", "Macro-F1")
    grid.set_titles("{row_name} | {col_name}")
    grid.fig.suptitle(f"Context effect by regime — {model_family} | {setting}", y=1.02)

    save_plot(grid.fig, save_path)
    return agg



def plot_heatmaps_context_variant_by_regime(
    master_df: pd.DataFrame,
    save_dir: Path,
    *,
    setting: str = "zero_shot",
    model_family: str = "mBERT",
    include_mwe_segment: bool = True,
    train_lang_joint: str = "EN_PT_GL",
    eval_languages: tuple[str, ...] = ("EN", "PT", "GL"),
    metric: str = "macro_f1",
    figsize: tuple[int,int] = (7, 2),
) -> None:
    """
    Saves one heatmap per (regime, eval_language):
      rows = context_label (Full/Target)
      cols = variant
      values = macro-F1
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_master_with_regime(
        master_df,
        setting=setting,
        model_family=model_family,
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        eval_languages=eval_languages,
    )

    agg = (
        df.groupby(["regime","eval_language","context_label","variant"], dropna=False)[metric]
        .mean()
        .reset_index()
    )

    for reg in REGIME_ORDER:
        for lang in eval_languages:
            sub = agg[(agg["regime"] == reg) & (agg["eval_language"] == lang)].copy()
            if sub.empty:
                continue

            heat = sub.pivot_table(
                index="context_label",
                columns="variant",
                values=metric,
                aggfunc="first",
            ).reindex(index=CONTEXT_ORDER)

            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(heat, annot=True, fmt=".3f", cbar=True, linewidths=0.5, ax=ax)
            ax.set_title(f"{model_family} | {setting} | {reg} | eval={lang}")
            ax.set_xlabel("Variant")
            ax.set_ylabel("Context")

            save_plot(fig, save_dir / f"heatmap__{model_family}__{setting}__{reg}__eval-{lang}.png")



def plot_en_pt_gap_by_regime(
    master_df: pd.DataFrame,
    save_path: Path,
    *,
    setting: str = "zero_shot",
    model_family: str = "mBERT",
    include_mwe_segment: bool = True,
    train_lang_joint: str = "EN_PT_GL",
    metric: str = "macro_f1",
) -> pd.DataFrame:
    """
    Gap plot by regime: (EN macro-F1 - PT macro-F1) for each (variant, context_label).
    Facets: columns=regime. Hue=context_label. X=variant.
    """
    df = prepare_master_with_regime(
        master_df,
        setting=setting,
        model_family=model_family,
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        eval_languages=("EN","PT"),   # only these two for gap
    )

    agg = (
        df.groupby(["regime","eval_language","variant","context_label"], dropna=False)[metric]
        .mean()
        .reset_index()
    )

    wide = agg.pivot_table(
        index=["regime","variant","context_label"],
        columns="eval_language",
        values=metric,
        aggfunc="first",
    ).reset_index()

    if "EN" not in wide.columns or "PT" not in wide.columns:
        raise ValueError("Need both EN and PT available for EN–PT gap.")

    wide["gap_en_minus_pt"] = wide["EN"] - wide["PT"]
    wide["variant"] = pd.Categorical(wide["variant"], categories=VARIANT_ORDER, ordered=True)
    wide["context_label"] = pd.Categorical(wide["context_label"], categories=CONTEXT_ORDER, ordered=True)
    wide["regime"] = pd.Categorical(wide["regime"], categories=REGIME_ORDER, ordered=True)

    grid = sns.catplot(
        data=wide,
        kind="bar",
        x="variant",
        y="gap_en_minus_pt",
        hue="context_label",
        col="regime",
        height=3.2,
        aspect=1.4,
    )
    for ax in grid.axes.flat:
        ax.axhline(0, color="black", linewidth=1)
        ax.tick_params(axis="x", rotation=30)

    grid.set_axis_labels("Variant", "Gap (EN − PT) macro-F1")
    grid.set_titles("{col_name}")
    grid.fig.suptitle(f"EN–PT gap by regime — {model_family} | {setting}", y=1.02)

    save_plot(grid.fig, save_path)
    return wide           