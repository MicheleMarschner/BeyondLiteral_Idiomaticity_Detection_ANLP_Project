"""
!TODO 
- manchmal 2 von einer variante -> entscheiden
- one-shot gain überprüfen

"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from analysis_submodule.utils.helper import (
    CONTEXT_ORDER, LANG_ORDER, METRIC_ORDER, VARIANT_ORDER, 
    add_joint_language, normalize_context, normalize_variant, 
    prepare_master_for_settings, prepare_master_with_regime, save_plot
)

def baseline_overview_table(master_df: pd.DataFrame) -> pd.DataFrame:
    df = master_df.copy()

    # filter baseline slice
    df = df[
        (df["setting"] == "zero_shot") &
        (df["include_mwe_segment"] == True) &
        (df["context"].astype(str).str.lower() == "previous_target_next") &
        (df["transform"].fillna("none").astype(str).str.lower() == "none") &
        (df["features"].fillna("").astype(str).str.lower().isin(["", "empty"]))
    ].copy()

    df["eval_language"] = df["eval_language"].astype(str).str.strip().replace({"overall": "Joint"})

    # aggregate
    agg = (
        df.groupby(["model_family", "eval_language"], dropna=False)["macro_f1"]
        .mean()
        .reset_index()
    )

    # add joint if missing (mean EN/PT)
    if "joint" not in set(agg["eval_language"]):
        base = agg[agg["eval_language"].isin(["EN", "PT"])].copy()
        joint = base.groupby("model_family")["macro_f1"].mean().reset_index()
        joint["eval_language"] = "Joint"
        agg = pd.concat([agg, joint], ignore_index=True)

    tab = agg.pivot(index="model_family", columns="eval_language", values="macro_f1")
    tab = tab[[c for c in ["EN", "PT", "Joint"] if c in tab.columns]]

    # MultiIndex columns: macro-F1 over EN/PT/joint
    tab.columns = pd.MultiIndex.from_product([["macro-F1"], tab.columns.tolist()])

    return tab


def context_signal_grouped_language_table(
    master_df: pd.DataFrame,
    setting: str = "zero_shot",
    include_mwe_segment: bool = True,
) -> dict[str, pd.DataFrame]:
    """Returns table per Model Family where cell shows macro F1 per signal and context type grouped by language"""

    df = master_df.copy()
    df = df[(df["setting"] == setting) & (df["include_mwe_segment"] == include_mwe_segment)].copy()

    df = normalize_variant(df)
    df = normalize_context(df)
    df = add_joint_language(df)

    # aggregate over seeds if any
    agg = (
        df.groupby(["model_family", "eval_language", "variant", "context_label"], dropna=False)["macro_f1"]
        .mean()
        .reset_index()
    )

    out: dict[str, pd.DataFrame] = {}

    for mf in sorted(agg["model_family"].dropna().unique()):
        sub = agg[agg["model_family"] == mf].copy()

        # wide: context columns per (language, variant)
        piv = sub.pivot_table(
            index=["variant", "eval_language"],
            columns="context_label",
            values="macro_f1",
            aggfunc="first",
        ).reset_index()

        # compute Δ = Full - Target
        if "Full" not in piv.columns or "Target" not in piv.columns:
            raise ValueError(f"Missing Full/Target for model_family={mf}. Have cols: {piv.columns.tolist()}")
        piv["Δ"] = piv["Full"] - piv["Target"]

        # now pivot to MultiIndex columns: (language, metric)
        wide = piv.pivot_table(
            index="variant",
            columns="eval_language",
            values=["Full", "Target", "Δ"],
            aggfunc="first",
        )

        # reorder to language blocks: EN/PT/joint and metric order Full,Target,Δ inside each block
        new_cols = []
        for lang in LANG_ORDER:
            for metric in METRIC_ORDER:
                if (metric, lang) in wide.columns:
                    new_cols.append((metric, lang))
        wide = wide.reindex(columns=pd.MultiIndex.from_tuples(new_cols, names=wide.columns.names))

        # swap levels to get (language, metric) for nicer LaTeX multicolumn headers
        wide.columns = wide.columns.swaplevel(0, 1)  # (lang, metric)
        wide = wide.sort_index(axis=1, level=[0, 1])

        # enforce per-language metric order Full, Target, Δ
        # (pandas sorting may not keep your metric order reliably)
        ordered_cols = []
        for lang in [l for l in LANG_ORDER if l in wide.columns.levels[0]]:
            for metric in METRIC_ORDER:
                if (lang, metric) in wide.columns:
                    ordered_cols.append((lang, metric))
        wide = wide.reindex(columns=pd.MultiIndex.from_tuples(ordered_cols, names=["language", "metric"]))

        out[mf] = wide

    return out



# -----------------------------------------------------------------------------
# RQ3: Connected points (Target vs Full) — per model_family, per language
# -----------------------------------------------------------------------------

def plot_context_connected_points(
    master_df: pd.DataFrame,
    save_dir: Path,
    model_family: str = "mBERT",
    regime: str = "isolated",
    setting: str = "zero_shot",
    include_mwe_segment: bool = True,
    eval_languages: list[str] = ("EN", "PT", "Joint"),
    train_lang_joint: str = "EN_PT_GL",
    metric: str = "macro_f1",
) -> None:
    '''
    Connected points
    '''
    df = prepare_master_with_regime(
        master_df,
        setting=setting,
        model_family=model_family,
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        eval_languages=tuple(eval_languages),
    )

    q = df[
        (df["regime"] == regime) &
        (df["eval_language"].astype(str).isin(list(eval_languages)))
    ].copy()

    if q.empty:
        print(f"[slope] No data for model_family={model_family}, setting={setting}")
        return

    grid = sns.catplot(
        data=q,
        x="context_label",
        y=metric,
        hue="variant",
        col="language",
        kind="point",
        height=4.2,
        aspect=1.0,
        markers="o",
        order=CONTEXT_ORDER,
    )
    grid.fig.suptitle(f"Does Context Help? (Slope Analysis) — {model_family}", y=1.05)
    grid.set_axis_labels("", "Macro F1")

    file_path = save_dir / f"plot_context_connected__{setting}__{model_family}.png" 
    save_plot(grid.fig, file_path)
   

def plot_context_impact_slope(
    master_df: pd.DataFrame,
    save_dir: Path,
    setting: str = "zero_shot",
    model_family: str = "mBERT",
    regime: str = "isolated",
    eval_languages: list[str] | None = None,
    train_lang_joint: str = "EN_PT_GL",
    include_mwe_segment: bool = True,
    height: float = 3.2,
    aspect: float = 1.3
) -> None:
    '''
    Connected points plot comparing Full vs Target for each variant.
    Facets: rows=language, cols=model_family.
    '''
    df = prepare_master_with_regime(
        master_df,
        setting=setting,
        model_family=model_family,
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        eval_languages=("EN", "PT", "GL", "Joint"),
    )

    plot_df = df[df["regime"] == regime].copy()

    if eval_languages is None:
        eval_languages = [l for l in ["EN", "PT", "GL", "Joint"] if l in set(plot_df["eval_language"].astype(str))]


    plot_df = df[df["eval_language"].astype(str).isin(eval_languages)].copy()
    plot_df["variant"] = pd.Categorical(plot_df["variant"], categories=VARIANT_ORDER, ordered=True)

    # Connected points: seaborn point plot connects by default when x is categorical.
    # We set x=context_label and y=macro_f1 so it draws a line between Target and Full.
    grid = sns.catplot(
        data=plot_df,
        kind="point",
        x="context_label",
        y="macro_f1",
        hue="variant",
        row="eval_language",
        col="model_family",
        dodge=True,
        markers="o",
        linestyles="-",
        height=height,
        aspect=aspect,
    )
    grid.set_axis_labels("", "Macro-F1")
    grid.set_titles("{row_name} | {col_name}")
    
    file_path = save_dir / f"plot_context_impact__{setting}__{model_family}.png" 
    save_plot(grid.fig, file_path)




# -----------------------------------------------------------------------------
# RQ3: Heatmap (context × variant) — per model_family
# -----------------------------------------------------------------------------

def plot_context_variant_heatmaps_per_model(
    master_df: pd.DataFrame,
    save_dir: Path,
    setting: str = "zero_shot",
    model_family: str = "mBERT",
    include_mwe_segment: bool = True,
    regime: str = "isolated",
    eval_languages: list[str] | None = None,
    train_lang_joint: str = "EN_PT_GL",
    figsize: tuple[int, int] = (7, 2),
) -> None:
    '''
    For each model_family: draw one heatmap per eval_language.
    Heatmap axes: rows=context (Full/Target), cols=variant, value=macro-F1.
    '''
    df = prepare_master_with_regime(
        master_df,
        setting=setting,
        model_family=model_family,
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        eval_languages=("EN", "PT", "GL", "Joint"),
    )

    df = df[df["regime"] == regime].copy()
    if df.empty:
        print(f"[heatmap] No data for model_family={model_family}, regime={regime}, setting={setting}")
        return

    if eval_languages is None:
        eval_languages = [l for l in ["EN", "PT", "GL", "Joint"] if l in set(df["eval_language"].astype(str))]

    df["variant"] = pd.Categorical(df["variant"], categories=VARIANT_ORDER, ordered=True)
    df["context_label"] = pd.Categorical(df["context_label"], categories=CONTEXT_ORDER, ordered=True)
    
    for lang in eval_languages:
        sub = df[df["eval_language"].astype(str) == lang].copy()
        if sub.empty:
            continue

        heat = sub.pivot_table(
            index="context_label",
            columns="variant",
            values="macro_f1",
            aggfunc="first",
        ).reindex(index=CONTEXT_ORDER)

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            heat,
            annot=True,
            fmt=".3f",
            cbar=True,
            linewidths=0.5,
        )
        ax.set_title(f"{model_family} | {lang} | macro-F1 (zero-shot)")
        ax.set_xlabel("Variant")
        ax.set_ylabel("Context")

        file_path = save_dir / f"heatmap__{model_family}__{lang}__zero_shot.png"
        save_plot(fig, file_path)


def plot_performance_heatmap(
    master_df: pd.DataFrame,
    save_dir: Path,
    setting: str = "zero_shot",
    model_family: str = "mBERT",
    include_mwe_segment: bool = True,
    regime: str = "isolated",
    train_lang_joint: str = "EN_PT_GL",
    eval_languages: list[str] = ("EN", "PT", "Joint"),
    metric: str = "macro_f1",
) -> None:
    '''
    Heatmap: rows=variant, cols=(language × context_label), values=macro_f1.
    '''
    df = prepare_master_with_regime(
        master_df,
        setting=setting,
        model_family=model_family,
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        eval_languages=("EN", "PT", "GL", "Joint"),
    )
    df = df[df["regime"] == regime].copy()
    df = df[df["eval_language"].astype(str).isin(list(eval_languages))].copy()

    if df.empty:
        print(f"[heatmap] No data for model_family={model_family}, regime={regime}, setting={setting}")
        return

    pivot = df.pivot_table(
        index="variant",
        columns=["eval_language", "context_label"],
        values=metric,
        aggfunc="mean",
    )

    # enforce column order
    new_cols = []
    for lang in eval_languages:
        for ctx in CONTEXT_ORDER:
            if (lang, ctx) in pivot.columns:
                new_cols.append((lang, ctx))
    pivot = pivot.reindex(columns=pd.MultiIndex.from_tuples(new_cols, names=pivot.columns.names))

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", linewidths=0.5, cbar_kws={"label": "Macro F1"}, ax=ax)
    ax.set_title(f"{model_family} | {regime} | {setting}: Variant × Context", fontsize=13)
    ax.set_ylabel("Input Variant")
    ax.set_xlabel("Eval language / Context")

    file_path = save_dir / f"heatmap__{model_family}__{setting}__{regime}.png"
    save_plot(fig, file_path)



# -------------------------------------------------------------------
# RQ4: One-shot gains
# -------------------------------------------------------------------
def plot_one_shot_gains_baseline(
    master_df: pd.DataFrame,
    save_dir: Path,
    setting: str = "zero_shot",
    model_family: str = "mBERT",
    include_mwe_segment: bool = True,
    regime: str = "isolated",
    train_lang_joint: str = "EN_PT_GL",
    eval_languages: list[str] = ("EN", "PT", "Joint"),
    metric: str = "macro_f1",
) -> None:
    """
    Compare zero_shot vs one_shot on the fairest baseline slice:
    - variant=Standard
    - context=Full Context
    """
    df = prepare_master_for_settings(master_df, settings=["zero_shot", "one_shot"])
    df["language"] = df["language"].astype(str).str.strip().str.replace(" ", "", regex=False)

    # apply same regime logic
    is_iso = (df["language_mode"] == "per_language") & (df["language"] == df["eval_language"])
    is_joint = (df["language_mode"] == "multilingual") & (df["language"] == train_lang_joint)
    if regime == "isolated":
        df = df[is_iso].copy()
    else:
        df = df[is_joint].copy()

    q = df[
        (df["include_mwe_segment"] == include_mwe_segment)
        & (df["model_family"] == model_family)
        & (df["eval_language"].astype(str).isin(list(eval_languages)))
        & (df["variant"] == "Standard")
        & (df["context_label"] == "Full")
    ].copy()

    if q.empty:
        print(f"[one-shot] No baseline rows for model_family={model_family}, regime={regime}")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(
        data=q,
        x="eval_language",
        y=metric,
        hue="setting",
        edgecolor="black",
        order=[l for l in eval_languages if l in set(q["eval_language"].astype(str))],
        ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3)

    ax.set_title(f"One-Shot Gain (Baseline) — {model_family} | {regime}", fontsize=12)
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Setting", loc="upper left")

    file_path = save_dir / f"zero_vs_one_baseline__{model_family}__{regime}.png"
    save_plot(fig, file_path)



# -----------------------------------------------------------------------------
# RQ5: Cross lingual transfer
# -----------------------------------------------------------------------------

## EN–PT language gap (EN - PT) barplot — per model_family, per context
def plot_en_pt_gap_barplot(
    master_df: pd.DataFrame,
    save_dir: Path,
    setting: str = "zero_shot",
    model_family: str = "mBERT",
    include_mwe_segment: bool = True,
    regime: str = "isolated",
    train_lang_joint: str = "EN_PT_GL",
    height: float = 3.2,
    aspect: float = 1.5
) -> None:
    '''
    Barplot of EN–PT gap per variant and context: gap = F1_EN - F1_PT.
    Facet by model_family. Hue = context (Full/Target).
    '''
    df = prepare_master_with_regime(
        master_df,
        setting=setting,
        model_family=model_family,
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        eval_languages=("EN", "PT"),
    )
    df = df[df["regime"] == regime].copy()

    if df.empty:
        print(f"[gap] No data for model_family={model_family}, regime={regime}, setting={setting}")
        return

    wide = df.pivot_table(
        index=["variant", "context_label"],
        columns="eval_language",
        values="macro_f1",
        aggfunc="mean",
    ).reset_index()

    if "EN" not in wide.columns or "PT" not in wide.columns:
        print("[gap] Need both EN and PT to compute gap.")
        return

    wide["gap_en_minus_pt"] = wide["EN"] - wide["PT"]
    wide["variant"] = pd.Categorical(wide["variant"], categories=VARIANT_ORDER, ordered=True)
    wide["context_label"] = pd.Categorical(wide["context_label"], categories=CONTEXT_ORDER, ordered=True)

    grid = sns.catplot(
        data=wide,
        kind="bar",
        x="variant",
        y="gap_en_minus_pt",
        hue="context_label",
        height=height,
        aspect=aspect,
    )
    for ax in grid.axes.flat:
        ax.axhline(0, color="black", linewidth=1)
        ax.tick_params(axis="x", rotation=30)

    grid.set_axis_labels("Variant", "Gap (EN − PT) macro-F1")
    grid.fig.suptitle(f"EN–PT Gap — {model_family} | {regime} | {setting}", y=1.02)

    file_path = save_dir / f"barplot_EN_PT_gap__{model_family}__{regime}.png"
    save_plot(grid.fig, file_path)




def plot_f1_over_variants_4lines(
    master_df: pd.DataFrame,
    save_path: Path,
    model_family: str,
    regime: str = "isolated",              
    train_lang_joint: str = "EN_PT_GL",    
    setting: str = "zero_shot",
    include_mwe_segment: bool = True,
    eval_languages: tuple[str, ...] = ("EN", "PT"),
    metric: str = "macro_f1",
) -> pd.DataFrame:
    """
    4 lines total:
      color = eval_language (EN/PT)
      style = context_label (Full/Target)
      x = variant
    Filtered to ONE regime to avoid mixing isolated/joint.
    """
    df = prepare_master_with_regime(
        master_df,
        setting=setting,
        model_family=model_family,
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        eval_languages=eval_languages,  # only EN/PT here
    )

    df = df[df["regime"] == regime].copy()
    if df.empty:
        raise ValueError(f"No rows after filtering for regime={regime}. Check train_lang_joint / language_mode tags.")

    # aggregate over seeds (and any duplicates)
    agg = (
        df.groupby(["eval_language", "context_label", "variant"], dropna=False)[metric]
        .mean()
        .reset_index()
    )

    agg["variant"] = pd.Categorical(agg["variant"], categories=VARIANT_ORDER, ordered=True)
    agg["context_label"] = pd.Categorical(agg["context_label"], categories=CONTEXT_ORDER, ordered=True)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8.5, 4.2))

    sns.lineplot(
        data=agg.sort_values("variant"),
        x="variant",
        y=metric,
        hue="eval_language",
        style="context_label",
        markers=True,
        dashes=True,
        linewidth=2,
        ax=ax,
    )

    ax.set_title(f"{model_family} ({regime}) — Macro-F1 over input variants ({setting})")
    ax.set_xlabel("Input variant")
    ax.set_ylabel("Macro-F1")
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylim(0, 1.0)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="")

    save_plot(fig, save_path)
    return agg