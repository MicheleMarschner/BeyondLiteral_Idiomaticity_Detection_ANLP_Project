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
    prepare_neutral_master, save_plot
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

    df["eval_language"] = df["eval_language"].astype(str).str.strip().replace({"overall": "joint"})

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
        joint["eval_language"] = "joint"
        agg = pd.concat([agg, joint], ignore_index=True)

    tab = agg.pivot(index="model_family", columns="eval_language", values="macro_f1")
    tab = tab[[c for c in ["EN", "PT", "joint"] if c in tab.columns]]

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
    model_family: str,
    setting: str = "zero_shot",
    include_mwe_segment: bool = True,
    languages: list[str] = ("EN", "PT", "joint"),
    metric: str = "macro_f1",
) -> None:
    '''
    Connected points: x=context_label (Target, Full), y=macro_f1, hue=variant, facet by language.
    '''
    df = prepare_neutral_master(master_df)

    q = df[
        (df["setting"] == setting)
        & (df["include_mwe_segment"] == include_mwe_segment)
        & (df["model_family"] == model_family)
        & (df["language"].astype(str).isin(list(languages)))
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
    languages: list[str] | None = None,
    height: float = 3.2,
    aspect: float = 1.3,
    model_family: str = "mBERT"
) -> None:
    '''
    Connected points plot comparing Full vs Target for each variant.
    Facets: rows=language, cols=model_family.
    '''
    df = prepare_neutral_master(master_df, setting=setting)


    if languages is None:
        languages = [l for l in ["EN", "PT", "joint"] if l in set(df["eval_language"].astype(str))]

    plot_df = df[df["eval_language"].astype(str).isin(languages)].copy()
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
    languages: list[str] | None = None,
    figsize: tuple[int, int] = (7, 2),
    model_family: str = "mBERT"
) -> None:
    '''
    For each model_family: draw one heatmap per eval_language.
    Heatmap axes: rows=context (Full/Target), cols=variant, value=macro-F1.
    '''
    df = prepare_neutral_master(master_df, setting=setting)

    if languages is None:
        languages = [l for l in ["EN", "PT", "joint"] if l in set(df["eval_language"].astype(str))]

    df["variant"] = pd.Categorical(df["variant"], categories=VARIANT_ORDER, ordered=True)
    df["context_label"] = pd.Categorical(df["context_label"], categories=CONTEXT_ORDER, ordered=True)

    for mf in sorted(df["model_family"].dropna().unique()):
        sub_m = df[df["model_family"] == mf].copy()

        for lang in languages:
            sub = sub_m[sub_m["eval_language"].astype(str) == lang].copy()
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
    model_family: str,
    setting: str = "zero_shot",
    include_mwe_segment: bool = True,
    languages: list[str] = ("EN", "PT", "Joint"),
    metric: str = "macro_f1",
) -> None:
    '''
    Heatmap: rows=variant, cols=(language × context_label), values=macro_f1.
    '''
    df = prepare_neutral_master(master_df)

    q = df[
        (df["setting"] == setting)
        & (df["include_mwe_segment"] == include_mwe_segment)
        & (df["model_family"] == model_family)
        & (df["language"].astype(str).isin(list(languages)))
    ].copy()

    if q.empty:
        print(f"[heatmap] No data for model_family={model_family}, setting={setting}")
        return

    pivot = q.pivot_table(
        index="variant",
        columns=["language", "context_label"],
        values=metric,
        aggfunc="mean",
    )

    # enforce column order
    new_cols = []
    for lang in languages:
        for ctx in CONTEXT_ORDER:
            if (lang, ctx) in pivot.columns:
                new_cols.append((lang, ctx))
    pivot = pivot.reindex(columns=pd.MultiIndex.from_tuples(new_cols, names=pivot.columns.names))

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        linewidths=0.5,
        cbar_kws={"label": "Macro F1"},
    )
    ax.set_title(f"{model_family} {setting}: Features vs Context", fontsize=13)
    ax.set_ylabel("Input Variant")
    ax.set_xlabel("Language / Context")

    file_path = save_dir / f"heatmap__{model_family}__zero_shot.png"
    save_plot(fig, file_path)



# -------------------------------------------------------------------
# RQ4: One-shot gains
# -------------------------------------------------------------------
def plot_one_shot_gains_baseline(
    master_df: pd.DataFrame,
    save_dir: Path,
    model_family: str,
    include_mwe_segment: bool = True,
    languages: list[str] = ("EN", "PT", "joint"),
    metric: str = "macro_f1",
) -> None:
    """
    Compare zero_shot vs one_shot on the fairest baseline slice:
    - variant=Standard
    - context=Full Context
    """
    df = prepare_neutral_master(master_df)

    q = df[
        (df["setting"].isin(["zero_shot", "one_shot"]))
        & (df["include_mwe_segment"] == include_mwe_segment)
        & (df["model_family"] == model_family)
        & (df["language"].astype(str).isin(list(languages)))
        & (df["variant"] == "Standard")
        & (df["context_label"] == "Full Context")
    ].copy()

    if q.empty:
        print(f"[one-shot] No baseline rows for model_family={model_family}")
        return

    fig, ax = plt.figure(figsize=(7, 5))
    sns.barplot(
        data=q,
        x="language",
        y=metric,
        hue="setting",
        edgecolor="black",
        order=[l for l in LANG_ORDER if l in set(q["language"].astype(str))],
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3)

    ax.set_title(f"Impact of One-Shot Training (Standard + Full) — {model_family}", fontsize=12)
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.05)
    ax.set_legend(title="Setting", loc="upper left")

    file_path = save_dir / "zero_vs_one.png"
    save_plot(fig, file_path)


def plot_one_shot_gain(
    one_shot_delta: pd.DataFrame,
    save_dir: Path,
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

    save_plot(fig, save_dir)



# -----------------------------------------------------------------------------
# RQ5: Cross lingual transfer
# -----------------------------------------------------------------------------

## EN–PT language gap (EN - PT) barplot — per model_family, per context
def plot_en_pt_gap_barplot(
    master_df: pd.DataFrame,
    save_dir: Path,
    setting: str = "zero_shot",
    height: float = 3.2,
    aspect: float = 1.5,
    model_family: str = "mBERT"
) -> None:
    '''
    Barplot of EN–PT gap per variant and context: gap = F1_EN - F1_PT.
    Facet by model_family. Hue = context (Full/Target).
    '''
    df = prepare_neutral_master(master_df, setting=setting)

    # We need EN and PT present
    need = {"EN", "PT"}
    have = set(df["eval_language"].astype(str).unique())
    if not need.issubset(have):
        raise ValueError(f"Need both EN and PT in eval_language for gap plot. Have: {sorted(have)}")

    # wide: EN/PT per (model_family, variant, context)
    wide = df.pivot_table(
        index=["model_family", "variant", "context_label"],
        columns="eval_language",
        values="macro_f1",
        aggfunc="first",
    ).reset_index()

    wide["gap_en_minus_pt"] = wide["EN"] - wide["PT"]
    wide["variant"] = pd.Categorical(wide["variant"], categories=VARIANT_ORDER, ordered=True)
    wide["context_label"] = pd.Categorical(wide["context_label"], categories=CONTEXT_ORDER, ordered=True)

    grid = sns.catplot(
        data=wide,
        kind="bar",
        x="variant",
        y="gap_en_minus_pt",
        hue="context_label",
        col="model_family",
        height=height,
        aspect=aspect,
    )
    for ax in grid.axes.flat:
        ax.axhline(0, color="black", linewidth=1)
        ax.tick_params(axis="x", rotation=30)

    grid.set_axis_labels("Variant", "Gap (EN − PT) macro-F1")
    grid.set_titles("{col_name}")
    
    file_path = save_dir / f"barplot_EN_PT_gap_{model_family}.png"
    save_plot(grid.fig, file_path)

