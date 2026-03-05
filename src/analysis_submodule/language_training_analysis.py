import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from analysis_submodule.utils.helper import CONTEXT_ORDER, VARIANT_ORDER, REGIME_ORDER, normalize_context, normalize_variant, prepare_master_with_regime, save_plot


def table3_joint_minus_isolated_deltas(df_reg: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Per model_family table: delta = joint - isolated.
    Rows=variant; Cols=(language, metric) where metric in [Full, Target, Δ].
    """
    agg = (
        df_reg.groupby(["model_family", "eval_language", "variant", "context_label", "regime"], dropna=False)["macro_f1"]
        .mean()
        .reset_index()
    )

    # wide regimes
    wide_r = agg.pivot_table(
        index=["model_family", "eval_language", "variant", "context_label"],
        columns="regime",
        values="macro_f1",
        aggfunc="mean",
    ).reset_index()

    if "isolated" not in wide_r.columns:
        wide_r["isolated"] = float("nan")
    if "joint" not in wide_r.columns:
        wide_r["joint"] = float("nan")

    wide_r["delta"] = wide_r["joint"] - wide_r["isolated"]

    out: dict[str, pd.DataFrame] = {}

    for mf in sorted(wide_r["model_family"].dropna().unique()):
        sub = wide_r[wide_r["model_family"] == mf].copy()

        piv = sub.pivot_table(
            index=["variant", "eval_language"],
            columns="context_label",
            values="delta",
            aggfunc="first",
        ).reset_index()

        if "Full" not in piv.columns or "Target" not in piv.columns:
            # if missing, still build what exists (but you'll likely want to fix coverage)
            piv["Full"] = piv.get("Full", float("nan"))
            piv["Target"] = piv.get("Target", float("nan"))

        piv["Δ"] = piv["Full"] - piv["Target"]

        wide = piv.pivot_table(
            index="variant",
            columns="eval_language",
            values=["Full", "Target", "Δ"],
            aggfunc="first",
        )

        wide.columns = wide.columns.swaplevel(0, 1)  # (language, metric)

        ordered_cols = []
        for lang in [l for l in ["EN", "PT", "GL"] if l in wide.columns.levels[0]]:
            for met in ["Full", "Target", "Δ"]:
                if (lang, met) in wide.columns:
                    ordered_cols.append((lang, met))
        wide = wide.reindex(columns=pd.MultiIndex.from_tuples(ordered_cols, names=["language", "metric"]))

        out[mf] = wide

    return out


def build_table_joint_vs_isolated(
    master_df: pd.DataFrame,
    model_family: str = "mBERT",
    setting: str = "zero_shot",
    include_mwe_segment: bool = True,
    context: str = "previous_target_next",   # Full baseline
    variant: str = "Standard",               # output of normalize_variant
    train_lang_joint: str = "EN_PT_GL",      # your joint training tag in `language`
    eval_languages: tuple[str, ...] = ("EN", "PT", "GL"),
) -> pd.DataFrame:
    """
    Build the comparison table for: isolated (per_language) vs joint (multilingual),
    evaluated per language, under a fixed baseline input setup.
    Output columns: language, isolated, joint, delta (joint - isolated).
    """
    df = master_df.copy()

    df = df[
        (df["setting"] == setting) &
        (df["model_family"] == model_family) &
        (df["include_mwe_segment"] == include_mwe_segment) &
        (df["context"] == context)
    ].copy()

    df = normalize_variant(df)  # adds df["variant"]
    df = df[df["variant"].astype(str) == variant].copy()

    df = df[df["eval_language"].isin(eval_languages)].copy()
    if df.empty:
        raise ValueError("No rows after filtering. Check context/variant/setting/model_family.")

    # isolated: train language == eval language (per_language)
    iso = df[
        (df["language_mode"] == "per_language") &
        (df["language"] == df["eval_language"])
    ].copy()
    iso["regime"] = "isolated"

    # joint: multilingual with your joint training tag
    joint = df[
        (df["language_mode"] == "multilingual") &
        (df["language"] == train_lang_joint)
    ].copy()
    joint["regime"] = "joint"

    comp = pd.concat([iso, joint], ignore_index=True)
    if comp.empty:
        raise ValueError(
            "No rows for isolated/joint regimes. "
            "Check language_mode values and train_lang_joint."
        )

    # average over seeds if multiple
    agg = (
        comp.groupby(["eval_language", "regime"], dropna=False)["macro_f1"]
        .mean()
        .reset_index()
        .rename(columns={"eval_language": "language"})
    )

    wide = agg.pivot_table(
        index="language",
        columns="regime",
        values="macro_f1",
        aggfunc="mean",
    ).reset_index()

    # ensure columns exist
    for c in ["isolated", "joint"]:
        if c not in wide.columns:
            wide[c] = float("nan")

    wide["delta"] = wide["joint"] - wide["isolated"]

    # ordering
    wide["language"] = pd.Categorical(wide["language"], categories=list(eval_languages), ordered=True)
    wide = wide.sort_values("language").reset_index(drop=True)

    return wide


def plot_joint_vs_isolated_connected(
    table: pd.DataFrame,
    out_path: Path,
    title: str = "mBERT: isolated vs joint (Macro-F1)",
) -> None:
    """
    Connected dot plot based on the output of build_table_joint_vs_isolated().
    Expects columns: language, isolated, joint, delta.
    """
    required = {"language", "isolated", "joint"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Table missing columns: {sorted(missing)}")

    plot_df = table.melt(
        id_vars=["language"],
        value_vars=["isolated", "joint"],
        var_name="regime",
        value_name="macro_f1",
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    # connect isolated -> joint per language
    for lang in table["language"].astype(str).tolist():
        row = table[table["language"].astype(str) == lang].iloc[0]
        ax.plot(["isolated", "joint"], [row["isolated"], row["joint"]], linewidth=1)

    sns.stripplot(
        data=plot_df,
        x="regime",
        y="macro_f1",
        hue="language",
        dodge=True,
        size=7,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Macro-F1")
    ax.legend(title="Eval language", bbox_to_anchor=(1.02, 1), loc="upper left")

    save_plot(fig, out_path)


def plot_regime_connected_big_figure(
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
    One big figure:
      rows = eval_language
      cols = context_label
      x    = regime (isolated, joint)
      y    = macro_f1
      hue  = variant
    Returns the aggregated table used for plotting.
    """
    df = prepare_master_with_regime(
        master_df,
        setting=setting,
        model_family=model_family,
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        eval_languages=eval_languages,
    )
    df.to_csv(save_path/"master_regime_change.csv")

    # Mean over seeds / duplicates
    agg = (
        df.groupby(["eval_language","context_label","variant","regime"], dropna=False)[metric]
        .mean()
        .reset_index()
    )

    grid = sns.catplot(
        data=agg,
        kind="point",
        x="regime",
        y=metric,
        hue="variant",
        row="eval_language",
        col="context_label",
        order=REGIME_ORDER,
        hue_order=VARIANT_ORDER,
        dodge=True,
        markers="o",
        linestyles="-",
        height=3.2,
        aspect=1.2,
    )
    grid.set_axis_labels("", "Macro-F1")
    grid.set_titles("{row_name} | {col_name}")
    grid.fig.suptitle(f"Isolated vs Joint (connected) — {model_family} | {setting}", y=1.02)

    file_path = "regime_connected__mBERT__zero_shot.png"

    save_plot(grid.fig, save_path/file_path)
    return agg