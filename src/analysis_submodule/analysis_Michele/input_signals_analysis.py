from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D

from analysis_submodule.analysis_Michele.utils.helper_analysis import CONTEXT_ORDER, VARIANT_ORDER, assert_unique, pivot_strict, save_plot
from analysis_submodule.analysis_Michele.utils.plots import pretty_context_label, pretty_eval_language, pretty_metric, pretty_title


## EN–PT language gap (EN - PT) barplot — per model_family, per context
def plot_en_pt_gap(
    df_setup: pd.DataFrame,
    save_path: Path,
    *,
    title: str,
    metric: str = "macro_f1",
) -> pd.DataFrame:
    """
    Gap plot: (EN - PT) for each (variant, context_label).
    Strict: must have exactly one EN and one PT per (variant, context_label).
    """
    df = df_setup.copy()
    df = df[df["eval_language"].astype(str).isin(["EN", "PT"])].copy()

    # Ensure uniqueness at the cell level
    assert_unique(df, keys=["eval_language", "variant", "context_label"], what="plot_en_pt_gap input")

    wide = pivot_strict(
        df,
        index=["variant", "context_label"],
        columns=["eval_language"],
        values=metric,
        what="plot_en_pt_gap pivot",
    ).reset_index()

    if "EN" not in wide.columns or "PT" not in wide.columns:
        raise ValueError("Need both EN and PT available for EN–PT gap.")

    wide["gap_en_minus_pt"] = wide["EN"] - wide["PT"]
    wide["variant"] = pd.Categorical(wide["variant"], categories=VARIANT_ORDER, ordered=True)
    wide["context_label"] = pd.Categorical(wide["context_label"], categories=CONTEXT_ORDER, ordered=True)

    # Now barplot is safe because wide has exactly one row per bar.
    g = sns.catplot(
        data=wide,
        kind="bar",
        x="variant",
        y="gap_en_minus_pt",
        hue="context_label",
        height=3.2,
        aspect=1.5,
        errorbar=None,
    )
    for ax in g.axes.flat:
        ax.axhline(0, color="black", linewidth=1)
        ax.tick_params(axis="x", rotation=30)

    g.set_axis_labels("Variant", "Gap (EN − PT) macro-F1")
    g.fig.suptitle(title, y=1.02)

    save_plot(g.fig, save_path)
    return wide



# -----------------------------------------------------------------------------
# RQ3: Trend Line (context × variant) — per model_family
# -----------------------------------------------------------------------------
def plot_input_variants_lines_per_lan_setup(
    df_setup: pd.DataFrame,
    save_path: Path,
    title: str,
    eval_order: list[str],
    metric: str = "macro_f1",
) -> pd.DataFrame:
    """
    Lines over variants:
      hue   = eval_language (EN/PT only)
      style = context_label (Full/Target)
      x     = variant

    Joint excluded (use plot_joint_over_variants separately).
    GL excluded by design (too much).
    """
    df = df_setup.copy()
    df = df.dropna(subset=["eval_language", "context_label", "variant", metric]).copy()
    if df.empty:
        return df

    present = set(df["eval_language"].astype(str))
    candidates = [l for l in eval_order if l in present and l in ("EN", "PT")]
    if not candidates:
        return df

    df = df[df["eval_language"].astype(str).isin(candidates)].copy()

    assert_unique(
        df,
        keys=["eval_language", "context_label", "variant"],
        what="plot_input_variants_lines_per_lan_setup uniqueness",
    )

    df["variant"] = pd.Categorical(df["variant"].astype(str), categories=VARIANT_ORDER, ordered=True)
    df["context_label"] = pd.Categorical(
        df["context_label"].astype(str),
        categories=CONTEXT_ORDER,
        ordered=True,
    )
    df["eval_language"] = pd.Categorical(
        df["eval_language"].astype(str),
        categories=candidates,
        ordered=True,
    )

    df["variant"] = df["variant"].cat.remove_unused_categories()
    df["context_label"] = df["context_label"].cat.remove_unused_categories()
    df["eval_language"] = df["eval_language"].cat.remove_unused_categories()

    out = df[["eval_language", "context_label", "variant", metric]].copy().sort_values(
        ["eval_language", "context_label", "variant"]
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8.8, 4.3))

    sns.lineplot(
        data=out,
        x="variant",
        y=metric,
        hue="eval_language",
        style="context_label",
        markers=True,
        dashes=True,
        linewidth=2,
        estimator=np.mean,
        errorbar=None,
        ax=ax,
    )

    ax.set_title(pretty_title(title), fontsize=11, pad=12)
    ax.set_xlabel("Input variant", fontsize=10, labelpad=10)
    ax.set_ylabel(pretty_metric(metric), fontsize=10, labelpad=12)

    ax.tick_params(axis="x", rotation=25, labelsize=9, pad=6)
    ax.tick_params(axis="y", labelsize=9, pad=4)

    ax.set_ylim(0, 1.0)

    ax.grid(axis="y", linewidth=0.6, alpha=0.30)
    ax.grid(axis="x", visible=False)

    handles, labels = ax.get_legend_handles_labels()

    pretty_labels = []
    for label in labels:
        if label == "eval_language":
            pretty_labels.append("Eval language")
        elif label == "context_label":
            pretty_labels.append("Context label")
        elif label in {"EN", "PT", "GL", "Joint"}:
            pretty_labels.append(pretty_eval_language(label))
        else:
            pretty_labels.append(pretty_context_label(label))

    # insert an empty spacer row before "Context label"
    if "Context label" in pretty_labels:
        idx = pretty_labels.index("Context label")
        handles.insert(idx, Line2D([], [], linestyle="none", linewidth=0))
        pretty_labels.insert(idx, "")

    legend = ax.legend(
        handles,
        pretty_labels,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        title="",
        fontsize=9,
        borderaxespad=0.3,
        labelspacing=0.4,
        handletextpad=0.6,
    )

    fig.subplots_adjust(right=0.84, top=0.88)

    save_plot(fig, save_path)
    return out

################################################################
# -----------------------------------------------------------------------------
# tables 
# -----------------------------------------------------------------------------
#### TODO fliegt eher raus oder appendix
def table2_context_variant(df_setup: pd.DataFrame, eval_order: list[str]) -> dict[str, pd.DataFrame]:
    """
    Per model_family table.
    MultiIndex columns: (language, metric) where metric in [Full, Target, Δ].
    Works for both setups (but the set of eval languages differs).
    """
    df = df_setup.copy()

    # Need Full and Target available per (model_family, eval_language, variant)
    assert_unique(
        df,
        keys=["model_family", "eval_language", "variant", "context_label"],
        what="table2_context_variant input",
    )

    out: dict[str, pd.DataFrame] = {}
    for mf in sorted(df["model_family"].dropna().unique()):
        sub = df[df["model_family"] == mf].copy()

        # wide: rows=(variant, eval_language), cols=context_label
        ct = pivot_strict(
            sub,
            index=["variant", "eval_language"],
            columns=["context_label"],
            values="macro_f1",
            what=f"table2_context_variant context pivot mf={mf}",
        ).reset_index()

        if "Full" not in ct.columns or "Target" not in ct.columns:
            raise ValueError(f"[table2] Missing Full/Target for model_family={mf}. Have: {ct.columns.tolist()}")

        ct["Δ"] = ct["Full"] - ct["Target"]

        # now wide: index=variant, columns=(eval_language, metric)
        wide = pivot_strict(
            ct,
            index=["variant"],
            columns=["eval_language"],
            values="Full",
            what=f"table2_full mf={mf}",
        )
        wide_t = pivot_strict(
            ct,
            index=["variant"],
            columns=["eval_language"],
            values="Target",
            what=f"table2_target mf={mf}",
        )
        wide_d = pivot_strict(
            ct,
            index=["variant"],
            columns=["eval_language"],
            values="Δ",
            what=f"table2_delta mf={mf}",
        )

        # combine into MultiIndex columns (language, metric)
        combined = pd.concat({"Full": wide, "Target": wide_t, "Δ": wide_d}, axis=1)
        combined.columns = combined.columns.swaplevel(0, 1)  # -> (eval_language, metric)

        # enforce ordering (language first, then metric)
        ordered_cols = []
        for lang in [l for l in eval_order if l in combined.columns.levels[0]]:
            for met in ["Full", "Target", "Δ"]:
                if (lang, met) in combined.columns:
                    ordered_cols.append((lang, met))
        combined = combined.reindex(columns=pd.MultiIndex.from_tuples(ordered_cols, names=["language", "metric"]))

        out[mf] = combined

    return out



#### TODO fliegt eher raus oder appendix
def plot_joint_over_variants(
    df_setup: pd.DataFrame,
    save_path: Path,
    title: str,
    metric: str = "macro_f1",
) -> pd.DataFrame:
    """
    Joint-only plot (matches styling of plot_input_variants_lines_per_lan_setup):
      - single color for both lines
      - Full vs Target distinguished by line style (solid vs dashed) + markers
      - x = variant
    """
    df = df_setup.copy()
    df = df[df["eval_language"].astype(str) == "Joint"].copy()
    df = df.dropna(subset=["context_label", "variant", metric]).copy()
    if df.empty:
        return df

    assert_unique(
        df,
        keys=["context_label", "variant"],
        what="plot_joint_over_variants uniqueness",
    )

    df["variant"] = pd.Categorical(df["variant"].astype(str), categories=VARIANT_ORDER, ordered=True)
    df["context_label"] = pd.Categorical(df["context_label"].astype(str), categories=CONTEXT_ORDER, ordered=True)
    df["variant"] = df["variant"].cat.remove_unused_categories()
    df["context_label"] = df["context_label"].cat.remove_unused_categories()

    out = df[["context_label", "variant", metric]].copy().sort_values(["context_label", "variant"])

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8.8, 4.3))  # match other plot

    color = "tab:green"

    # solid line
    full = out[out["context_label"] == "Full"].copy()
    if not full.empty:
        sns.lineplot(
            data=full,
            x="variant",
            y=metric,
            color=color,
            marker="o",
            linestyle="-",
            linewidth=2,
            estimator=np.mean,   
            errorbar=None,
            ax=ax,
            label="Full",
        )

    # dashed line
    tgt = out[out["context_label"] == "Target"].copy()
    if not tgt.empty:
        sns.lineplot(
            data=tgt,
            x="variant",
            y=metric,
            color=color,
            marker="X",
            linestyle="--",
            linewidth=2,
            estimator=np.mean,   
            errorbar=None,
            ax=ax,
            label="Target",
        )

    # Match text/formatting with plot_input_variants_lines_per_lan_setup
    ax.set_title(title)
    ax.set_xlabel("Input variant")
    ax.set_ylabel("Macro-F1")
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylim(0, 1.0)

    # Match legend placement
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="context_label")

    save_plot(fig, save_path)
    return out



