import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from analysis_submodule.utils.helper import CONTEXT_ORDER, VARIANT_ORDER, LANGUAGE_SETUP_ORDER, save_plot
from analysis_submodule.main_analysis import get_data_for_setup, pivot_strict


# -----------------------------------------------------------------------------
# INTERNAL HELPERS (refactor, minimal)
# -----------------------------------------------------------------------------
def _get_iso_and_joint(
    master_df: pd.DataFrame,
    *,
    setting: str,
    include_mwe_segment: bool,
    train_lang_joint: str,
    eval_languages: tuple[str, ...],
    model_family: str | None = None,
    baseline_only: bool = False,  # Standard + Full
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (iso_df, joint_df) using get_data_for_setup.
    Optionally filters to model_family and baseline (Standard+Full), and restricts eval_languages.
    """
    iso = get_data_for_setup(
        master_df,
        setup="isolated",
        setting=setting,
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        drop_probe_runs=True,
    )
    joint = get_data_for_setup(
        master_df,
        setup="multilingual",
        setting=setting,
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        drop_probe_runs=True,
    )

    if model_family is not None:
        iso = iso[iso["model_family"] == model_family].copy()
        joint = joint[joint["model_family"] == model_family].copy()

    if baseline_only:
        iso = iso[(iso["variant"] == "Standard") & (iso["context_label"] == "Full")].copy()
        joint = joint[(joint["variant"] == "Standard") & (joint["context_label"] == "Full")].copy()

    iso = iso[iso["eval_language"].astype(str).isin(eval_languages)].copy()
    joint = joint[joint["eval_language"].astype(str).isin(eval_languages)].copy()

    return iso, joint


# -----------------------------------------------------------------------------
# TABLES
# -----------------------------------------------------------------------------
def table_train_delta_baseline(
    master_df: pd.DataFrame,
    setting: str = "zero_shot",
    include_mwe_segment: bool = True,
    train_lang_joint: str = "EN_PT_GL",
    metric: str = "macro_f1",
    eval_languages: tuple[str, ...] = ("EN", "PT"),
) -> pd.DataFrame:
    """
    Baseline = Standard + Full.
    Rows: model_family (all)
    Columns: (eval_language, iso/multi/Δ)
    """
    iso, joint = _get_iso_and_joint(
        master_df,
        setting=setting,
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        eval_languages=eval_languages,
        model_family=None,
        baseline_only=True,
    )

    iso_w = pivot_strict(
        iso,
        index=["model_family"],
        columns=["eval_language"],
        values=metric,
        what="baseline iso pivot",
    )
    joint_w = pivot_strict(
        joint,
        index=["model_family"],
        columns=["eval_language"],
        values=metric,
        what="baseline joint pivot",
    )

    iso_w = iso_w.reindex(index=sorted(iso_w.index))
    joint_w = joint_w.reindex(index=sorted(joint_w.index))

    cols = [l for l in eval_languages if l in iso_w.columns and l in joint_w.columns]
    iso_w = iso_w[cols]
    joint_w = joint_w[cols]

    delta_w = (joint_w - iso_w)

    pieces = []
    for lang in cols:
        pieces.append(
            pd.DataFrame(
                {
                    (lang, "iso"): iso_w[lang],
                    (lang, "multi"): joint_w[lang],
                    (lang, "Δ"): delta_w[lang],
                }
            )
        )
    out = pd.concat(pieces, axis=1)
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["eval_language", "stat"])
    return out


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
      Δtrain = multilingual - isolated
    for each (eval_language in EN/PT, context_label in Full/Target, variant).
    """
    iso, joint = _get_iso_and_joint(
        master_df,
        setting=setting,
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        eval_languages=eval_languages,
        model_family=None,
        baseline_only=False,
    )

    for c in ["model_family", "eval_language", "variant", "context_label"]:
        iso[c] = iso[c].astype(str)
        joint[c] = joint[c].astype(str)

    cell_keys = ["model_family", "eval_language", "variant", "context_label"]

    iso_s = iso.set_index(cell_keys)[metric].sort_index()
    joint_s = joint.set_index(cell_keys)[metric].reindex(iso_s.index)

    delta = (joint_s - iso_s).rename("Δ").reset_index()

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

def plot_language_setup_connected_big_figure(
    master_df: pd.DataFrame,
    save_path: Path,
    *,
    setting: str = "zero_shot",
    model_family: str = "mBERT",
    include_mwe_segment: bool = True,
    train_lang_joint: str = "EN_PT_GL",
    eval_languages: tuple[str, ...] = ("EN", "PT"),
    metric: str = "macro_f1",
) -> pd.DataFrame:
    """
    Overview: isolated vs joint training across eval_language × context_label with hue=variant.
    """
    iso, joint = _get_iso_and_joint(
        master_df,
        setting=setting,
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        eval_languages=eval_languages,
        model_family=model_family,
        baseline_only=False,
    )

    iso["language_setup"] = "isolated"
    joint["language_setup"] = "joint"
    df = pd.concat([iso, joint], ignore_index=True)

    df = df.dropna(subset=["language_setup", "eval_language", "context_label", "variant", metric]).copy()

    df["language_setup"] = pd.Categorical(df["language_setup"], categories=LANGUAGE_SETUP_ORDER, ordered=True)
    df["eval_language"] = pd.Categorical(
        df["eval_language"].astype(str),
        categories=[l for l in eval_languages if l in set(df["eval_language"].astype(str))],
        ordered=True,
    )
    df["variant"] = pd.Categorical(df["variant"].astype(str), categories=VARIANT_ORDER, ordered=True)
    df["context_label"] = pd.Categorical(df["context_label"].astype(str), categories=CONTEXT_ORDER, ordered=True)

    df["language_setup"] = df["language_setup"].cat.remove_unused_categories()
    df["eval_language"] = df["eval_language"].cat.remove_unused_categories()
    df["variant"] = df["variant"].cat.remove_unused_categories()
    df["context_label"] = df["context_label"].cat.remove_unused_categories()

    grid = sns.catplot(
        data=df,
        kind="point",
        x="language_setup",
        y=metric,
        hue="variant",
        row="eval_language",
        col="context_label",
        order=LANGUAGE_SETUP_ORDER,
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
    grid.fig.suptitle(f"Isolated vs Joint — {model_family} | {setting}", y=1.02)

    save_plot(grid.fig, save_path)
    return df

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from analysis_submodule.utils.helper import CONTEXT_ORDER, VARIANT_ORDER, LANGUAGE_SETUP_ORDER, save_plot
from analysis_submodule.main_analysis import get_data_for_setup, pivot_strict


# -----------------------------------------------------------------------------
# INTERNAL: compute Δtrain = joint - isolated in long format (reuse _get_iso_and_joint)
# -----------------------------------------------------------------------------
def _delta_train_long(
    master_df: pd.DataFrame,
    *,
    setting: str,
    include_mwe_segment: bool,
    train_lang_joint: str,
    eval_languages: tuple[str, ...] = ("EN", "PT"),
    metric: str = "macro_f1",
) -> pd.DataFrame:
    """
    Returns long df with:
      model_family, context_label, variant, eval_language, delta_train
    where delta_train = joint - isolated
    """
    iso, joint = _get_iso_and_joint(
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
        iso[c] = iso[c].astype(str)
        joint[c] = joint[c].astype(str)

    iso_s = iso.set_index(keys)[metric].sort_index()
    joint_s = joint.set_index(keys)[metric].reindex(iso_s.index)

    delta = (joint_s - iso_s).rename("delta_train").reset_index()
    return delta


# -----------------------------------------------------------------------------
# Plot B (gemeinsame Figur): Δtrain über Varianten
#   row = context_label (Full/Target)
#   col = model_family
#   hue = eval_language (EN/PT)
# -----------------------------------------------------------------------------
def plot_delta_train_over_variants_grid(
    master_df: pd.DataFrame,
    save_path: Path,
    *,
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
      y = Δtrain (joint − isolated)
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
        estimator=np.mean,  # no-op if unique
        errorbar=None,
    )

    for ax in g.axes.flat:
        ax.axhline(0, color="black", linewidth=1)
        ax.tick_params(axis="x", rotation=25)
        ax.set_xlabel("Input variant")
        ax.set_ylabel("Δtrain (joint − isolated)")

    g.set_titles("{row_name} | {col_name}")
    g.fig.suptitle(title, y=1.02)

    save_plot(g.fig, save_path)
    return df


# -----------------------------------------------------------------------------
# Barplot G2: pro model_family eine Figur, row=context_label
#   x = variant
#   hue = eval_language (EN/PT)
#   y = Δtrain
# -----------------------------------------------------------------------------
def plot_delta_train_bars_g2_per_model_family(
    master_df: pd.DataFrame,
    save_dir: Path,
    *,
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
      y = Δtrain (joint − isolated)
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

    save_dir.mkdir(parents=True, exist_ok=True)

    for mf in sorted(df["model_family"].dropna().unique().tolist()):
        sub = df[df["model_family"] == mf].copy()
        if sub.empty:
            continue

        sns.set_theme(style="whitegrid")
        g = sns.catplot(
            data=sub.sort_values("variant"),
            kind="bar",
            x="variant",
            y="delta_train",
            hue="eval_language",
            row="context_label",
            height=height,
            aspect=aspect,
            errorbar=None,
        )

        for ax in g.axes.flat:
            ax.axhline(0, color="black", linewidth=1)
            ax.tick_params(axis="x", rotation=25)
            ax.set_xlabel("Input variant")
            ax.set_ylabel("Δtrain (joint − isolated)")

        g.set_titles("{row_name}")
        g.fig.suptitle(f"{mf} | Δtrain (joint − isolated) | {setting}", y=1.02)

        save_plot(g.fig, save_dir / f"delta_train_bars__{mf}__{setting}.png")

    return df