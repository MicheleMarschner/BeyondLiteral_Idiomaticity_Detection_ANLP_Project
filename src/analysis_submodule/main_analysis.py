"""
“The zero-shot and one-shot test sets contain different instances (no ID overlap), so we cannot do paired-by-example deltas.”

“We therefore report a type-controlled comparison, restricting evaluation to the 32 MWE types shared across both test sets.”

“This isolates changes attributable to the one-shot training language_setup from changes in the expression inventory.”

That’s defensible and linguist-aligned (type inventory control).
"""



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from analysis_submodule.utils.helper import save_multicol_latex

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------
VARIANT_ORDER = [
    "Standard",
    "Highlight",
    "Highlight + NER",
    "Highlight + Glosses",
    "NER",
    "Glosses",
]
CONTEXT_ORDER = ["Full", "Target"]

EVAL_ORDER_MULTI = ["EN", "PT", "GL", "Joint"]
EVAL_ORDER_ISO   = ["EN", "PT", "Joint"]

LANGUAGE_SETUP_ORDER = ["isolated", "joint"]
TRAIN_LANG_JOINT = "EN_PT_GL"


# -----------------------------------------------------------------------------
# NORMALIZATION (strict)
# -----------------------------------------------------------------------------
def normalize_variant(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["transform"] = df["transform"].fillna("none").astype(str).str.strip().str.lower()
    df["features"] = df["features"].fillna("").astype(str).str.strip().str.lower()
    df["features_norm"] = df["features"].replace({"": "empty", "none": "empty", "nan": "empty"})

    def _variant(t: str, f: str) -> str:
        if t == "none" and f == "empty": return "Standard"
        if t == "highlight" and f == "empty": return "Highlight"
        if t == "none" and f == "ner": return "NER"
        if t == "none" and f == "glosses": return "Glosses"
        if t == "highlight" and f == "ner": return "Highlight + NER"
        if t == "highlight" and f == "glosses": return "Highlight + Glosses"
        raise ValueError(f"Unexpected transform/features combo: {t}|{f}")

    df["variant"] = [_variant(t, f) for t, f in zip(df["transform"], df["features_norm"])]
    df["variant"] = pd.Categorical(df["variant"], categories=VARIANT_ORDER, ordered=True)
    return df


def normalize_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    raw = df["context"].astype(str).str.strip().str.lower()
    mapping = {"previous_target_next": "Full", "target": "Target"}
    df["context_label"] = raw.map(mapping)

    if df["context_label"].isna().any():
        bad = sorted(set(df.loc[df["context_label"].isna(), "context"].astype(str)))
        raise ValueError(f"Unexpected context values: {bad}")

    df["context_label"] = pd.Categorical(df["context_label"], categories=CONTEXT_ORDER, ordered=True)
    return df


# -----------------------------------------------------------------------------
# STRICT "NO AGGREGATION" HELPERS
# -----------------------------------------------------------------------------
def assert_unique(
    df: pd.DataFrame,
    *,
    keys: list[str],
    what: str,
) -> None:
    g = df.groupby(keys, dropna=False).size().reset_index(name="n")
    bad = g[g["n"] > 1]
    if not bad.empty:
        # show a small diagnostic; include run_dir/seed if present
        show = bad.head(20).to_string(index=False)
        raise ValueError(
            f"[{what}] Duplicate rows for keys={keys} (would require aggregation).\n"
            f"Examples (first 20):\n{show}"
        )


def pivot_strict(
    df: pd.DataFrame,
    *,
    index: list[str],
    columns: list[str],
    values: str,
    what: str,
) -> pd.DataFrame:
    assert_unique(df, keys=index + columns, what=what)
    return df.pivot(index=index, columns=columns, values=values)


def save_plot(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# VIEW SELECTION (multilingual vs isolated)
# -----------------------------------------------------------------------------
def get_data_for_setup(
    master_df: pd.DataFrame,
    *,
    setup: str,  # "multilingual" or "isolated"
    setting: str = "zero_shot",
    include_mwe_segment: bool = True,
    train_lang_joint: str = TRAIN_LANG_JOINT,
    drop_probe_runs: bool = True,
) -> pd.DataFrame:
    df = master_df.copy()
    for c in ["setting", "language_mode", "language", "eval_language", "model_family", "run_dir"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    df = df[(df["setting"] == setting) & (df["include_mwe_segment"] == include_mwe_segment)].copy()

    if drop_probe_runs and "run_dir" in df.columns:
        df = df[~df["run_dir"].str.contains("probe", case=False, na=False)].copy()

    df = normalize_variant(df)
    df = normalize_context(df)

    if setup == "multilingual":
        subset = df[
            (df["language_mode"] == "multilingual") &
            (df["language"] == train_lang_joint) &
            (df["eval_language"].isin(EVAL_ORDER_MULTI))
        ].copy()
        subset["language_setup"] = "joint"
        subset["eval_language"] = pd.Categorical(subset["eval_language"], categories=EVAL_ORDER_MULTI, ordered=True)
        subset["language_setup"] = pd.Categorical(subset["language_setup"], categories=LANGUAGE_SETUP_ORDER, ordered=True)
        return subset

    if setup == "isolated":
        per_lang = df[
            (df["language_mode"] == "per_language") &
            (df["language"] == df["eval_language"]) &
            (df["eval_language"].isin(["EN", "PT"]))
        ].copy()
        per_lang["language_setup"] = "isolated"

        joint_from_multi = df[
            (df["language_mode"] == "multilingual") &
            (df["language"] == train_lang_joint) &
            (df["eval_language"] == "Joint")
        ].copy()
        joint_from_multi["language_setup"] = "joint"

        subset = pd.concat([per_lang, joint_from_multi], ignore_index=True)
        subset = subset[subset["eval_language"].isin(EVAL_ORDER_ISO)].copy()

        subset["eval_language"] = pd.Categorical(subset["eval_language"], categories=EVAL_ORDER_ISO, ordered=True)
        subset["language_setup"] = pd.Categorical(subset["language_setup"], categories=LANGUAGE_SETUP_ORDER, ordered=True)
        return subset

    raise ValueError(f"Unknown setup: {setup}")


# -----------------------------------------------------------------------------
# TABLES (adapted: no aggregation)
# -----------------------------------------------------------------------------
def table_baseline_scoreboard(df_setup: pd.DataFrame, *, eval_order: list[str]) -> pd.DataFrame:
    """
    Baseline slice: Full + Standard.
    One row per model_family. Columns: eval_language.
    Assumes df_setup is already filtered to ONE setup (isolated OR multilingual).
    """
    df = df_setup.copy()
    df = df[(df["context_label"] == "Full") & (df["variant"] == "Standard")].copy()

    tab = pivot_strict(
        df,
        index=["model_family"],
        columns=["eval_language"],
        values="macro_f1",
        what="table_baseline_scoreboard",
    )

    col_order = [c for c in eval_order if c in tab.columns]
    tab = tab.reindex(columns=col_order)

    tab.columns = pd.MultiIndex.from_product([["macro-F1"], tab.columns.tolist()])
    return tab


def table2_context_variant(df_setup: pd.DataFrame, *, eval_order: list[str]) -> dict[str, pd.DataFrame]:
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


# -----------------------------------------------------------------------------
# RQ5: Cross lingual transfer
# -----------------------------------------------------------------------------

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

def plot_f1_over_variants_lines(
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

    # ONLY EN/PT (no GL, no Joint)
    present = set(df["eval_language"].astype(str))
    candidates = [l for l in eval_order if l in present and l in ("EN", "PT")]
    if not candidates:
        return df

    df = df[df["eval_language"].astype(str).isin(candidates)].copy()

    assert_unique(
        df,
        keys=["eval_language", "context_label", "variant"],
        what="plot_f1_over_variants_lines uniqueness",
    )

    df["variant"] = pd.Categorical(df["variant"].astype(str), categories=VARIANT_ORDER, ordered=True)
    df["context_label"] = pd.Categorical(df["context_label"].astype(str), categories=CONTEXT_ORDER, ordered=True)
    df["eval_language"] = pd.Categorical(df["eval_language"].astype(str), categories=candidates, ordered=True)

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
        estimator=np.mean,  # NO-OP due to assert_unique
        errorbar=None,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel("Input variant")
    ax.set_ylabel("Macro-F1")
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylim(0, 1.0)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="")

    save_plot(fig, save_path)
    return out


def plot_joint_over_variants(
    df_setup: pd.DataFrame,
    save_path: Path,
    title: str,
    metric: str = "macro_f1",
) -> pd.DataFrame:
    """
    Joint-only plot (matches styling of plot_f1_over_variants_lines):
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

    # Match text/formatting with plot_f1_over_variants_lines
    ax.set_title(title)
    ax.set_xlabel("Input variant")
    ax.set_ylabel("Macro-F1")
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylim(0, 1.0)

    # Match legend placement
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="context_label")

    save_plot(fig, save_path)
    return out



# -----------------------------------------------------------------------------
# One-shot vs zero-shot (BASECASE) 
# -----------------------------------------------------------------------------
def filter_basecase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basecase slice:
      - variant == Standard
      - context_label == Full
    Uses normalized columns produced by get_data_for_setup.
    """
    q = df[(df["variant"] == "Standard") & (df["context_label"] == "Full")].copy()
    return q


def compute_one_shot_gain_basecase(
    master_df: pd.DataFrame,
    setup: str,  # "isolated" or "multilingual"
    include_mwe_segment: bool = True,
    train_lang_joint: str = "EN_PT_GL",
    eval_languages: tuple[str, ...] = ("EN", "PT"),  # no Joint per your latest decision
    metric: str = "macro_f1",
    drop_probe_runs: bool = True,
) -> pd.DataFrame:
    """
    Returns long df with:
      model_family, eval_language, gain
    where gain = one_shot - zero_shot, computed on the basecase slice.
    """

    # 1) Build the two setup-consistent views (reusing your selector)
    df_zero = get_data_for_setup(
        master_df,
        setup=setup,
        setting="zero_shot",
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        drop_probe_runs=drop_probe_runs,
    )
    df_one = get_data_for_setup(
        master_df,
        setup=setup,
        setting="one_shot",
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        drop_probe_runs=drop_probe_runs,
    )

    # 2) Filter to basecase
    df_zero = filter_basecase(df_zero)
    df_one = filter_basecase(df_one)

    # 3) Keep only EN/PT (explicitly no Joint here)
    df_zero = df_zero[df_zero["eval_language"].astype(str).isin(eval_languages)].copy()
    df_one = df_one[df_one["eval_language"].astype(str).isin(eval_languages)].copy()

    if df_zero.empty or df_one.empty:
        raise ValueError(
            f"[one-shot gain] Empty basecase slice after filtering. "
            f"setup={setup}, eval_languages={eval_languages}. "
            f"zero_rows={len(df_zero)}, one_rows={len(df_one)}"
        )

    # 4) No aggregation allowed: ensure one value per cell
    cell_keys = ["model_family", "eval_language"]
    assert_unique(df_zero, keys=cell_keys, what=f"one-shot gain basecase ZERO ({setup})")
    assert_unique(df_one, keys=cell_keys, what=f"one-shot gain basecase ONE ({setup})")

    # 5) Align and compute gain
    z = df_zero.set_index(cell_keys)[metric].rename("zero_shot")
    o = df_one.set_index(cell_keys)[metric].rename("one_shot")

    # ensure same coverage
    if not z.index.equals(o.index):
        missing_in_one = z.index.difference(o.index)
        missing_in_zero = o.index.difference(z.index)
        raise ValueError(
            f"[one-shot gain] Coverage mismatch between zero_shot and one_shot for setup={setup}.\n"
            f"Missing in one_shot: {list(missing_in_one)[:20]}\n"
            f"Missing in zero_shot: {list(missing_in_zero)[:20]}"
        )

    gain = (o - z).rename("gain").reset_index()
    gain["eval_language"] = gain["eval_language"].astype(str)

    return gain


def plot_one_shot_gain_grouped_bars(
    gain_df: pd.DataFrame,
    save_path: Path,
    title: str,
    eval_order: list[str] = ["EN", "PT"],
) -> None:
    """
    Grouped gain-only bars:
      x   = model_family
      hue = eval_language (EN/PT)
      y   = gain
    """
    df = gain_df.copy()
    df["model_family"] = df["model_family"].astype(str)
    df["eval_language"] = df["eval_language"].astype(str).str.strip()

    present_langs = [l for l in eval_order if l in set(df["eval_language"])]
    df = df[df["eval_language"].isin(present_langs)].copy()
    df["eval_language"] = pd.Categorical(df["eval_language"], categories=present_langs, ordered=True)

    model_order = sorted(df["model_family"].unique())

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(9.0, 4.2))

    sns.barplot(
        data=df,
        x="model_family",
        y="gain",
        hue="eval_language",
        order=model_order,
        hue_order=present_langs,
        errorbar=None,
        ax=ax,
    )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Model family")
    ax.set_ylabel("Gain (one_shot − zero_shot)")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Eval language", bbox_to_anchor=(1.02, 1), loc="upper left")

    save_plot(fig, save_path)


def plot_one_shot_gain_heatmap(
    gain_df: pd.DataFrame,
    save_path: Path,
    *,
    title: str,
    eval_order: list[str] = ["EN", "PT"],
    figsize: tuple[int, int] = (6, 3),
) -> pd.DataFrame:
    """
    Heatmap: rows=model_family, cols=eval_language (EN/PT), values=gain.
    """
    df = gain_df.copy()
    df["model_family"] = df["model_family"].astype(str)
    df["eval_language"] = df["eval_language"].astype(str).str.strip()

    present_langs = [l for l in eval_order if l in set(df["eval_language"])]
    df = df[df["eval_language"].isin(present_langs)].copy()

    # strict cell uniqueness
    assert_unique(df, keys=["model_family", "eval_language"], what="one-shot gain heatmap cells")

    mat = df.pivot(index="model_family", columns="eval_language", values="gain")
    mat = mat.reindex(index=sorted(mat.index), columns=present_langs)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(mat, annot=True, fmt=".3f", linewidths=0.5, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Eval language")
    ax.set_ylabel("Model family")

    save_plot(fig, save_path)
    return mat


# -----------------------------------------------------------------------------
# Runner: two setups, two plots each
# -----------------------------------------------------------------------------
def run_one_shot_gain_two_setups(
    master_df: pd.DataFrame,
    results_root: Path,
    setup: str = "isolated",
    include_mwe_segment: bool = True,
    train_lang_joint: str = "EN_PT_GL",
    eval_languages: tuple[str, ...] = ("EN", "PT"),  # no Joint
    metric: str = "macro_f1",
) -> None:

    gain_df = compute_one_shot_gain_basecase(
        master_df,
        setup=setup,
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        eval_languages=eval_languages,
        metric=metric,
    )

    out_dir = results_root / "one_shot_vs_zero_shot" / setup
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_one_shot_gain_grouped_bars(
        gain_df,
        out_dir / "gain_grouped_bars.png",
        title=f"One-shot gain (basecase) — {setup} — {metric}",
        eval_order=list(eval_languages),
    )

    plot_one_shot_gain_heatmap(
        gain_df,
        out_dir / "gain_heatmap.png",
        title=f"One-shot gain heatmap (basecase) — {setup} — {metric}",
        eval_order=list(eval_languages),
        figsize=(6, 3),
    )

    gain_df.to_csv(out_dir / "gain_table.csv", index=False)



def run_analysis(master_df: pd.DataFrame, results_root: Path, *, setting: str = "zero_shot") -> None:
    for setup, eval_order in [("multilingual", EVAL_ORDER_MULTI), ("isolated", EVAL_ORDER_ISO)]:
        df_setup = get_data_for_setup(master_df, setup=setup, setting=setting)

        run_one_shot_gain_two_setups(master_df, results_root, setup, 
                                     train_lang_joint = "EN_PT_GL", 
                                     eval_languages = ("EN", "PT")
                                    )
        
        # table1 baseline
        baseline_table = table_baseline_scoreboard(df_setup, eval_order=eval_order)
        save_multicol_latex(baseline_table, results_root, f"table__baseline__{setup}", decimals=3)

        for mf in sorted(df_setup["model_family"].dropna().unique()):
            df_mf = df_setup[df_setup["model_family"] == mf].copy()
            out_dir = results_root / setup / mf
            out_dir.mkdir(parents=True, exist_ok=True)
            
            safe_mf = str(mf).replace("/", "_").replace(" ", "_")

            # table2 context x variant
            variant_table = table2_context_variant(df_mf, eval_order=eval_order)
            for k, tab in variant_table.items():
                save_multicol_latex(tab, out_dir, f"table_context_variant__{k}__{setup}", decimals=3)


            plot_f1_over_variants_lines(
                df_mf,
                out_dir / "plot_variants_lines.png",
                title=f"{setup.title()} | {mf} | Macro-F1 over variants ({setting})",
                eval_order=eval_order,
            )

            plot_joint_over_variants(
                df_mf,
                out_dir / "plot_variants_joint_only.png",
                title=f"{setup.title()} | {mf} | Joint macro-F1 over variants ({setting})",
            )

            # EN–PT gap
            if set(["EN", "PT"]).issubset(set(df_mf["eval_language"].astype(str))):
                plot_en_pt_gap(
                    df_mf,
                    out_dir / "plot_en_pt_gap.png",
                    title=f"{setup.title()} | {mf} | EN–PT gap ({setting})",
                )