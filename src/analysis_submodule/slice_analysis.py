import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from pathlib import Path

from analysis_submodule.utils.helper import normalize_variant, save_plot

# -------------------------
# Small clean helpers
# -------------------------
def describe_run(df_run: pd.DataFrame) -> dict:
    """
    Return a compact descriptor for a run_dir.
    Assumes df_run is already filtered to ONE run_dir.
    """
    cols = [
        "run_dir", "setting", "language_mode", "language", "model_family", "seed",
        "context", "features", "transform", "include_mwe_segment",
    ]
    first = df_run.iloc[0]
    desc = {c: first.get(c) for c in cols if c in df_run.columns}

    # Reuse normalize_variant(df): build a 1-row df and read the computed label
    one = pd.DataFrame([{"transform": desc.get("transform"), "features": desc.get("features")}])
    one = normalize_variant(one)
    desc["variant"] = str(one.loc[0, "variant"])

    return desc

def macro_f1_safe(y_true, y_pred) -> float:
    """Macro-F1, returns NaN if slice empty."""
    if len(y_true) == 0:
        return float("nan")
    return float(f1_score(y_true, y_pred, average="macro"))


# -------------------------
# 1) Single-run hard/control gap (dynamic by run_dir)
# -------------------------
def hard_control_gap_for_run(
    slices_overview: pd.DataFrame,
    *,
    run_dir: str,
    language: str | None = None,   # e.g. "EN" / "PT" ; None => use all languages in that run
) -> tuple[dict, pd.DataFrame]:
    """
    Compute macro-F1 on slice_ambiguous == hard vs control for ONE run_dir.
    Returns: (run_descriptor, result_df)
    result_df columns: language, f1_hard, f1_control, gap_hard_minus_control, n_hard, n_control
    """
    required = {"run_dir", "slice_ambiguous", "label", "test_pred", "language"}
    missing = required - set(slices_overview.columns)
    if missing:
        raise ValueError(f"slices_overview missing columns: {sorted(missing)}")

    df_run = slices_overview[slices_overview["run_dir"] == run_dir].copy()
    if df_run.empty:
        raise ValueError(f"No rows found for run_dir='{run_dir}'")

    # optional language filter
    if language is not None:
        df_run = df_run[df_run["language"] == language].copy()
        if df_run.empty:
            raise ValueError(f"No rows for run_dir='{run_dir}' with language='{language}'")

    # ensure we only evaluate hard/control
    df_run = df_run[df_run["slice_ambiguous"].isin(["hard", "control"])].copy()
    if df_run.empty:
        raise ValueError(f"run_dir='{run_dir}': slice_ambiguous has no 'hard'/'control' rows")

    desc = describe_run(df_run)

    rows = []
    for lang in sorted(df_run["language"].dropna().unique()):
        sub_lang = df_run[df_run["language"] == lang]

        hard = sub_lang[sub_lang["slice_ambiguous"] == "hard"]
        control = sub_lang[sub_lang["slice_ambiguous"] == "control"]

        f1_h = macro_f1_safe(hard["label"], hard["test_pred"])
        f1_c = macro_f1_safe(control["label"], control["test_pred"])

        rows.append({
            "language": lang,
            "f1_hard": f1_h,
            "f1_control": f1_c,
            "gap_hard_minus_control": (f1_h - f1_c) if (pd.notna(f1_h) and pd.notna(f1_c)) else float("nan"),
            "n_hard": int(len(hard)),
            "n_control": int(len(control)),
        })

    res = pd.DataFrame(rows).sort_values("language")
    return desc, res


# -------------------------
# 2) Aggregate plot across runs (mean over seeds), by model_family × context × variant
# -------------------------
def compute_hard_control_gap_all_runs(slices_overview: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-run, per-language hard/control macro-F1 and gap.
    Output includes run metadata for grouping/plotting.
    """
    required = {"run_dir", "slice_ambiguous", "label", "test_pred", "language",
                "model_family", "context", "transform", "features", "setting", "seed"}
    missing = required - set(slices_overview.columns)
    if missing:
        raise ValueError(f"slices_overview missing columns: {sorted(missing)}")

    df = slices_overview[slices_overview["slice_ambiguous"].isin(["hard", "control"])].copy()
    if df.empty:
        raise ValueError("No rows with slice_ambiguous in {'hard','control'} found.")

    df = normalize_variant(df)

    out_rows = []
    group_cols = ["run_dir", "language"]
    for (rd, lang), g in df.groupby(group_cols, dropna=False):
        hard = g[g["slice_ambiguous"] == "hard"]
        control = g[g["slice_ambiguous"] == "control"]

        f1_h = macro_f1_safe(hard["label"], hard["test_pred"])
        f1_c = macro_f1_safe(control["label"], control["test_pred"])

        first = g.iloc[0]
        out_rows.append({
            "run_dir": rd,
            "language": lang,
            "setting": first["setting"],
            "model_family": first["model_family"],
            "context": first["context"],
            "variant": first["variant"],
            "seed": first["seed"],
            "f1_hard": f1_h,
            "f1_control": f1_c,
            "gap_hard_minus_control": (f1_h - f1_c) if (pd.notna(f1_h) and pd.notna(f1_c)) else float("nan"),
        })

    return pd.DataFrame(out_rows)


def plot_hard_control_gap_aggregated(
    gaps_df: pd.DataFrame,
    save_path: Path | None = None,
    languages: list[str] = ("EN", "PT"),
) -> None:
    """
    Aggregated barplot: mean gap (hard-control) over runs (typically over seeds).
    Facet by model_family. Hue=context. X=variant. One plot per language (row facets).
    """
    plot_df = gaps_df[gaps_df["language"].isin(languages)].copy()
    if plot_df.empty:
        raise ValueError(f"No rows for languages={languages} in gaps_df")

    # nice ordering for x axis
    variant_order = ["Standard", "Highlight", "Highlight + NER", "Highlight + Glosses", "NER", "Glosses"]
    plot_df["variant"] = pd.Categorical(plot_df["variant"], categories=variant_order, ordered=True)

    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=plot_df,
        kind="bar",
        x="variant",
        y="gap_hard_minus_control",
        hue="context",
        col="model_family",
        row="language",
        estimator="mean",
        errorbar="se",
        height=3.6,
        aspect=1.5,
    )

    for ax in g.axes.flat:
        ax.axhline(0, color="black", linewidth=1)
        ax.tick_params(axis="x", rotation=25)

    g.set_axis_labels("Variant", "Gap (macro-F1 hard − control)")
    g.set_titles("{row_name} | {col_name}")
    g.fig.tight_layout()

    
    save_plot(g.fig, save_path)



# ----------------------------
# RQ2 support: hard vs control delta (from slice_metrics_long.csv)
# ----------------------------
def rq2_plot_hard_control_delta(
    df_slices: pd.DataFrame,
    df_base: pd.DataFrame,
    out_path: Path,
    *,
    hard_label: str,
    control_label: str,
    aggregate: bool = True,
    metric: str = "macro_f1",
) -> None:
    """
    Barplot of (macro_f1 on hard slice) - (macro_f1 on control slice), by run config.
    If aggregate=True: average per (setting, context|signal) with SE error bars.
    """
    required_slices = {"run_dir", "slice", metric}
    missing = required_slices - set(df_slices.columns)
    if missing:
        raise ValueError(f"df_slices missing columns: {sorted(missing)}")

    required_base = {"run_dir", "setting", "context", "features", "transform"}
    missing = required_base - set(df_base.columns)
    if missing:
        raise ValueError(f"df_base missing columns: {sorted(missing)}")

    need = df_slices[df_slices["slice"].isin([hard_label, control_label])].copy()
    if need.empty:
        raise ValueError(
            f"No rows for slices '{hard_label}'/'{control_label}'. "
            f"Available slice labels: {sorted(df_slices['slice'].dropna().unique())}"
        )

    wide = (
        need.pivot_table(index="run_dir", columns="slice", values=metric, aggfunc="first")
        .reset_index()
    )
    if hard_label not in wide.columns or control_label not in wide.columns:
        raise ValueError(
            f"After pivot, missing '{hard_label}' or '{control_label}'. "
            f"Have: {wide.columns.tolist()}"
        )

    wide["delta_hard_minus_control"] = wide[hard_label] - wide[control_label]

    join_base = df_base[list(required_base)].drop_duplicates(subset=["run_dir"]).copy()
    join_base = normalize_features(join_base)

    df = wide.merge(join_base, on="run_dir", how="left")
    df = normalize_features(df)
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
