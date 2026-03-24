from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from analysis_submodule.analysis_Michele.utils.data_views import filter_baseline, get_data_for_setup
from analysis_submodule.analysis_Michele.utils.helper_analysis import assert_unique, save_plot
from config import PATHS
from evaluation.metrics import compute_metrics
from analysis_submodule.analysis_Michele.utils.plots import add_y_margin_for_annotations, annotate_bar_values, pretty_eval_language, pretty_model_family, pretty_title
from utils.helper import ensure_dir

# -----------------------------------------------------------------------------
# helpers to create overlap between one-shot and zero-shot test samples
# -----------------------------------------------------------------------------
def _resolve_experiment_dir(experiments_root: Path, run_dir: str) -> Path:
    """Resolve run_dir to the actual experiment folder on disk"""
    exp_path = experiments_root / run_dir
    if exp_path.exists():
        return exp_path

    candidates = [
        run_dir.replace("EN_PT_GL", "EN,PT,GL"),
        run_dir.replace("EN_PT", "EN,PT"),
        run_dir.replace("EN_GL", "EN,GL"),
        run_dir.replace("PT_GL", "PT,GL"),
    ]

    for candidate in candidates:
        path = experiments_root / candidate
        if path.exists():
            return path
        

def _norm_text(x: object) -> str:
    """Normalize text minimally for exact sentence-key matching"""
    if pd.isna(x):
        return ""
    return " ".join(str(x).strip().split())


def _make_prev_target_next_key(df: pd.DataFrame) -> pd.Series:
    """Build exact match key from Previous + Target + Next"""
    
    prev = df["Previous"].map(_norm_text)
    target = df["Target"].map(_norm_text)
    next = df["Next"].map(_norm_text)
    
    return prev + " ||| " + target + " ||| " + next


def _load_predictions_with_text_key(
    pred_csv: Path,
    split_csv: Path,
    split_id_col: str = "ID",
) -> pd.DataFrame:
    """ Load one run's predictions and attach Language + exact sentence key
    from the corresponding test split file
    """

    pred = pd.read_csv(pred_csv)
    split = pd.read_csv(split_csv)

    # normalize prediction column names
    pred = pred.rename(
        columns={
            "test_pred": "pred",
            "test_proba_literal": "proba",
        }
    ).copy()

    # normalize id column in split
    split = split.rename(columns={split_id_col: "id"}).copy()

    pred["id"] = pred["id"].astype(str)
    split["id"] = split["id"].astype(str)

    split_meta = split[["id", "Language", "Previous", "Target", "Next"]].copy()
    split_meta["match_key"] = _make_prev_target_next_key(split_meta)

    out = pred.merge(split_meta[["id", "Language", "match_key"]], on="id", how="left")

    return out

def _macro_f1_on_overlap_subset(
    pred_df: pd.DataFrame,
    overlap_keys: set[str],
    eval_language: str,
) -> float:
    """Recompute macro-F1 on the overlap subset for one eval language"""
    subset = pred_df[pred_df["match_key"].isin(overlap_keys)].copy()

    if eval_language != "Joint":
        subset = subset[subset["Language"].astype(str) == str(eval_language)].copy()

    return float(compute_metrics(subset["label"], subset["pred"])["macro_f1"])


def compute_one_shot_gain_baseline_overlap(
    master_df: pd.DataFrame,
    experiments_root: Path,
    one_shot_test_csv: Path,
    zero_shot_test_csv: Path,
    setup: str,
    include_mwe_segment: bool = True,
    train_lang_joint: str = "EN_PT_GL",
    eval_languages: tuple[str, ...] = ("EN", "PT"),
    drop_probe_runs: bool = True,
) -> pd.DataFrame:
    """
    """
    _, _, pairs = _prepare_one_zero_baseline_pairs(
        master_df=master_df,
        setup=setup,
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        eval_languages=eval_languages,
        drop_probe_runs=drop_probe_runs,
    )

    rows = []

    for _, row in pairs.iterrows():
        run_dir_one = row["run_dir_one"]
        run_dir_zero = row["run_dir_zero"]
        model_family = row["model_family"]
        eval_language = str(row["eval_language"])

        one_dir = _resolve_experiment_dir(experiments_root, run_dir_one)
        zero_dir = _resolve_experiment_dir(experiments_root, run_dir_zero)

        pred_one = _load_predictions_with_text_key(
            one_dir / "test_predictions.csv",
            one_shot_test_csv,
            split_id_col="ID",
        )
        pred_zero = _load_predictions_with_text_key(
            zero_dir / "test_predictions.csv",
            zero_shot_test_csv,
            split_id_col="ID",
        )

        overlap_keys = set(pred_one["match_key"]).intersection(set(pred_zero["match_key"]))
    
        one_lang_keys = set(
            pred_one.loc[pred_one["Language"].astype(str) == eval_language, "match_key"]
        )
        zero_lang_keys = set(
            pred_zero.loc[pred_zero["Language"].astype(str) == eval_language, "match_key"]
        )
        overlap_lang_keys = one_lang_keys.intersection(zero_lang_keys)

        print(
            f"[INFO] {model_family} | eval={eval_language} | "
            f"overlap={len(overlap_lang_keys)} | "
            f"one_unique={len(one_lang_keys)} | "
            f"zero_unique={len(zero_lang_keys)} | "
            f"max_possible={min(len(one_lang_keys), len(zero_lang_keys))}"
        )

        one_score = _macro_f1_on_overlap_subset(pred_one, overlap_keys, eval_language)
        zero_score = _macro_f1_on_overlap_subset(pred_zero, overlap_keys, eval_language)

        one_subset = pred_one[pred_one["match_key"].isin(overlap_keys)].copy()
        if eval_language != "Joint":
            one_subset = one_subset[
                one_subset["Language"].astype(str) == eval_language
            ].copy()

        rows.append(
            {
                "model_family": model_family,
                "eval_language": eval_language,
                "run_dir_one": run_dir_one,
                "run_dir_zero": run_dir_zero,
                "one_shot_overlap": one_score,
                "zero_shot_overlap": zero_score,
                "gain": one_score - zero_score,
                "n_overlap": len(one_subset),
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# simple One Shot vs Zero Shot (input-variant baseline) 
# -----------------------------------------------------------------------------
def _prepare_one_zero_baseline_pairs(
    master_df: pd.DataFrame,
    setup: str,
    include_mwe_segment: bool = True,
    train_lang_joint: str = "EN_PT_GL",
    eval_languages: tuple[str, ...] = ("EN", "PT"),
    drop_probe_runs: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare aligned one-shot / zero-shot baseline views and run pairs"""
    
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

    df_zero = filter_baseline(df_zero)
    df_one = filter_baseline(df_one)

    df_zero = df_zero[df_zero["eval_language"].astype(str).isin(eval_languages)].copy()
    df_one = df_one[df_one["eval_language"].astype(str).isin(eval_languages)].copy()

    cell_keys = ["model_family", "eval_language"]
    assert_unique(df_zero, keys=cell_keys, what=f"baseline ZERO ({setup})")
    assert_unique(df_one, keys=cell_keys, what=f"baseline ONE ({setup})")

    pairs = df_one[["run_dir", "model_family", "eval_language"]].rename(
        columns={"run_dir": "run_dir_one"}
    ).merge(
        df_zero[["run_dir", "model_family", "eval_language"]].rename(
            columns={"run_dir": "run_dir_zero"}
        ),
        on=["model_family", "eval_language"],
        how="inner",
        validate="one_to_one",
    )

    return df_one, df_zero, pairs


def compute_one_shot_gain_baseline(
    master_df: pd.DataFrame,
    setup: str,
    include_mwe_segment: bool = True,
    train_lang_joint: str = "EN_PT_GL",
    eval_languages: tuple[str, ...] = ("EN", "PT"),
    metric: str = "macro_f1",
    drop_probe_runs: bool = True,
) -> pd.DataFrame:
    """
    Returns long df with:
      model_family, eval_language, gain
    where gain = one_shot - zero_shot, computed on the baseline slice.
    """
    df_one, df_zero, _ = _prepare_one_zero_baseline_pairs(
        master_df=master_df,
        setup=setup,
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        eval_languages=eval_languages,
        drop_probe_runs=drop_probe_runs,
    )

    cell_keys = ["model_family", "eval_language"]

    zero = df_zero.set_index(cell_keys)[metric].rename("zero_shot")
    one = df_one.set_index(cell_keys)[metric].rename("one_shot")

    gain = (one - zero).rename("gain").reset_index()
    gain["eval_language"] = gain["eval_language"].astype(str)

    return gain


def plot_one_shot_gain_grouped_bars(
    gain_df: pd.DataFrame,
    save_path: Path,
    title: str,
    eval_order: list[str] = ["EN", "PT"],
) -> None:
    """Grouped gain-only bar plot"""
    df = gain_df.copy()
    df["model_family"] = df["model_family"].astype(str)
    df["eval_language"] = df["eval_language"].astype(str).str.strip()

    present_langs = [l for l in eval_order if l in set(df["eval_language"])]
    df = df[df["eval_language"].isin(present_langs)].copy()
    df["eval_language"] = pd.Categorical(df["eval_language"], categories=present_langs, ordered=True)

    model_order = sorted(df["model_family"].unique())

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7.4, 3.9))

    sns.barplot(
        data=df,
        x="model_family",
        y="gain",
        hue="eval_language",
        order=model_order,
        hue_order=present_langs,
        errorbar=None,
        width=0.55,
        ax=ax,
    )

    ax.set_title(pretty_title(title), fontsize=11, pad=12)
    ax.set_xlabel("Model family", fontsize=10, labelpad=10)
    ax.set_ylabel("Δ macro-F1 (One Shot − Zero Shot)", fontsize=10, labelpad=12)
    ax.axhline(0, color="0.55", linewidth=0.8)

    ax.tick_params(axis="x", rotation=25, labelsize=9, pad=6)
    ax.tick_params(axis="y", labelsize=9, pad=4)

    ax.grid(axis="y", linewidth=0.6, alpha=0.30)
    ax.grid(axis="x", visible=False)

    ticks = ax.get_xticks()
    labels = [pretty_model_family(t.get_text()) for t in ax.get_xticklabels()]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=25, ha="right")

    legend = ax.legend(
        title="Eval language",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=9,
        title_fontsize=9,
        borderaxespad=0.3,
        labelspacing=0.4,
        handletextpad=0.6,
    )

    if legend is not None:
        for text in legend.texts:
            text.set_text(pretty_eval_language(text.get_text()))

    add_y_margin_for_annotations(ax, top_frac=0.10, bottom_frac=0.07)
    annotate_bar_values(ax, decimals=3, fontsize=8, show_zero=False)

    fig.subplots_adjust(right=0.84, top=0.88)

    save_plot(fig, save_path)



# -----------------------------------------------------------------------------
# main function
# -----------------------------------------------------------------------------
def run_one_shot_experiment(
    master_df: pd.DataFrame,
    results_root: Path,
    setup: str = "monolingual",
    include_mwe_segment: bool = True,
    train_lang_joint: str = "EN_PT_GL",
    eval_languages: tuple[str, ...] = ("EN", "PT"),  # no Joint
    metric: str = "macro_f1",
) -> None:
    one_shot_test_csv = PATHS.data_preprocessed / "one_shot_splits/one_shot_test.csv"
    zero_shot_test_csv = PATHS.data_preprocessed / "zero_shot_splits/zero_shot_test.csv"


    delta_zero_one_df = compute_one_shot_gain_baseline(
        master_df,
        setup=setup,
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        eval_languages=eval_languages,
        metric=metric,
    )

    delta_zero_one_overlap_df = compute_one_shot_gain_baseline_overlap(
        master_df=master_df,
        experiments_root=PATHS.runs,
        one_shot_test_csv=one_shot_test_csv,
        zero_shot_test_csv=zero_shot_test_csv,
        setup=setup,
        include_mwe_segment=include_mwe_segment,
        train_lang_joint=train_lang_joint,
        eval_languages=eval_languages,
    )

    save_dir = results_root / "one_shot_vs_zero_shot" / setup
    ensure_dir(save_dir)

    plot_one_shot_gain_grouped_bars(
        delta_zero_one_df,
        save_dir / "delta_zero_one_grouped_bars.png",
        title=f"One Shot gain over Zero Shot (full test set) - {setup}",
        eval_order=list(eval_languages),
    )
    delta_zero_one_df.to_csv(save_dir / "delta_zero_one_table.csv", index=False)

    plot_one_shot_gain_grouped_bars(
        delta_zero_one_overlap_df,
        save_dir / "delta_zero_one_overlap_grouped_bars.png",
        title=f"One Shot gain over Zero Shot (overlapping subset) - {setup}",
        eval_order=list(eval_languages),
    )
    delta_zero_one_overlap_df.to_csv(save_dir / "delta_zero_one_overlap_table.csv", index=False)