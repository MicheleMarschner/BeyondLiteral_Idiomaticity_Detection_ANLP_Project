
from pathlib import Path
from typing import Tuple
import pandas as pd

from analysis_submodule.analysis_Michele.utils.helper_analysis import CONTEXT_ORDER, EVAL_ORDER_MULTI, TRAINING_SETUP_ORDER, TRAIN_LANG_JOINT, VARIANT_ORDER
from analysis import evaluate_subslices
from evaluation import run_evaluation
from utils.helper import read_csv_data


def load_results_overviews(
    experiments_root: Path,
    results_root: Path,
    results_sub_dir: Path,
    split_type: str = "test",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the main result overviews used in the analysis"""

    master_csv_path = results_root / "master_metrics_long.csv"
    if not master_csv_path.exists():
        run_evaluation()
    master_df = read_csv_data(master_csv_path)

    slice_csv_path = results_root / "slices_overview.csv"
    if not slice_csv_path.exists():
        evaluate_subslices(split_type=split_type)
    slices_df = read_csv_data(slice_csv_path)

    masking_csv_path = results_sub_dir / "stress_masking_summary_monolingual.csv"
    if not masking_csv_path.exists():
       raise FileNotFoundError(
            f"{masking_csv_path} not found. "
            "Stress-masking results require rerunning all experiments with training enabled, "
            "since the model weights are not available in the saved artifacts."
        )

    return master_df, slices_df



# -----------------------------------------------------------------------------
# stress masking views
# -----------------------------------------------------------------------------
def _parse_masking_run_dir_input_block(run_dir: str) -> tuple[str, str, str]:
    """
    Parse the input block from standard run_dir patterns.

    Expected standard monolingual form:
      zero_shot__EN__previous_target_next_True_none_glosses__mBERT__seed51

    We parse from the right so the function remains robust to additional
    prefixes in run_dir, but special runs should still be filtered before use.
    """
    parts = str(run_dir).split("__")
    if len(parts) < 5:
        raise ValueError(
            f"Unexpected masking run_dir format: {run_dir}. "
            "Expected at least 5 '__'-separated parts."
        )

    input_block = parts[-3]

    try:
        context, include_mwe_segment, transform, features = input_block.rsplit("_", 3)
    except ValueError as e:
        raise ValueError(
            f"Could not parse masking input block from run_dir={run_dir}"
        ) from e

    return context, transform, features


def load_stress_masking_monolingual(results_root: Path) -> pd.DataFrame:
    """
    Load monolingual stress-masking summary and normalize it to the same
    context/variant labels used in the main analysis.

    Special runs such as cross-lingual and ModernBERT are excluded here.
    """
    masking_csv_path = results_root / "stress_masking_summary_monolingual.csv"
    if not masking_csv_path.exists():
        raise FileNotFoundError(
            f"{masking_csv_path} not found. "
            "Run the stress-masking analysis first."
        )

    df = read_csv_data(masking_csv_path).copy()

    for c in ["setting", "language_mode", "model_family", "language", "run_dir"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # keep only the standard zero-shot monolingual EN/PT runs used for the masking plot
    if "setting" in df.columns:
        df = df[df["setting"] == "zero_shot"].copy()

    if "language_mode" in df.columns:
        df = df[df["language_mode"] == "per_language"].copy()

    if "model_family" in df.columns:
        df = df[df["model_family"] != "modernBERT"].copy()

    if "language" in df.columns:
        df = df[df["language"].isin(["EN", "PT"])].copy()

    if "run_dir" in df.columns:
        df = df[~df["run_dir"].str.startswith("cross__", na=False)].copy()

    parsed = df["run_dir"].astype(str).apply(_parse_masking_run_dir_input_block)
    parsed_df = pd.DataFrame(
        parsed.tolist(),
        columns=["context", "transform", "features"],
        index=df.index,
    )
    df = pd.concat([df, parsed_df], axis=1)

    df["eval_language"] = df["language"].astype(str).str.strip()

    df = normalize_variant(df)
    df = normalize_context(df)

    df = df[df["eval_language"].isin(["EN", "PT"])].copy()
    df["eval_language"] = pd.Categorical(
        df["eval_language"],
        categories=["EN", "PT"],
        ordered=True,
    )

    return df.sort_values(
        ["model_family", "eval_language", "context_label", "variant"]
    ).reset_index(drop=True)


def summarize_global_n_vs_single_gap(
    df_mask: pd.DataFrame,
    model_family: str = "mBERT",
) -> pd.DataFrame:
    """
    Summarize how far global_n_mask differs from global_single_mask,
    aggregated over variants within each language × context slice.
    """
    df = df_mask.copy()
    df = df[df["model_family"].astype(str) == str(model_family)].copy()

    needed = [
        "eval_language",
        "context_label",
        "variant",
        "delta_macro_f1_global_single_mask",
        "delta_macro_f1_global_n_mask",
    ]
    df = df.dropna(subset=needed).copy()

    if df.empty:
        return df

    df["gap_global_n_minus_single"] = (
        df["delta_macro_f1_global_n_mask"] - df["delta_macro_f1_global_single_mask"]
    )

    out = (
        df.groupby(["eval_language", "context_label"], observed=True)
        .agg(
            mean_abs_gap=("gap_global_n_minus_single", lambda s: float(s.abs().mean())),
            max_abs_gap=("gap_global_n_minus_single", lambda s: float(s.abs().max())),
            mean_signed_gap=("gap_global_n_minus_single", "mean"),
            n_variants=("gap_global_n_minus_single", "size"),
        )
        .reset_index()
        .sort_values(["eval_language", "context_label"])
        .reset_index(drop=True)
    )

    return out

# -----------------------------------------------------------------------------
# normalizations + view selection
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


def get_data_for_setup(
    master_df: pd.DataFrame,
    setup: str,  # "multilingual" or "monolingual"
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
        subset["training_setup"] = "multilingual"
        subset["eval_language"] = pd.Categorical(subset["eval_language"], categories=EVAL_ORDER_MULTI, ordered=True)
        subset["training_setup"] = pd.Categorical(subset["training_setup"], categories=TRAINING_SETUP_ORDER, ordered=True)
        return subset

    if setup == "monolingual":
        per_lang = df[
            (df["language_mode"] == "per_language") &
            (df["language"] == df["eval_language"]) &
            (df["eval_language"].isin(["EN", "PT"]))
        ].copy()
        per_lang["training_setup"] = "monolingual"

        # joint_from_multi = df[
        #     (df["language_mode"] == "multilingual") &
        #     (df["language"] == train_lang_joint) &
        #     (df["eval_language"] == "Joint")
        # ].copy()
        # joint_from_multi["training_setup"] = "multilingual"

        # subset = pd.concat([per_lang, joint_from_multi], ignore_index=True)
        #subset = subset[subset["eval_language"].isin(EVAL_ORDER_ISO)].copy()
        subset = per_lang.copy()

        #subset["eval_language"] = pd.Categorical(subset["eval_language"], categories=EVAL_ORDER_ISO, ordered=True)
        subset["eval_language"] = pd.Categorical(subset["eval_language"], categories=["EN", "PT"], ordered=True)
        subset["training_setup"] = pd.Categorical(subset["training_setup"], categories=TRAINING_SETUP_ORDER, ordered=True)
        return subset

    raise ValueError(f"Unknown setup: {setup}")


def filter_baseline(df: pd.DataFrame) -> pd.DataFrame:

    return df[
        (df["context_label"] == "Full") &
        (df["variant"] == "Standard")
    ].copy()