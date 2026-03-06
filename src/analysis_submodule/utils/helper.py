from pathlib import Path
from typing import Any, Tuple
import pandas as pd
import matplotlib.pyplot as plt

from analysis.evaluate_subslices import evaluate_subslices
from evaluation.run_evaluation import run_evaluation
from analysis_submodule.stress_masking import run_stress_masking_all
from utils.helper import ensure_dir
from utils.helper import read_csv_data


VARIANT_ORDER = [
    "Standard",
    "Highlight",
    "Highlight + NER",
    "Highlight + Glosses",
    "NER",
    "Glosses",
]
EVAL_LANGUAGE_ORDER = ["EN", "PT", "Joint"]  
CONTEXT_ORDER = ["Full", "Target"]  
METRIC_ORDER = ["Full", "Target", "Δ"]
LANGUAGE_SETUP_ORDER = ["isolated", "joint"]


def normalize_variant(df: pd.DataFrame) -> pd.DataFrame:
    """Create a readable 'variant' label from transform+features"""
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
    """Map raw context to 'Full' vs 'Target'"""
    df = df.copy()
    
    context_raw = df["context"].astype(str).str.strip().str.lower()
    mapping = {"previous_target_next": "Full", "target": "Target"}
    df["context_label"] = context_raw.map(mapping)
    if df["context_label"].isna().any():
        bad = sorted(set(df.loc[df["context_label"].isna(), "context"].astype(str)))
        raise ValueError(f"Unexpected context values: {bad}")
    df["context_label"] = pd.Categorical(df["context_label"], categories=CONTEXT_ORDER, ordered=True)
    return df


def normalize_language_setup(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw language mode to isolated and joint labels"""
    
    df = df.copy()
    df["language_mode"] = df["language_mode"].astype(str).str.strip()

    df["language_setup"] = df["language_mode"].map(LANGUAGE_MODE_TO_SETUP)
    if df["language_setup"].isna().any():
        bad_val = sorted(set(df.loc[df["language_setup"].isna(), "language_mode"]))
        raise ValueError(f"Unexpected language_mode values: {bad_val}")

    df["language_setup"] = pd.Categorical(
        df["language_setup"],
        categories=LANGUAGE_SETUP_ORDER,
        ordered=True,
    )
    return df


## !TODO still needed?? I should have another one by now: _filter_basecase_experiment
def filter_baseline(df: pd.DataFrame, *, setting: str, include_mwe_segment: bool = True) -> pd.DataFrame:
    """Filter to the neutral baseline slice you described"""
    df = df.copy()
    df = df[df["setting"] == setting]
    df = df[df["include_mwe_segment"] == include_mwe_segment]
    return df


def load_results_overviews(
    experiments_root: Path,
    results_root: Path,
    results_sub_dir: Path,
    split_type: str = "test",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the main result overviews used in the analysis"""

    master_csv_path = results_root / "master_metrics_long.csv"
    if not master_csv_path.exists():
        run_evaluation()
    master_df = read_csv_data(master_csv_path)

    slice_csv_path = results_root / "slices_overview.csv"
    if not slice_csv_path.exists():
        evaluate_subslices(split_type=split_type)
    slices_df = read_csv_data(slice_csv_path)

    masking_csv_path = results_sub_dir / "stress_masking_summary.csv"
    if not masking_csv_path.exists():
        run_stress_masking_over_all_runs(experiments_root, results_root)
    masking_df = read_csv_data(masking_csv_path)

    return master_df, slices_df, masking_df


def create_folder_structure(results_sub_dir: Path):
    
    plots_path = results_sub_dir / "plots"
    ensure_dir(plots_path)

    tables_path = results_sub_dir / "tables"
    ensure_dir(tables_path)

    return tables_path, plots_path


## !TODO noch gebraucht oder austauschen
def prepare_master_for_settings(
    master_df: pd.DataFrame,
    settings: list[str],
    include_mwe_segment: bool = True,
) -> pd.DataFrame:
    """Prepare the master table for comparing settings."""
    df = master_df.copy()
    df["eval_language"] = df["eval_language"].astype(str).str.strip()

    df = df[df["setting"].isin(settings)].copy()
    df = df[df["include_mwe_segment"] == include_mwe_segment].copy()

    df = normalize_variant(df)
    df = normalize_context(df)

    df["eval_language"] = pd.Categorical(
        df["eval_language"],
        categories=[l for l in EVAL_LANGUAGE_ORDER if l in set(df["eval_language"])],
        ordered=True,
    )
    return df


def prepare_master_for_settings_with_language_setup(
    master_df: pd.DataFrame,
    settings: list[str],
    model_family: str | None = None,
    include_mwe_segment: bool = True,
    eval_languages: tuple[str, ...] = ("EN", "PT", "GL", "Joint"),
) -> pd.DataFrame:
    """Prepare the master table for setting comparisons within language setups."""
    df = master_df.copy()

    df["language"] = df["language"].astype(str).str.strip().str.replace(" ", "", regex=False)
    df["eval_language"] = df["eval_language"].astype(str).str.strip()
    df["language_mode"] = df["language_mode"].astype(str).str.strip()

    df = df[
        (df["setting"].isin(settings)) &
        (df["include_mwe_segment"] == include_mwe_segment) &
        (df["eval_language"].isin(eval_languages))
    ].copy()

    if model_family is not None:
        df = df[df["model_family"] == model_family].copy()

    df = normalize_variant(df)
    df = normalize_context(df)

    is_isolated = (
        (df["language_mode"] == "per_language") &
        (df["language"] == df["eval_language"])
    )
    is_joint = (
        (df["language_mode"] == "multilingual") &
        (df["language"] == "EN_PT_GL")
    )

    df = df[is_isolated | is_joint].copy()

    df["language_setup"] = "isolated"
    df.loc[is_joint.loc[df.index], "language_setup"] = "joint"

    df["language_setup"] = pd.Categorical(
        df["language_setup"],
        categories=LANGUAGE_SETUP_ORDER,
        ordered=True,
    )
    df["variant"] = pd.Categorical(df["variant"], categories=VARIANT_ORDER, ordered=True)
    df["context_label"] = pd.Categorical(df["context_label"], categories=CONTEXT_ORDER, ordered=True)
    df["eval_language"] = pd.Categorical(
        df["eval_language"],
        categories=[l for l in EVAL_LANGUAGE_ORDER if l in set(df["eval_language"].astype(str))],
        ordered=True,
    )

    return df


def prepare_master_with_language_setup(
    master_df: pd.DataFrame,
    setting: str = "zero_shot",
    model_family: str | None = None,        # None = keep all
    include_mwe_segment: bool = True,
    eval_languages: tuple[str, ...] = ("EN", "PT", "GL", "Joint"),
) -> pd.DataFrame:
    """Prepare the master table for isolated versus joint comparisons."""
    df = master_df.copy()

    df["language"] = df["language"].astype(str).str.strip().str.replace(" ", "", regex=False)
    df["eval_language"] = df["eval_language"].astype(str).str.strip()
    df["language_mode"] = df["language_mode"].astype(str).str.strip()

    df = df[
        (df["setting"] == setting) &
        (df["include_mwe_segment"] == include_mwe_segment) &
        (df["eval_language"].isin(eval_languages))
    ].copy()

    if model_family is not None:
        df = df[df["model_family"] == model_family].copy()

    df = normalize_variant(df)
    df = normalize_context(df)

    is_isolated = (
        (df["language_mode"] == "per_language") &
        (df["language"] == df["eval_language"])
    )
    is_joint = (
        (df["language_mode"] == "multilingual") &
        (df["language"] == "EN_PT_GL")
    )

    df = df[is_isolated | is_joint].copy()

    df["language_setup"] = "isolated"
    df.loc[is_joint.loc[df.index], "language_setup"] = "joint"

    df["language_setup"] = pd.Categorical(
        df["language_setup"],
        categories=LANGUAGE_SETUP_ORDER,
        ordered=True,
    )
    df["variant"] = pd.Categorical(df["variant"], categories=VARIANT_ORDER, ordered=True)
    df["context_label"] = pd.Categorical(df["context_label"], categories=CONTEXT_ORDER, ordered=True)
    df["eval_language"] = pd.Categorical(
        df["eval_language"],
        categories=[l for l in EVAL_LANGUAGE_ORDER if l in set(df["eval_language"].astype(str))],
        ordered=True,
    )

    return df


# ----------------------------
# Save tables (Latex) and plots
# ----------------------------
def save_table_latex(df: pd.DataFrame, out_dir: Path, name: str, decimals: int = 3) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    df2 = df.copy()
    for c in df2.columns:
        if pd.api.types.is_numeric_dtype(df2[c]):
            df2[c] = df2[c].round(decimals)

    tex_path = out_dir / f"{name}.tex"

    # LaTeX: output needs to be pasted into \input{...} or copy-paste
    latex = df2.reset_index().to_latex(index=False, escape=True)
    tex_path.write_text(latex, encoding="utf-8")

    return tex_path


def save_multicol_latex(
    table: pd.DataFrame,
    save_dir: Path,
    name: str,
    decimals: int = 3,
) -> Path:
    """Save one MultiIndex-column table as LaTeX with grouped headers (booktabs)"""
    df = table.copy().round(decimals)

    float_fmt = f"%.{decimals}f"
    tex = df.to_latex(
        escape=True,
        multicolumn=True,
        multicolumn_format="c",
        na_rep="",
        float_format=float_fmt,
    )

    save_path = save_dir / f"{name}.tex"
    save_path.write_text(tex, encoding="utf-8")
    return save_path


def save_tables_latex(tables: dict[str, pd.DataFrame], save_dir: Path, name_prefix: str, decimals: int = 3) -> None:
    """Save multiple tables as LaTeX files"""
    for key, df in tables.items():
        safe_key = str(key).replace("/", "_").replace(" ", "_")
        save_table_latex(df, save_dir, f"{name_prefix}__{safe_key}", decimals=decimals)


def save_plot(fig: plt.Figure, out_path: Path) -> None:
    """Save a plot to disk"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_grid(grid, out_path: Path) -> None:
    """Save seaborn FacetGrid/catplot."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.fig.tight_layout()
    grid.fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(grid.fig)

