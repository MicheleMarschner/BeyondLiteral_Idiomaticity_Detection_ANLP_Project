from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from analysis.evaluate_subslices import evaluate_subslices
from analysis_submodule.stress_masking import run_stress_masking_over_all_runs
from evaluation.run_evaluation import run_evaluation
from utils.helper import read_csv_data


VARIANT_ORDER = [
    "Standard",
    "Highlight",
    "Highlight + NER",
    "Highlight + Glosses",
    "NER",
    "Glosses",
]

LANG_ORDER = ["EN", "PT", "Joint"]  # "joint" wird ggf. aus EN/PT berechnet
CONTEXT_ORDER = ["Full", "Target"]  # Full zuerst (dein Wunsch)
METRIC_ORDER = ["Full", "Target", "Δ"]

def normalize_variant(df: pd.DataFrame) -> pd.DataFrame:
    """Create a readable 'variant' label from transform+features."""
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
        # (Kommentar) falls du später combos wie glosses+ner hast, sag Bescheid, dann erweitern wir hier.
        raise ValueError(f"Unexpected transform/features combo: {t}|{f}")

    df["variant"] = [_variant(t, f) for t, f in zip(df["transform"], df["features_norm"])]
    df["variant"] = pd.Categorical(df["variant"], categories=VARIANT_ORDER, ordered=True)
    return df


def normalize_context(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw context to 'Full' vs 'Target'."""
    df = df.copy()
    # (Kommentar) ich nehme an, dass du hier genau diese Werte hast:
    # 'previous_target_next' und 'target'
    context_raw = df["context"].astype(str).str.strip().str.lower()
    mapping = {"previous_target_next": "Full", "target": "Target"}
    df["context_label"] = context_raw.map(mapping)
    if df["context_label"].isna().any():
        bad = sorted(set(df.loc[df["context_label"].isna(), "context"].astype(str)))
        raise ValueError(f"Unexpected context values: {bad}")
    df["context_label"] = pd.Categorical(df["context_label"], categories=CONTEXT_ORDER, ordered=True)
    return df


def add_joint_language(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure eval_language has EN/PT/joint. If joint is missing, compute it as mean(EN, PT)
    per (model_family, setting, include_mwe_segment, context_label, variant).
    """
    df = df.copy()
    df["eval_language"] = df["eval_language"].astype(str).str.strip()
    df["eval_language"] = df["eval_language"].replace({"overall": "Joint"})  # if you ever have overall

    if "Joint" in set(df["eval_language"].unique()):
        return df

    base = df[df["eval_language"].isin(["EN", "PT"])].copy()
    if base.empty:
        return df

    joint = (
        base.groupby(
            ["model_family", "setting", "include_mwe_segment", "context_label", "variant"],
            dropna=False
        )["macro_f1"]
        .mean()
        .reset_index()
    )
    joint["eval_language"] = "Joint"

    return pd.concat([df, joint], ignore_index=True, sort=False)


def filter_baseline(df: pd.DataFrame, *, setting: str, include_mwe_segment: bool = True) -> pd.DataFrame:
    """Filter to the neutral baseline slice you described."""
    df = df.copy()
    df = df[df["setting"] == setting]
    df = df[df["include_mwe_segment"] == include_mwe_segment]
    return df



# ----------------------------
# Small helpers
# ----------------------------
def _short(s: str) -> str:
    return (
        str(s)
        .replace("previous_target_next", "full")
        .replace("target-only", "target")
    )


def save_plot(fig: plt.Figure, out_path: Path) -> None:

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



def load_results_overviews(experiments_root, results_root, results_sub_dir, split_type="test"):
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




def prepare_baseline_master(master_df: pd.DataFrame, *, setting: str = "zero_shot") -> pd.DataFrame:
    """
    Prepare a clean neutral view of the master table:
    - zero-shot only (default)
    - adds 'variant' (transform+features), 'context_label' (Full/Target), and 'joint' language if missing
    """
    df = filter_baseline(master_df, setting=setting, include_mwe_segment=True)
    df = normalize_variant(df)
    df = normalize_context(df)
    df = add_joint_language(df)

    # enforce language order for plotting
    df["eval_language"] = pd.Categorical(df["eval_language"], categories=[l for l in LANG_ORDER if l in set(df["eval_language"])], ordered=True)

    return df


def prepare_master_for_settings(
    master_df: pd.DataFrame,
    settings: list[str],
    include_mwe_segment: bool = True,
) -> pd.DataFrame:
    """
    Prepare master for plots that compare multiple settings (e.g., zero_shot vs one_shot).
    Adds variant + context_label + Joint if missing.
    """
    df = master_df.copy()
    df = df[df["setting"].isin(settings)].copy()
    df = df[df["include_mwe_segment"] == include_mwe_segment].copy()

    df = normalize_variant(df)
    df = normalize_context(df)
    df = add_joint_language(df)

    df["eval_language"] = pd.Categorical(
        df["eval_language"],
        categories=[l for l in LANG_ORDER if l in set(df["eval_language"])],
        ordered=True
    )
    return df


def save_table_latex(df: pd.DataFrame, out_dir: Path, name: str, decimals: int = 3):
    out_dir.mkdir(parents=True, exist_ok=True)

    df2 = df.copy()
    for c in df2.columns:
        if pd.api.types.is_numeric_dtype(df2[c]):
            df2[c] = df2[c].round(decimals)

    tex_path = out_dir / f"{name}.tex"

    # LaTeX: paste into \input{...} or copy-paste
    latex = df2.reset_index().to_latex(index=False, escape=True)
    tex_path.write_text(latex, encoding="utf-8")

    return tex_path


# ----------------------------
# Save to LaTeX (no tabulate needed)
# ----------------------------
def save_multicol_latex(
    table: pd.DataFrame,
    out_dir: Path,
    name: str,
    decimals: int = 3,
) -> Path:
    """
    Save one MultiIndex-column table as LaTeX with grouped headers (booktabs).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    df2 = table.copy().round(decimals)

    tex = df2.to_latex(
        escape=True,
        multicolumn=True,
        multicolumn_format="c",
        na_rep="",
    )

    out_path = out_dir / f"{name}.tex"
    out_path.write_text(tex, encoding="utf-8")
    return out_path



def save_tables_latex(tables: dict[str, pd.DataFrame], out_dir: Path, name_prefix: str, decimals: int = 3):
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, df in tables.items():
        safe_key = str(key).replace("/", "_").replace(" ", "_")
        save_table_latex(df, out_dir, f"{name_prefix}__{safe_key}", decimals=decimals)