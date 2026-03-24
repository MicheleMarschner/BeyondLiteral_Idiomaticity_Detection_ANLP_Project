from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from utils.helper import ensure_dir

# -----------------------------------------------------------------------------
# constants
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

TRAINING_SETUP_ORDER = ["monolingual", "multilingual"]
TRAIN_LANG_JOINT = "EN_PT_GL"

PT_TRANSFER_ORDER = ["PT mono", "EN+PT multi", "EN→PT"]
CLASS_RECALL_ORDER = ["Idiomatic", "Literal"]


def create_folder_structure(results_sub_dir: Path):
    
    plots_path = results_sub_dir / "plots"
    ensure_dir(plots_path)

    tables_path = results_sub_dir / "tables"
    ensure_dir(tables_path)

    return tables_path, plots_path


# -----------------------------------------------------------------------------
# control helpers
# -----------------------------------------------------------------------------
def assert_unique(
    df: pd.DataFrame,
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
    index: list[str],
    columns: list[str],
    values: str,
    what: str,
) -> pd.DataFrame:
    assert_unique(df, keys=index + columns, what=what)
    return df.pivot(index=index, columns=columns, values=values)


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
    """Save one MultiIndex-column table as LaTeX with grouped headers (booktabs)."""
    df = table.copy().round(decimals)

    float_fmt = f"%.{decimals}f"
    tex = df.to_latex(
        escape=True,
        multicolumn=True,
        multicolumn_format="c",
        na_rep="-",
        float_format=float_fmt,
    )

    tex = tex.replace("DELTA\\_TMP", "$\\Delta$")

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

