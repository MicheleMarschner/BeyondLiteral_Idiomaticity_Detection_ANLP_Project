import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd


#TRAIN_PATH = Path("./data/raw/Data/train_zero_shot.csv")
#DEV_PATH = Path("./data/raw/Data/dev.csv")
TRAIN_PATH = Path("./data/preprocessed/zero_shot_splits/zero_shot_train.csv")
DEV_PATH = Path("./data/preprocessed/zero_shot_splits/zero_shot_test.csv")
DEV_GOLD_PATH = Path("./data/raw/Data/dev_gold.csv")
OUTPUT_PATH = Path("./results/eda/dataset_eda_preprocessed_test.md")


def _normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure labels are integers."""
    df = df.copy()
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    return df


def load_data(file_path: Path) -> pd.DataFrame:
    """Load the train split."""
    df = pd.read_csv(file_path)
    return _normalize_labels(df)


def load_dev(dev_path: Path, dev_gold_path: Path) -> pd.DataFrame:
    """Load dev and merge with dev_gold on ID."""
    dev = pd.read_csv(dev_path)
    gold = pd.read_csv(dev_gold_path)

    merged = dev.merge(
        gold[["ID", "DataID", "Language", "label"]],
        on="ID",
        how="inner",
        suffixes=("", "_gold"),
        validate="one_to_one",
    )

    if "Language_gold" in merged.columns:
        merged["Language"] = merged["Language_gold"].fillna(merged["Language"])
        merged = merged.drop(columns=["Language_gold"])

    return _normalize_labels(merged)


def _make_ratio_table(series: pd.Series, value_name: str) -> pd.DataFrame:
    """Create count and ratio table for a categorical column."""
    counts = series.value_counts(dropna=False)
    ratios = series.value_counts(normalize=True, dropna=False)

    df = pd.DataFrame({
        value_name: counts.index.astype(str),
        "count": counts.values,
        "ratio": ratios.values,
    })
    df["ratio"] = df["ratio"].round(4)
    return df


def _mwe_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute requested MWE-level summary statistics."""
    tmp = df.copy()
    tmp["MWE"] = tmp["MWE"].astype(str).str.strip()

    mwe_freq = tmp["MWE"].value_counts().sort_values(ascending=False)
    label_nunique = tmp.groupby("MWE")["label"].nunique()

    both_class_mwes = sorted(label_nunique[label_nunique == 2].index.tolist())
    both_class_freq = mwe_freq[mwe_freq.index.isin(both_class_mwes)]

    min_freq = int(mwe_freq.min()) if not mwe_freq.empty else None
    max_freq = int(mwe_freq.max()) if not mwe_freq.empty else None

    min_freq_mwes = sorted(mwe_freq[mwe_freq == min_freq].index.tolist()) if min_freq is not None else []
    max_freq_mwes = sorted(mwe_freq[mwe_freq == max_freq].index.tolist()) if max_freq is not None else []

    both_min_freq = int(both_class_freq.min()) if not both_class_freq.empty else None
    both_max_freq = int(both_class_freq.max()) if not both_class_freq.empty else None

    both_min_freq_mwes = (
        sorted(both_class_freq[both_class_freq == both_min_freq].index.tolist())
        if both_min_freq is not None else []
    )
    both_max_freq_mwes = (
        sorted(both_class_freq[both_class_freq == both_max_freq].index.tolist())
        if both_max_freq is not None else []
    )

    return {
        "n_rows": int(len(tmp)),
        "n_unique_mwes": int(tmp["MWE"].nunique()),
        "mwes_in_both_classes_count": int(len(both_class_mwes)),
        "mwes_in_both_classes_ratio": round(
            len(both_class_mwes) / tmp["MWE"].nunique(), 4
        ) if tmp["MWE"].nunique() > 0 else 0.0,
        "mwe_frequency_min": min_freq,
        "mwe_frequency_max": max_freq,
        "both_class_mwe_frequency_min": both_min_freq,
        "both_class_mwe_frequency_max": both_max_freq,
        "min_freq_mwes": min_freq_mwes,
        "max_freq_mwes": max_freq_mwes,
        "both_min_freq_mwes": both_min_freq_mwes,
        "both_max_freq_mwes": both_max_freq_mwes,
    }


def _simple_markdown_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a markdown table without extra dependencies."""
    if df.empty:
        return "| empty |\n|---|\n| no rows |"

    headers = [str(col) for col in df.columns]
    rows = df.fillna("").astype(str).values.tolist()

    table = []
    table.append("| " + " | ".join(headers) + " |")
    table.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in rows:
        safe_row = [cell.replace("\n", "<br>") for cell in row]
        table.append("| " + " | ".join(safe_row) + " |")

    return "\n".join(table)


def _mwe_list_table(mwes: List[str], frequency: int | None) -> pd.DataFrame:
    """Create a table listing MWEs with the same frequency."""
    if frequency is None or not mwes:
        return pd.DataFrame(columns=["MWE", "frequency"])

    return pd.DataFrame({
        "MWE": mwes,
        "frequency": [frequency] * len(mwes),
    })


def _dataset_section(df: pd.DataFrame, name: str) -> str:
    """Build markdown section for one dataset split."""
    class_table = _make_ratio_table(df["label"], "label")
    class_table["class_name"] = class_table["label"].map({
        "0": "idiom",
        "1": "literal",
    }).fillna("unknown")
    class_table = class_table[["label", "class_name", "count", "ratio"]]

    language_table = _make_ratio_table(df["Language"], "Language")

    mwe_stats = _mwe_statistics(df)
    mwe_summary_table = pd.DataFrame([{
        "n_rows": mwe_stats["n_rows"],
        "n_unique_mwes": mwe_stats["n_unique_mwes"],
        "mwes_in_both_classes_count": mwe_stats["mwes_in_both_classes_count"],
        "mwes_in_both_classes_ratio": mwe_stats["mwes_in_both_classes_ratio"],
        "mwe_frequency_min": mwe_stats["mwe_frequency_min"],
        "mwe_frequency_max": mwe_stats["mwe_frequency_max"],
        "both_class_mwe_frequency_min": mwe_stats["both_class_mwe_frequency_min"],
        "both_class_mwe_frequency_max": mwe_stats["both_class_mwe_frequency_max"],
    }])

    min_mwe_table = _mwe_list_table(
        mwe_stats["min_freq_mwes"],
        mwe_stats["mwe_frequency_min"],
    )
    max_mwe_table = _mwe_list_table(
        mwe_stats["max_freq_mwes"],
        mwe_stats["mwe_frequency_max"],
    )
    both_min_mwe_table = _mwe_list_table(
        mwe_stats["both_min_freq_mwes"],
        mwe_stats["both_class_mwe_frequency_min"],
    )
    both_max_mwe_table = _mwe_list_table(
        mwe_stats["both_max_freq_mwes"],
        mwe_stats["both_class_mwe_frequency_max"],
    )

    parts = [
        f"## {name}",
        "",
        "### Class distribution",
        _simple_markdown_table(class_table),
        "",
        "### Language distribution",
        _simple_markdown_table(language_table),
        "",
        "### MWE summary",
        _simple_markdown_table(mwe_summary_table),
        "",
        "### MWEs with minimum frequency",
        _simple_markdown_table(min_mwe_table),
        "",
        "### MWEs with maximum frequency",
        _simple_markdown_table(max_mwe_table),
        "",
        "### MWEs present in both classes with minimum frequency",
        _simple_markdown_table(both_min_mwe_table),
        "",
        "### MWEs present in both classes with maximum frequency",
        _simple_markdown_table(both_max_mwe_table),
        "",
    ]
    return "\n".join(parts)


def _dataset_summary_dict(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    """Build the same EDA summary as a JSON-serializable dictionary."""
    class_table = _make_ratio_table(df["label"], "label")
    class_table["class_name"] = class_table["label"].map({
        "0": "idiom",
        "1": "literal",
    }).fillna("unknown")
    class_table = class_table[["label", "class_name", "count", "ratio"]]

    language_table = _make_ratio_table(df["Language"], "Language")
    mwe_stats = _mwe_statistics(df)

    return {
        "dataset": name,
        "class_distribution": class_table.to_dict(orient="records"),
        "language_distribution": language_table.to_dict(orient="records"),
        "mwe_summary": {
            "n_rows": mwe_stats["n_rows"],
            "n_unique_mwes": mwe_stats["n_unique_mwes"],
            "mwes_in_both_classes_count": mwe_stats["mwes_in_both_classes_count"],
            "mwes_in_both_classes_ratio": mwe_stats["mwes_in_both_classes_ratio"],
            "mwe_frequency_min": mwe_stats["mwe_frequency_min"],
            "mwe_frequency_max": mwe_stats["mwe_frequency_max"],
            "both_class_mwe_frequency_min": mwe_stats["both_class_mwe_frequency_min"],
            "both_class_mwe_frequency_max": mwe_stats["both_class_mwe_frequency_max"],
            "min_freq_mwes": mwe_stats["min_freq_mwes"],
            "max_freq_mwes": mwe_stats["max_freq_mwes"],
            "both_min_freq_mwes": mwe_stats["both_min_freq_mwes"],
            "both_max_freq_mwes": mwe_stats["both_max_freq_mwes"],
        },
    }


def build_eda_json(
    train_path: Path = TRAIN_PATH,
    dev_path: Path = DEV_PATH,
    dev_gold_path: Path = DEV_GOLD_PATH,
) -> Dict[str, Any]:
    """Build the full EDA summary as a dictionary."""
    train_df = load_data(train_path)
    dev_df = load_data(dev_path)
    #dev_df = load_dev(dev_path, dev_gold_path)

    return {
        "train": _dataset_summary_dict(train_df, "train"),
        "dev": _dataset_summary_dict(dev_df, "dev"),
    }

def build_eda_markdown(
    train_path: Path = TRAIN_PATH,
    dev_path: Path = DEV_PATH,
    dev_gold_path: Path = DEV_GOLD_PATH,
) -> str:
    """Build the full markdown report for train and dev."""
    train_df = load_data(train_path)
    dev_df = load_data(dev_path)
    #dev_df = load_dev(dev_path, dev_gold_path)

    parts = [
        "# Dataset EDA",
        "",
        f"- Train file: `{train_path}`",
        f"- Dev file: `{dev_path}`",
        f"- Dev gold file: `{dev_gold_path}`",
        "",
        _dataset_section(train_df, "train"),
        _dataset_section(dev_df, "dev"),
    ]

    return "\n".join(parts)


def write_eda_markdown(output_path: Path = OUTPUT_PATH) -> Path:
    """Generate and save the markdown EDA report."""
    markdown = build_eda_markdown()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    output_file = write_eda_markdown()
    print(f"Saved markdown report to: {output_file}")

    eda_json = build_eda_json()
    print(json.dumps(eda_json, indent=2, ensure_ascii=False))