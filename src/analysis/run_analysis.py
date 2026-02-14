import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
#import seaborn as sns

from src.config import Paths, PATHS

## general stats
def generate_general_stats(df: pd.DataFrame):
    class_ratio = df['Labels'].value_counts(normalize=True)  # Label balance
    n_unique_MWEs = df['MWE'].nunique()  # Unique MWEs
    samples_per_language = df['Language'].value_counts()  # Language distribution

    return class_ratio, n_unique_MWEs, samples_per_language


def generate_stats_by_lang(df: pd.DataFrame):

    lang_label = (
    df.groupby("Language")["Labels"]
        .agg(n="count", idiom_rate="mean", idiom_n="sum")
        .assign(literal_n=lambda x: x["n"] - x["idiom_n"])
        .reset_index()
    )

    unique_mwes_per_lang = (
    df.groupby("Language")["MWE"]
        .nunique()
        .reset_index(name="n_unique_mwes")
    )

    lang_label = lang_label.merge(unique_mwes_per_lang, on="Language", how="left")

    return lang_label



def make_freq_bins(freq: pd.Series) -> pd.Categorical:
    # bins that work well for SemEval PIE dev/test (usually ~5-20 per type),
    # but robust if your split differs.
    edges = [1, 2, 5, 10, 20, 50, 10**9]
    labels = ["1", "2-4", "5-9", "10-19", "20-49", "50+"]
    return pd.cut(freq, bins=edges, labels=labels, right=False)


## MWE frequency bins
def create_mwe_freq_bins(df: pd.DataFrame):
    mwe_counts = df["MWE"].value_counts()
    df["mwe_freq"] = df["MWE"].map(mwe_counts)
    df["mwe_freq_bin"] = make_freq_bins(df["mwe_freq"])

    freq_tbl = (
        df.groupby(["mwe_freq_bin", "Labels"])
            .size().unstack(fill_value=0)
            .rename(columns={0: "literal", 1: "idiom"})
            .assign(total=lambda x: x["literal"] + x["idiom"])
            .reset_index()
    )

    return freq_tbl


## Ambiguous MWE slice: a) present both as idiom + literal, b) max. 40/60
# label mixture per (language, mwe) type
def create_ambiguous_mwe(df: pd.DataFrame):

    type_stats = (
        df.groupby(["Language", "MWE"])
            .agg(n=("ID", "count"),
                idiom_n=("Labels", "sum"),
                literal_n=("Labels", lambda x: (1 - x).sum()),
                idiom_rate=("Labels", "mean"))
            .reset_index()
    )
    type_stats["label_mixture"] = np.select(
        [
            (type_stats["idiom_n"] > 0) & (type_stats["literal_n"] > 0),
            (type_stats["idiom_n"] > 0) & (type_stats["literal_n"] == 0),
            (type_stats["idiom_n"] == 0) & (type_stats["literal_n"] > 0),
        ],
        ["both", "idiom_only", "literal_only"],
        default="unknown",
    )

    return type_stats


def run_eda_on_test(path: Path=PATHS):

    test_data_path = path.data_preprocessed / "dev_merged.csv"
    df = pd.read_csv(test_data_path)

    class_ratio, n_unique_MWEs, samples_per_language = generate_general_stats(df)
    lang_label = generate_stats_by_lang(df)
    freq_tbl = create_mwe_freq_bins(df)
    type_stats = create_ambiguous_mwe(df)

    print("\n====================")
    print("SPLIT OVERVIEW")
    print("====================")
    print(f"Number of sampels: {len(df)}")
    print(f"Languages: {samples_per_language}")
    print(f"Unique MWEs (types): {n_unique_MWEs}")
    print(f"Idiom rate (label=1): {class_ratio}")

    print("\n--------------------")
    print("Stats by language")
    print("--------------------")
    print(lang_label)

    print("\n--------------------")
    print("MWE frequency bins (examples)")
    print("--------------------")
    print(freq_tbl.to_string(index=False))


    print("\n--------------------")
    print("Top 10 MWEs per language (by frequency in this split)")
    print("--------------------")
    for lang in sorted(df["Language"].unique()):
        top = df[df["Language"] == lang]["MWE"].value_counts().head(10).reset_index()
        top.columns = ["MWE", "count"]
        print(f"\n[{lang}]")
        print(top.to_string(index=False))


    print("\n--------------------")
    print("MWE type mixture (per language)")
    print("--------------------")
    mix_tbl = (
        type_stats.groupby(["Language", "label_mixture"])
                    .size()
                    .reset_index(name="n_mwe_types")
    )
    print(mix_tbl.to_string(index=False))