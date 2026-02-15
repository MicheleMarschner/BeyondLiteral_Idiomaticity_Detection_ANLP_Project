import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, Union, Sequence

from config import PATHS
from utils.helper import get_ids_by_pair

def generate_general_stats(df: pd.DataFrame) -> pd.DataFrame:

    overview = pd.DataFrame([{
        "n_samples": len(df),
        "unique_mwes": df["MWE"].nunique(),
        "idioms_(%)": float((df['Labels'] == 0).mean()),
    }])

    return overview


def generate_stats_by_lang(df: pd.DataFrame) -> pd.DataFrame:

    lang_label = (
    df.groupby("Language")["Labels"]
        .agg(n="count", literal_rate="mean", literal_n="sum")
        .assign(
            idiom_n=lambda x: x["n"] - x["literal_n"],
            idiom_rate=lambda x: 1 - x["literal_rate"],
        )
        .drop(columns=["literal_rate"])   # optional, keep if you want both
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
    
    edges = [1, 2, 5, 10, 20, 50, 10**9]
    labels = ["1", "2-4", "5-9", "10-19", "20-49", "50+"]
    return pd.cut(freq, bins=edges, labels=labels, right=False)


def add_mwe_freq_bin_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mwe_counts = df["MWE"].value_counts()
    df["mwe_freq"] = df["MWE"].map(mwe_counts).astype(int)
    df["mwe_freq_bin"] = make_freq_bins(df["mwe_freq"]).astype(str)  # store as string
    return df


def create_freq_bin_summary(df_binned: pd.DataFrame) -> pd.DataFrame:
    return (
        df_binned.groupby(["mwe_freq_bin", "Labels"])
            .size().unstack(fill_value=0)
            .reindex(columns=[0, 1], fill_value=0)
            .rename(columns={0: "idiom", 1: "literal"})
            .assign(total=lambda x: x["idiom"] + x["literal"])
            .reset_index()
    )


def top_k_mwe_per_lan(df: pd.DataFrame, lang: str):
    
    res = df[df["Language"] == lang]["MWE"].value_counts().head(10).reset_index()
    res.columns = ["MWE", "count"]

    return res


## Ambiguous MWE slice: a) present both as idiom + literal, b) max. 40/60
# label mixture per (language, mwe) type
def identify_potentially_ambiguous_mwe(df: pd.DataFrame) -> pd.DataFrame:

    type_stats = (
        df.groupby(["Language", "MWE"])
            .agg(n=("ID", "count"),
                literal_n=("Labels", "sum"),
                idiom_n=("Labels", lambda x: (1 - x).sum()),
                literal_rate=("Labels", "mean"))
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

    return (
        type_stats.groupby(["Language", "label_mixture"])
            .size()
            .reset_index(name="n_mwe_types")
    )


def ambiguous_mwe_table(
    df: pd.DataFrame,
    min_total: int = 2,
    lo: float = 0.35,
) -> pd.DataFrame:
    """
    Per-language ambiguous MWE types (idiom=0, literal=1), sorted by overall MWE frequency.
    Keeps MWEs that have both labels within each language and whose idiom share is in [lo, hi].
    """
    df = df.copy()
    df["Labels"] = df["Labels"].astype(int)

    # overall occurrence of each MWE in the split (across languages)
    overall_freq = df["MWE"].value_counts().rename("mwe_freq_overall").reset_index()
    overall_freq = overall_freq.rename(columns={"index": "MWE"})

    type_stats = (
        df.groupby(["Language", "MWE"])
          .agg(
              n=("ID", "count"),                    # occurrences in this language
              literal_n=("Labels", "sum"),          # label==1
              idiom_n=("Labels", lambda s: (1 - s).sum()),  # label==0
          )
          .reset_index()
          .assign(
              literal_pct=lambda x: x["literal_n"] / x["n"],
              idiom_pct=lambda x: x["idiom_n"] / x["n"],
          )
    )

    hi = 1-lo

    amb = type_stats[
        (type_stats["idiom_n"] > 0) &
        (type_stats["literal_n"] > 0) &
        (type_stats["n"] >= min_total) &
        (type_stats["idiom_pct"].between(lo, hi))
    ].copy()

    amb = amb.merge(overall_freq, on="MWE", how="left")

    # Sort by overall occurrence first, then per-language n, then closest to 50/50
    amb["mix_dist_from_50"] = (amb["idiom_pct"] - 0.5).abs()
    amb = amb.sort_values(
        ["Language", "mwe_freq_overall", "n", "mix_dist_from_50"],
        ascending=[True, False, False, True],
    ).drop(columns=["mix_dist_from_50"])

    return amb


def run_eda_on_test(df: pd.DataFrame) -> pd.DataFrame:

    overview = generate_general_stats(df)
    lang_label = generate_stats_by_lang(df)
    df_freq_bins = add_mwe_freq_bin_cols(df)
    freq_bins_summary = create_freq_bin_summary(df)
    mix_tbl = identify_potentially_ambiguous_mwe(df)

    print("\n====================")
    print("SPLIT OVERVIEW")
    print("====================")
    print(overview)

    print("\n--------------------")
    print("Stats by language")
    print("--------------------")
    print(lang_label)

    print("\n--------------------")
    print("MWE frequency bins (examples)")
    print("--------------------")
    print(freq_bins_summary.to_string(index=False))

    print("\n--------------------")
    print("Top 10 MWEs per language (by frequency in this split)")
    print("--------------------")
    for lang in sorted(df["Language"].unique()):
        top_k = top_k_mwe_per_lan(df, lang)
        print(f"\n[{lang}]")
        print(top_k.to_string(index=False))

    print("\n--------------------")
    print("MWE type mixture (per language)")
    print("--------------------")
    print(mix_tbl.to_string(index=False))


    # EN (≈466 tokens): tokens: ≥ 40–80, types: ≥ 8–15
    # PT (≈273 tokens): tokens: ≥ 25–60, types: ≥ 5–10
    amb = ambiguous_mwe_table(df, lo=0.3, min_total=3)
    for lang, sub in amb.groupby("Language"):
        print(f"\n[{lang}]")
        print(sub.to_string(index=False))

    hard_ids = get_ids_by_pair(df, amb, col1="Language", col2="MWE")

    return hard_ids


def add_ambiguous_slices(
    csv_path: Union[str, Path],
    hard_ids: Sequence,
    slice_col: str = "slice_ambiguous",
    default_value: str = "",
    random_state: int = 0,
) -> None:
    """
    Adds/overwrites `slice_col` in csv, marking:
      - rows with ID in hard_ids as "hard"
      - a matched random control sample from the remaining rows as "control"
        (matched on Language + mwe_freq_bin; requires those columns)
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    required_cols = {"ID", "Language", "mwe_freq_bin"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {csv_path}. "
                         f"Needed for matched control sampling.")

    hard_ids = set(hard_ids)

    # create slice column
    df['slice_ambiguous'] = default_value
    df.loc[df["ID"].isin(hard_ids), slice_col] = "hard"

    # build matched control from non-hard
    hard = df[df['slice_ambiguous'] == "hard"]
    rest = df[df['slice_ambiguous'] != "hard"]

    hard_counts = hard.groupby(["Language", "mwe_freq_bin"]).size()

    def sample_group(g: pd.DataFrame) -> pd.DataFrame:
        k = int(hard_counts.get(g.name, 0))  # g.name = (Language, mwe_freq_bin)
        if k <= 0:
            return g.iloc[0:0]
        if len(g) <= k:
            return g  # not enough rows; take all
        return g.sample(n=k, random_state=random_state)

    control = rest.groupby(["Language", "mwe_freq_bin"], group_keys=False).apply(sample_group)

    # mark control samples
    df.loc[df["ID"].isin(control["ID"]), slice_col] = "control"

    save_to = Path(csv_path)
    df.to_csv(save_to, index=False)


def add_mwe_freq_bin_slice(
    target_csv: Union[str, Path],
    df: pd.DataFrame,
) -> None:
    """
    Merge columns from df into target_csv by ID and save.

    target_csv must contain id_col.
    df must contain id_col + cols_to_add.
    """
    target_csv = Path(target_csv)
    df_target = pd.read_csv(target_csv)

    if "ID" not in df_target.columns:
        raise ValueError(f"{"ID"} not found in {target_csv}")

    missing = [c for c in ("ID", ("mwe_freq", "mwe_freq_bin")) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    add_df = df[["ID", ("mwe_freq", "mwe_freq_bin")]].drop_duplicates(subset=["ID"])

    df_out = df_target.merge(add_df, on="ID", how="left")

    save_to = Path(target_csv)
    df_out.to_csv(save_to, index=False)


def create_subslices(path: Path, hard_ids: pd.DataFrame, df_freq_bins: pd.DataFrame):
    add_ambiguous_slices(hard_ids, path)
    add_mwe_freq_bin_slice(path)


def run_analysis(path: Path=PATHS.data_preprocessed):

    test_data_path = path / "dev_merged.csv"
    df = pd.read_csv(test_data_path)

    hard_ids, df_freq_bins = run_eda_on_test(df)
    create_subslices(test_data_path, hard_ids, df_freq_bins)