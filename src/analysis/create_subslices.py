import pandas as pd
import numpy as np
from pathlib import Path

from typing import Dict, List, Tuple, Union, Sequence


def make_freq_bins(freq: pd.Series) -> pd.Categorical:
    """Sort MWE frequency counts into coarse bins (1, 2–4, 5–9, …, 50+)"""

    edges = [1, 2, 5, 10, 20, 30, 50, 10**9]
    labels = ["1", "2-4", "5-9", "10-19", "20-29", "30-49", "50+"]
    return pd.cut(freq, bins=edges, labels=labels, right=False)


def add_mwe_freq_bin_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Adds per-row MWE frequency metadata about MWE token count and frequency bin it belongs to"""
    df = df.copy()
    mwe_counts = df["MWE"].value_counts()
    df["mwe_freq"] = df["MWE"].map(mwe_counts).astype(int)
    df["mwe_freq_bin"] = make_freq_bins(df["mwe_freq"]).astype(str)  # store as string
    return df


def create_freq_bin_summary(df_binned: pd.DataFrame) -> pd.DataFrame:
    """Summarize idiom vs literal counts (and total) for each bin"""
    return (
        df_binned.groupby(["mwe_freq_bin", "label"])
            .size().unstack(fill_value=0)
            .reindex(columns=[0, 1], fill_value=0)
            .rename(columns={0: "idiom", 1: "literal"})
            .assign(total=lambda x: x["idiom"] + x["literal"])
            .reset_index()
    )


def extract_freq_bin_ids(
    df: pd.DataFrame,
    *,
    id_col: str = "ID",
    bin_col: str = "mwe_freq_bin",
) -> Dict[str, List[str]]:
    """Builds ID lists per frequency bin for slice evaluation"""

    if bin_col not in df.columns:
        raise ValueError(f"Missing '{bin_col}'. Run add_mwe_freq_bin_cols(df) first.")

    out: Dict[str, List[str]] = {}
    for b, sub in df.groupby(bin_col, dropna=False):
        out[f"freqbin={b}"] = sub[id_col].astype(str).tolist()
    return out


def top_k_mwe_per_lan(df: pd.DataFrame, lang: str):
    """Returns the most frequent MWEs for a given language (quick dataset bias check)"""
    res = df[df["Language"] == lang]["MWE"].value_counts().head(10).reset_index()
    res.columns = ["MWE", "count"]

    return res


def identify_potentially_ambiguous_mwe(df: pd.DataFrame, min_total: int=5) -> pd.DataFrame:
    """
    Identifies “ambiguous” MWEs and marks majority/minority instances
    A (Language, MWE) is ambiguous if it appears at least `min_total` times and occurs
    with both labels (idiom + literal). For each ambiguous (Language, MWE), instances
    are labeled as:
      - "minority" if their label is the rarer label for that MWE (within the language)
      - "majority" otherwise
    """
    
    # per (Language, MWE) stats
    type_stats = (
        df.groupby(["Language", "MWE"])
            .agg(n=("ID", "count"),
                literal_n=("label", "sum"),
                idiom_n=("label", lambda x: (1 - x).sum()), # label==1
                literal_rate=("label", "mean"))             # label==0
            .reset_index()
    )
    # ambiguous MWEs: both labels present + min_total
    amb = type_stats[
        (type_stats["n"] >= min_total) &
        (type_stats["literal_n"] > 0) &
        (type_stats["idiom_n"] > 0)
    ].copy()

    # determine minority label per (Language, MWE)
    # if literal_n < idiom_n => minority_label=1 else 0
    amb["minority_label"] = np.where(amb["literal_n"] < amb["idiom_n"], 1, 0)

    df = df.merge(
        amb[["Language", "MWE", "minority_label"]],
        on=["Language", "MWE"],
        how="left",
    )
    # True if the instance belongs to an ambiguous (Language, MWE) type
    df["is_ambiguous_mwe"] = df["minority_label"].notna()

    # slice_minority_instance
    df["slice_minority_instance"] = ""
    is_amb = df["minority_label"].notna()

    df.loc[is_amb, "slice_minority_instance"] = np.where(
        df.loc[is_amb, "label"].values == df.loc[is_amb, "minority_label"].values,
        "minority",
        "majority",
    )

    # filter sample ids of ambiguous slice
    ids_amb = df.loc[df["is_ambiguous_mwe"], "ID"].astype(str).tolist()
    ids_min = df.loc[df["slice_minority_instance"] == "minority", "ID"].astype(str).tolist()
    ids_maj = df.loc[df["slice_minority_instance"] == "majority", "ID"].astype(str).tolist()

    amb_ids = {
        "ambiguous_mwe_ids": ids_amb,
        "minority_instance_ids": ids_min,
        "majority_instance_ids": ids_maj,
    }

    # drop helper
    df = df.drop(columns=["minority_label"])

    return df, amb_ids


def build_slices_and_ids(df_raw: pd.DataFrame, *, min_total: int = 5) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Builds all subslice annotations and the respective ID lists. Adds frequency-bin slices and ambiguity slices
    """
    df = df_raw.copy()
    df["label"] = df["label"].astype(int)

    df = add_mwe_freq_bin_cols(df)
    freq_bins_summary = create_freq_bin_summary(df)
    df, slice_ids = identify_potentially_ambiguous_mwe(df, min_total)

    # freq bins
    slice_ids.update(extract_freq_bin_ids(df))

    return df, slice_ids


def add_ambiguous_slices(
    csv_path: Union[str, Path],
    hard_ids: Sequence,
    slice_col: str = "slice_ambiguous",
    default_value: str = "",
    random_state: int = 0,
) -> None:
    """
    Writes an ambiguity slice ("hard"/"control") into a CSV.

    Marks `hard_ids` as "hard". Then samples a matched "control" set from the remaining
    examples, matched by Language and mwe_freq_bin
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


def add_mwe_freq_bin_slice(target_csv: Union[str, Path], df: pd.DataFrame) -> None:
    """Merges mwe_freq and mwe_freq_bin into an existing CSV by ID"""
    target_csv = Path(target_csv)
    df_target = pd.read_csv(target_csv)

    if "ID" not in df_target.columns:
        raise ValueError(f"'ID' not found in {target_csv}")

    required = {"ID", "mwe_freq", "mwe_freq_bin"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in df: {sorted(missing)}")

    add_df = df[["ID", "mwe_freq", "mwe_freq_bin"]].drop_duplicates(subset=["ID"])
    df_out = df_target.merge(add_df, on="ID", how="left")
    df_out.to_csv(target_csv, index=False)


def add_subslices(path: Path, hard_ids, df_freq_bins: pd.DataFrame) -> None:
    """Wrapper that adds both ambiguity slices (hard/control) and MWE frequency-bin columns"""
    
    add_ambiguous_slices(csv_path=path, hard_ids=hard_ids)
    add_mwe_freq_bin_slice(target_csv=path, df=df_freq_bins)

