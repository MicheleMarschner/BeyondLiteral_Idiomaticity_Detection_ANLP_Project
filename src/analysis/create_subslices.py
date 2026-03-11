import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Sequence

from config import PATHS, Paths
from utils.helper import copy_original_dataset, read_csv_data


def _make_freq_bins(freq: pd.Series) -> pd.Categorical:
    """Sort MWE frequency counts into coarse bins"""

    edges = [1, 2, 5, 10, 20, 30, 50, 10**9]
    labels = ["1", "2-4", "5-9", "10-19", "20-29", "30-49", "50+"]
    
    return pd.cut(freq, bins=edges, labels=labels, right=False)


def add_mwe_freq_bin_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Adds split-based MWE frequency columns: mwe_freq, mwe_freq_bin"""
    res_df = df.copy()
    
    counts = res_df["MWE"].astype(str).value_counts()
    res_df["MWE"] = res_df["MWE"].astype(str)
    res_df["mwe_freq"] = res_df["MWE"].map(counts).fillna(0).astype(int)
    res_df["mwe_freq_bin"] = _make_freq_bins(res_df["mwe_freq"]).astype(str)
    
    return res_df


def add_train_mwe_freq_bin_cols(split_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Adds train-only MWE frequency metadata about MWE token count and frequency bin it belongs to"""
    
    df = split_df.copy()
    
    train_counts = train_df["MWE"].value_counts()
    df["train_mwe_freq"] = df["MWE"].map(train_counts).fillna(0).astype(int)

    df["train_mwe_freq_bin"] = _make_freq_bins(df["mwe_freq"]).astype(str)  
    # add seen/unseen flag for later analysis of one-shot vs. zero-shot
    df["seen_mwe_type"] = df["train_mwe_freq"] > 0
    
    return df


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
                idiom_n=("label", lambda x: (1 - x).sum()), # label==0
                literal_rate=("label", "mean"))             
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
    is_amb = df["minority_label"].notna()
    df["is_ambiguous_mwe"] = is_amb
    
    df["slice_minority_instance"] = ""      # slice_minority_instance

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
    examples. For each (Language, mwe_freq_bin) group, we randomly sample the same number 
    of non-hard examples as in the hard set, so hard and control have 
    comparable language and frequency-bin makeup.
    """
    csv_path = Path(csv_path)
    df = read_csv_data(csv_path)

    required_cols = {"ID", "Language", "mwe_freq_bin"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {sorted(missing)} in {csv_path}. "
                         f"Needed for matched control sampling.")

    hard_ids = set(hard_ids)

    # create slice column
    df.loc[df["ID"].isin(hard_ids), slice_col] = "hard"

    # build matched control from non-hard
    hard = df[df[slice_col] == "hard"]
    rest = df[df[slice_col] != "hard"]

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

    df.to_csv(csv_path, index=False)


def create_subslices(project_paths: Paths = PATHS, split_type: str = "test") -> None:
    """Creates subslice definitions for one_shot and zero_shot (given split) and evaluates runs on these slices"""

    for setting in ["one_shot", "zero_shot"]:
        train_path = project_paths.data_preprocessed / f"{setting}_splits/{setting}_train.csv"
        split_path = project_paths.data_preprocessed / f"{setting}_splits/{setting}_{split_type}.csv"
        analysis_data_path = project_paths.data_analysis / f"{setting}_{split_type}_analysis.csv"

        if analysis_data_path.exists():
            continue

        train_df = read_csv_data(train_path)
        split_df = read_csv_data(split_path)

        # copy original split file to analysis folder
        copy_original_dataset(split_path, analysis_data_path)

        # split-based frequency bins (for hard/control matching)
        split_df = add_mwe_freq_bin_cols(split_df)

        #train-based frequency bins for one-shot vs. zero-shot analysis
        split_df = add_train_mwe_freq_bin_cols(split_df, train_df)

        # ambiguity slices
        df_with_slices, amb_ids = identify_potentially_ambiguous_mwe(split_df, min_total=5)
        
        # write updated columns back to the analysis CSV
        df_with_slices.to_csv(analysis_data_path, index=False)

        # hard/control based on ambiguous ids
        add_ambiguous_slices(csv_path=analysis_data_path, hard_ids=amb_ids["ambiguous_mwe_ids"])

        df_saved = read_csv_data(analysis_data_path)
        df_saved = df_saved.drop(columns=["mwe_freq_bin", "mwe_freq"], errors="ignore")
        df_saved.to_csv(analysis_data_path, index=False)