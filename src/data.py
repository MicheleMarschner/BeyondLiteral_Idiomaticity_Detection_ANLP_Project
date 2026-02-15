import pandas as pd
from typing import Tuple, Dict, Any
from pathlib import Path

from utils.helper import read_csv_data
from config import MIN_TRAIN, MIN_VAL, MIN_TEST, MIN_PER_CLASS_TRAIN, MIN_PER_CLASS_VAL, MIN_PER_CLASS_TEST


# ! TODO: finish function to build input variants
def apply_input_variant(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Create the model input text for this config and store it in the `Input` column"""
    df = df.copy()
    df['Input'] = df['Target'].astype(str)
    return df


def load_data_splits(config: Dict[str, Any], data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test data for the chosen setting and build the input variant"""

    setting_folder = data_dir / f"{config['setting']}_splits"
    
    train_data_path = setting_folder / f"{config['setting']}_train.csv"
    val_data_path = setting_folder / f"{config['setting']}_dev.csv"
    test_data_path = setting_folder / f"{config['setting']}_test.csv"

    train_data = read_csv_data(train_data_path)
    val_data = read_csv_data(val_data_path)
    test_data = read_csv_data(test_data_path)

    train_data = train_data[train_data['Language'] == config['language']].copy()
    val_data = val_data[val_data['Language'] == config['language']].copy()
    test_data = test_data[test_data['Language'] == config['language']].copy()

    return train_data, val_data, test_data


def build_inputs_for_splits(
    train_data: pd.DataFrame, 
    val_data: pd.DataFrame, 
    test_data: pd.DataFrame, 
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the configured input variant to train/val/test splits and return the updated DataFrames"""

    train_data = apply_input_variant(train_data, config)
    val_data = apply_input_variant(val_data, config)
    test_data = apply_input_variant(test_data, config)

    return train_data, val_data, test_data


def _label_counts(df: pd.DataFrame, label_col: str = "Label") -> dict[int, int]:
    """Return label frequencies as dict"""

    vc = df[label_col].value_counts().to_dict()
    return {int(k): int(v) for k, v in vc.items()}


def compute_and_check_split_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    language: str,
    label_col: str = "Label",
) -> Tuple[dict[str, dict], bool, list[str]]:
    """Summarize split sizes and per-split label counts and checks if sample size is enough to run the experiment"""

    reasons = []

    # size checks
    split_sizes = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
    min_sizes = {"train": MIN_TRAIN, "val": MIN_VAL, "test": MIN_TEST}

    for split, n in split_sizes.items():
        min_n = min_sizes[split]
        if n < min_n:
            reasons.append(f"{split} too small: {n} < {min_n}")

    # label / per-class checks
    splits = {"train": train_df, "val": val_df, "test": test_df}
    min_per_class = {"train": MIN_PER_CLASS_TRAIN, "val": MIN_PER_CLASS_VAL, "test": MIN_PER_CLASS_TEST}

    label_counts_by_split = {}
    for split, df in splits.items():
        label_counts_by_split[split] = _label_counts(df, label_col)
    
    if language == 'GL': 
        return split_stats, (len(reasons) != 0), reasons

    for split, counts in label_counts_by_split.items():
        if len(counts) < 2:
            reasons.append(f"{split} missing a class: counts={counts}")
            continue

        min_label_count = min_per_class[split]
        for label, count in sorted(counts.items()):
            if count < min_label_count:
                reasons.append(f"{split} class {label} too small: {count} < {min_label_count}")

    split_stats = {"n": split_sizes, "label_counts": label_counts_by_split}

    return split_stats, (len(reasons) != 0), reasons