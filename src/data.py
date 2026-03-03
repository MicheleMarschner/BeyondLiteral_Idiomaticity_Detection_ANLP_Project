import pandas as pd
import re
from typing import Tuple, Dict, Any
from pathlib import Path

from utils.helper import read_csv_data
from config import MIN_TRAIN, MIN_DEV, MIN_TEST, MIN_PER_CLASS_TRAIN, MIN_PER_CLASS_DEV, MIN_PER_CLASS_TEST

# import functions required for building the input variant
from input.ner import apply_ner_batch
from input.glosses import get_glosses


def apply_input_variant(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Create the model input text for this config and store it in the `input` column"""

    df = df.copy()
    
    # get the input variant configuration for the chosen setting
    input_variant = config["input_variant"]

    context = input_variant["context"]
    include_mwe = input_variant["include_mwe_segment"]
    transform = input_variant["transform"]
    features = input_variant["features"]

    texts = []
    languages = []

    for _, row in df.iterrows():

        target = str(row["Target"])
        mwe = str(row["MWE"])

        pattern = re.compile(
            r'(?<!\w)' + re.escape(mwe) + r'(?!\w)',
            re.IGNORECASE
        )

        if transform == "mask":
            target = pattern.sub("[MASK]", target)

        elif transform == "highlight":
            target = pattern.sub(f"<MWE> {mwe} </MWE>", target)

        # build the input text based on the context configuration
        if context == "target":
            parts = [target]

        elif context == "previous_target":
            parts = [row["Previous"], target]

        elif context == "target_next":
            parts = [target, row["Next"]]

        elif context == "previous_target_next":
            parts = [row["Previous"], target, row["Next"]]
        else:
            raise ValueError(
                f"Unknown input_variant.context='{context}'. "
                "Expected one of: target, previous_target, target_next, previous_target_next"
            )

        parts = [str(p) for p in parts if pd.notna(p)]
        text = " [SEP] ".join(parts)

        # include mwe segment at the beginning of the text if configured to do so
        if include_mwe:
            text = f"{mwe} [SEP] {text}"
            
        texts.append(text)
        languages.append(row["Language"])

    # if feature = ner, apply the ner function in a batch to all texts and languages
    if "ner" in features:
        texts = apply_ner_batch(texts, languages)

    # if feature = glosses, get the glosses for each word in the MWE and append them to the text
    if "glosses" in features:

        updated_texts = []
        for i in range(len(df)):
            row = df.iloc[i]
            words = str(row["MWE"]).split()
            gloss_parts = [row["MWE"] + "."]

            for word in words:
                gloss_parts.extend(get_glosses(word, row["Language"]))
            gloss_segment = " ".join(gloss_parts)
            updated_texts.append(f"{texts[i]} {gloss_segment} [SEP]")
        
        texts = updated_texts

    # input column is used in train/test df, so return the upated_texts list as the input column in the df, 
    # rest of the columns remain unchanged
    df["input"] = texts 
    return df


def filter_by_language_mode(train_data, dev_data, test_data, config):
    """Filter splits by language mode: per-language, EN→PT cross-lingual, or multilingual (no filter, joint)."""
    
    mode = config["language_mode"]

    if mode == "per_language":
        lang = config["language"]
        train_data = train_data[train_data["Language"] == lang].copy()
        dev_data   = dev_data[dev_data["Language"] == lang].copy()
        test_data  = test_data[test_data["Language"] == lang].copy()

    elif mode == "cross_lingual":
        train_data = train_data[train_data["Language"] == "EN"].copy()
        dev_data   = dev_data[dev_data["Language"] == "EN"].copy()
        test_data  = test_data[test_data["Language"] == "PT"].copy()

    elif mode == "multilingual":
        pass

    else:
        raise ValueError(f"Unknown language_mode: {mode}")

    return train_data, dev_data, test_data


def load_data_splits(config: Dict[str, Any], data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/dev/test data for the chosen setting and build the input variant"""

    setting_folder = data_dir / f"{config['setting']}_splits"
    
    train_data_path = setting_folder / f"{config['setting']}_train.csv"
    dev_data_path = setting_folder / f"{config['setting']}_dev.csv"
    test_data_path = setting_folder / f"{config['setting']}_test.csv"

    train_data = read_csv_data(train_data_path)
    dev_data = read_csv_data(dev_data_path)
    test_data = read_csv_data(test_data_path)

    train_data, dev_data, test_data = filter_by_language_mode(train_data, dev_data, test_data, config)
    
    return train_data, dev_data, test_data



def build_inputs_for_splits(
    train_data: pd.DataFrame, 
    dev_data: pd.DataFrame, 
    test_data: pd.DataFrame, 
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply the configured input variant to train/dev/test splits and return the updated DataFrames"""

    train_data = apply_input_variant(train_data, config)
    dev_data = apply_input_variant(dev_data, config)
    test_data = apply_input_variant(test_data, config)

    return train_data, dev_data, test_data


def _label_counts(df: pd.DataFrame, label_col: str = "label") -> dict[int, int]: # changed label_col default to "label" according to the data files, was "Label" before
    """Return label frequencies as dict"""

    vc = df[label_col].value_counts().to_dict()
    return {int(k): int(v) for k, v in vc.items()}


def compute_and_check_split_stats(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    language: str,
    label_col: str = "label",  # changed label_col default to "label" according to the data files, was "Label" before
) -> Tuple[dict[str, dict], bool, list[str]]:
    """Summarize split sizes and per-split label counts and checks if sample size is enough to run the experiment"""

    reasons = []

    # size checks
    split_sizes = {"train": len(train_df), "dev": len(dev_df), "test": len(test_df)}
    min_sizes = {"train": MIN_TRAIN, "dev": MIN_DEV, "test": MIN_TEST}

    for split, n in split_sizes.items():
        min_n = min_sizes[split]
        if n < min_n:
            reasons.append(f"{split} too small: {n} < {min_n}")

    # label / per-class checks
    splits = {"train": train_df, "dev": dev_df, "test": test_df}
    min_per_class = {"train": MIN_PER_CLASS_TRAIN, "dev": MIN_PER_CLASS_DEV, "test": MIN_PER_CLASS_TEST}

    label_counts_by_split = {}
    for split, df in splits.items():
        label_counts_by_split[split] = _label_counts(df, label_col)
    
    split_stats = {"n": split_sizes, "label_counts": label_counts_by_split}
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

    return split_stats, (len(reasons) != 0), reasons