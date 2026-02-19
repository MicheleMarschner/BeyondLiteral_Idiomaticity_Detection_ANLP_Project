"""
preprocess.py

Purpose:
        Create per-experiment preprocessed CSVs from the raw splits and
        from precomputed NER / gloss outputs. This script is an orchestrator
        that applies a single configuration (from `config.py`) to the chosen
        split(s) and writes CSVs to `preprocessed/<config_name>/`.

Assumptions / expected layout:
        - Original splits: `zero_shot_splits/` and `one_shot_splits/` with files
            like `zero_train.csv`, `one_shot_test.csv`, etc.
        - NER outputs (optional): `NER/outputs/<base>_ner.csv` where `Target`
            is already inline-tagged and may include `ner_scores` / `mwe_is_entity`.
        - Gloss outputs (optional): `Glosses/outputs/<base>.csv` containing
            a `gloss` (or `glosses`) column.

Usage examples:
        # Run with a config selected by `config.py` (it will prompt for required
        # config args: --context, --mwe_inclusion, --transformation, --features)
        python preprocess.py --split zero_shot

        # Run with one-shot splits and a config (example config flags shown):
        python preprocess.py --split one_shot --context target --mwe_inclusion True \
                --transformation highlight --features ner

Behavior:
        - The script reads the appropriate source files depending on `features`:
                * `none` -> original CSVs under zero_shot_splits / one_shot_splits
                * `ner`  -> NER/outputs/*_ner.csv
                * `glosses` -> Glosses/outputs/*.csv
                * `ner_glosses` -> load ner file and merge gloss column
        - It applies MWE transformation only to `Target` (plain/mask/highlight),
            keeps only the requested context columns, preserves other original
            columns, and writes outputs under `preprocessed/<config.name>/`.

Notes:
        - NER and Glosses are expected to be produced separately. This script
            will reference those files; if they are missing, run their producers
            first or pregenerate them.
"""

import pandas as pd
import argparse
import re
import sys
from pathlib import Path
from config import get_config


BASE_DIR = Path.cwd()

ORIGINAL_ZERO = BASE_DIR / "zero_shot_splits"
ORIGINAL_ONE = BASE_DIR / "one_shot_splits"
NER_OUTPUT = BASE_DIR / "NER" / "outputs"
GLOSS_OUTPUT = BASE_DIR / "Glosses" / "outputs"
PREPROCESSED = BASE_DIR / "preprocessed"

PREPROCESSED.mkdir(exist_ok=True)

# Context columns to keep based on config
CONTEXT_KEEP = {
    "target": ["Target"],
    "previous_target": ["Previous", "Target"],
    "target_next": ["Target", "Next"],
    "previous_target_next": ["Previous", "Target", "Next"],
}

BASE_COLUMNS_ZERO = ["Language", "MWE", "Setting", "Label"]
BASE_COLUMNS_ONE = ["ID", "Language", "MWE", "Setting", "Label"]


# Load source data based on config
def load_source(split_type, filename, features):
    if features == "none":
        base = ORIGINAL_ZERO if split_type == "zero_shot" else ORIGINAL_ONE
        return pd.read_csv(base / filename)

    if features == "ner":
        return pd.read_csv(NER_OUTPUT / filename.replace(".csv", "_ner.csv"))

    if features == "glosses":
        return pd.read_csv(GLOSS_OUTPUT / filename)

    if features == "ner_glosses":
        df_ner = pd.read_csv(NER_OUTPUT / filename.replace(".csv", "_ner.csv"))
        df_gloss = pd.read_csv(GLOSS_OUTPUT / filename)

        gloss_col = "gloss" if "gloss" in df_gloss.columns else "glosses"
        df_ner["gloss"] = df_gloss[gloss_col].reset_index(drop=True)
        return df_ner

    raise ValueError("Unknown feature type")


# MWE transformation modes: plain, mask, highlight
def transform_text(text, mwe, mode):
    if pd.isna(text):
        return text

    text = str(text)
    mwe = str(mwe)

    if mode == "plain":
        return text

    pattern = re.compile(
        r"(?<!\w)" + re.escape(mwe) + r"(?!\w)",
        re.IGNORECASE,
    )

    if mode == "mask":
        return pattern.sub("[MASK]", text)

    if mode == "highlight":
        return pattern.sub(r"[SEP] \g<0> [SEP]", text)

    return text


def apply_config(df, config, split_type):

    df = df.copy()

    # Apply transformation only to Target
    if "Target" in df.columns:
        df["Target"] = df.apply(
            lambda row: transform_text(
                row["Target"],
                row["MWE"],
                config["transformation"],
            ),
            axis=1,
        )

    context_cols = CONTEXT_KEEP[config["context"]]

    base_cols = BASE_COLUMNS_ZERO if split_type == "zero_shot" else BASE_COLUMNS_ONE
    final_cols = base_cols.copy()

    # Add context columns
    final_cols += context_cols

    # Add feature columns
    if config["features"] in ["ner", "ner_glosses"]:
        final_cols += ["ner_scores", "mwe_is_entity"]

    if config["features"] in ["glosses", "ner_glosses"]:
        final_cols.append("gloss")

    # Include MWE column if requested
    if config["include_mwe"]:
        df["mwe"] = df["MWE"]
        final_cols.append("mwe")

    # Remove duplicates while preserving order
    final_cols = list(dict.fromkeys(final_cols))

    # Keep only existing columns
    final_cols = [c for c in final_cols if c in df.columns]

    return df[final_cols]



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["zero_shot", "one_shot", "both"],
        default="both",
    )

    args, _ = parser.parse_known_args()

    # Get selected config from config.py
    config = get_config()

    splits = ["zero_shot", "one_shot"] if args.split == "both" else [args.split]

    # Create output directory based on config name
    output_dir = PREPROCESSED / config["name"]
    output_dir.mkdir(exist_ok=True)

    print(f"\nProcessing configuration: {config['name']}")

    for split_type in splits:

        files = (
    ["zero_shot_train.csv", "zero_shot_test.csv", "zero_shot_dev.csv"]
    if split_type == "zero_shot"
    else ["one_shot_train.csv", "one_shot_test.csv", "one_shot_dev.csv"]
)
        for filename in files:
            df = load_source(split_type, filename, config["features"])
            processed = apply_config(df, config, split_type)

            processed.to_csv(output_dir / filename, index=False)



    print("\nPreprocessing complete.")
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    main()
