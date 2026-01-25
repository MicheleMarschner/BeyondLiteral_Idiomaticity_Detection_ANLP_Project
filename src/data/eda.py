## Counts + label balance per split/lang
## Split constraint verification (MWE overlaps for zero/one-shot)
## MWE frequency distribution (for one-shot memorization curve later)
## Ambiguity rate (token vs type - real PIE)
## Context availability (are prev/next often empty?)
## Span coverage (can we find the MWE in target?)
## Duplicate/leakage checks (prevent inflated results)
## Run complte EDA module


from pathlib import Path
import pandas as pd

# ---------- helpers ----------
def norm_mwe(s: str) -> str:
    return " ".join(str(s).lower().split())

def add_norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MWE_norm"] = df["MWE"].apply(norm_mwe)
    df["Label"] = df["Label"].astype(int)
    return df

# ---------- build slices from TRAIN ----------
def build_type_slices_from_train(
    train_path,
    *,
    freq_bins=(0, 1, 5, 10, 20, float("inf")),
    freq_labels=("1", "2-5", "6-10","11-20", "21+"),
) -> dict[str, pd.DataFrame]:
    """
    Returns {lang: type_table_df} built from train only.
    Each type_table has:
      MWE_norm, type_freq, freq_bin, n_unique_labels, is_ambiguous, idiom_rate
    """
    train_path = Path(train_path)
    train_df = add_norm_cols(pd.read_csv(train_path))

    out = {}
    for lang, df_lang in train_df.groupby("Language"):
        # type frequency
        freq = (
            df_lang.groupby("MWE_norm")
                  .size()
                  .rename("type_freq")
                  .reset_index()
        )
        freq["freq_bin"] = pd.cut(
            freq["type_freq"],
            bins=list(freq_bins),
            labels=list(freq_labels),
            include_lowest=True,
        )

        # ambiguity + idiom rate
        labs = (
            df_lang.groupby("MWE_norm")["Label"]
                  .agg(
                      n_unique_labels="nunique",
                      idiom_rate="mean",          # since Label is 0/1
                  )
                  .reset_index()
        )
        labs["is_ambiguous"] = labs["n_unique_labels"].ge(2)

        # combine into one per-language type table
        type_table = freq.merge(labs, on="MWE_norm", how="left")
        out[str(lang)] = type_table.sort_values("type_freq", ascending=False).reset_index(drop=True)

    return out

# ---------- attach slices to ANY split (dev/test/train) ----------
def attach_type_slices(df: pd.DataFrame, type_table: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns from type_table to df based on MWE_norm.
    Unseen types in train will have NaNs -> marked as unseen.
    """
    df = add_norm_cols(df)

    merged = df.merge(
        type_table[["MWE_norm", "type_freq", "freq_bin", "is_ambiguous", "idiom_rate", "entropy"]],
        on="MWE_norm",
        how="left",
    )

    merged["seen_in_train"] = merged["type_freq"].notna()
    merged["freq_bin"] = merged["freq_bin"].astype("object").where(merged["seen_in_train"], "unseen")
    merged["amb_slice"] = merged["is_ambiguous"].astype("object").where(merged["seen_in_train"], "unseen")
    # amb_slice will be True/False/unseen
    return merged



## Test if all MWE's have both labels
RAW_DATA_PATH = Path("./data/raw/Data")
TRAIN_ONE_SHOT_PATH = RAW_DATA_PATH / "train_one_shot.csv"
TRAIN_ZERO_SHOT_PATH = RAW_DATA_PATH / "train_zero_shot.csv"

paths = {
    "zero_shot": TRAIN_ZERO_SHOT_PATH,
    "one_shot":  TRAIN_ONE_SHOT_PATH,
}

slices_by_setting = {}
for setting, train_path in paths.items():
    type_tables_by_lang = build_type_slices_from_train(train_path)
    slices_by_setting[setting] = type_tables_by_lang



def summarize_type_table(type_table: pd.DataFrame) -> pd.DataFrame:
    # base summary
    summary = {
        "n_types": len(type_table),
        "n_ambiguous": int(type_table["is_ambiguous"].sum()),
        "pct_ambiguous": round(100 * type_table["is_ambiguous"].mean(), 2),
    }

    # bin counts (safe even if freq_bin is categorical)
    bin_counts = type_table["freq_bin"].astype(str).value_counts(dropna=False).to_dict()

    # put bins into columns like bin_1, bin_2-5, ...
    for k, v in bin_counts.items():
        summary[f"bin_{k}"] = int(v)

    return pd.DataFrame([summary])


for setting, by_lang in slices_by_setting.items():
    for lang, tt in by_lang.items():
        print(f"\n=== {setting} | {lang} ===")
        print(summarize_type_table(tt).to_string(index=False))

        tt = tt.copy()

        # word length per MWE type
        tt["mwe_len"] = tt["MWE_norm"].str.split().str.len()

        # distribution (how many MWE types have length 1, 2, 3, ...)
        print("\nMWE length distribution (type-level):")
        print(tt["mwe_len"].value_counts().sort_index().to_string())