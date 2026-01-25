from pathlib import Path
import pandas as pd
import numpy as np

# ---------- helpers ----------
def norm_mwe(s: str) -> str:
    return " ".join(str(s).lower().split())

def add_norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MWE_norm"] = df["MWE"].apply(norm_mwe)
    df["Label"] = df["Label"].astype(int)
    return df

def load_csv(path: Path) -> pd.DataFrame:
    return add_norm_cols(pd.read_csv(path))

def safe_empty_rate(series: pd.Series) -> float:
    s = series.fillna("").astype(str).str.strip()
    return float((s == "").mean())

def pretty_pct(x: float) -> float:
    return round(100 * float(x), 2)

# ---------- slices from TRAIN ----------
def build_type_slices_from_train(
    train_path,
    *,
    freq_bins=(0, 1, 5, 10, 20, float("inf")),
    freq_labels=("1", "2-5", "6-10", "11-20", "21+"),
) -> dict[str, pd.DataFrame]:
    train_df = load_csv(Path(train_path))
    out = {}
    for lang, df_lang in train_df.groupby("Language"):
        freq = df_lang.groupby("MWE_norm").size().rename("type_freq").reset_index()
        freq["freq_bin"] = pd.cut(
            freq["type_freq"], bins=list(freq_bins), labels=list(freq_labels), include_lowest=True
        )

        labs = (
            df_lang.groupby("MWE_norm")["Label"]
            .agg(n_unique_labels="nunique", idiom_rate="mean")
            .reset_index()
        )
        labs["is_ambiguous"] = labs["n_unique_labels"].ge(2)

        out[str(lang)] = (
            freq.merge(labs, on="MWE_norm", how="left")
            .sort_values("type_freq", ascending=False)
            .reset_index(drop=True)
        )
    return out

def attach_type_slices(df: pd.DataFrame, type_table: pd.DataFrame) -> pd.DataFrame:
    df = add_norm_cols(df)
    merged = df.merge(
        type_table[["MWE_norm", "type_freq", "freq_bin", "is_ambiguous", "idiom_rate"]],
        on="MWE_norm",
        how="left",
    )
    merged["seen_in_train"] = merged["type_freq"].notna()
    merged["freq_bin"] = merged["freq_bin"].astype("object").where(merged["seen_in_train"], "unseen")
    merged["amb_slice"] = merged["is_ambiguous"].astype("object").where(merged["seen_in_train"], "unseen")
    return merged

def summarize_type_table(type_table: pd.DataFrame) -> pd.DataFrame:
    summary = {
        "n_types": len(type_table),
        "n_ambiguous": int(type_table["is_ambiguous"].sum()),
        "pct_ambiguous": round(100 * type_table["is_ambiguous"].mean(), 2),
    }
    bin_counts = type_table["freq_bin"].astype(str).value_counts(dropna=False).to_dict()
    for k, v in bin_counts.items():
        summary[f"bin_{k}"] = int(v)
    return pd.DataFrame([summary])

# ---------- discovery ----------
def discover_split_paths(data_dir: Path, setting: str) -> dict[str, Path]:
    """
    Finds train/dev/test for a setting like 'zero_shot' by trying common filename patterns.
    """
    data_dir = Path(data_dir)

    def pick(candidates):
        for p in candidates:
            if p.exists():
                return p
        return None

    train = pick([data_dir / f"train_{setting}.csv"])
    dev = pick([
        data_dir / f"dev_{setting}.csv",
        data_dir / f"valid_{setting}.csv",
        data_dir / f"val_{setting}.csv",
        data_dir / f"validation_{setting}.csv",
    ])
    test = pick([data_dir / f"test_{setting}.csv"])

    if train is None:
        raise FileNotFoundError(f"Could not find train file for setting='{setting}' in {data_dir}")

    paths = {"train": train}
    if dev is not None: paths["dev"] = dev
    if test is not None: paths["test"] = test
    return paths

# ---------- checklist sections ----------
def counts_label_balance(df_all: pd.DataFrame) -> pd.DataFrame:
    out = (
        df_all.groupby(["Split", "Language"])["Label"]
        .agg(n="size", pos_rate="mean")
        .reset_index()
    )
    out["pos_rate_%"] = out["pos_rate"].apply(pretty_pct)
    return out.drop(columns=["pos_rate"])

def mwe_overlap(train_df: pd.DataFrame, other_df: pd.DataFrame, other_name: str) -> pd.DataFrame:
    rows = []
    for lang in sorted(other_df["Language"].unique()):
        train_types = set(train_df[train_df["Language"] == lang]["MWE_norm"].unique())
        other_types = other_df[other_df["Language"] == lang]["MWE_norm"].unique()
        overlap_pct = np.mean([t in train_types for t in other_types]) if len(other_types) else 0.0
        rows.append({
            "Language": lang,
            f"type_overlap_{other_name}_pct": round(100*overlap_pct, 2),
            f"n_{other_name}_types": int(pd.Series(other_types).nunique()),
            "n_train_types": len(train_types),
        })
    return pd.DataFrame(rows)

def context_availability(df_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, lang), g in df_all.groupby(["Split", "Language"]):
        rows.append({
            "Split": split, "Language": lang,
            "prev_empty_%": pretty_pct(safe_empty_rate(g["Previous"])),
            "next_empty_%": pretty_pct(safe_empty_rate(g["Next"])),
        })
    return pd.DataFrame(rows)

def span_coverage(df_all: pd.DataFrame) -> pd.DataFrame:
    g = df_all.copy()
    g["Target_norm"] = g["Target"].fillna("").astype(str).str.lower()
    g["found_in_target"] = g.apply(lambda r: norm_mwe(r["MWE"]) in r["Target_norm"], axis=1)
    out = g.groupby(["Split", "Language"])["found_in_target"].mean().reset_index()
    out["found_in_target_%"] = out["found_in_target"].apply(pretty_pct)
    return out.drop(columns=["found_in_target"])

def duplicate_checks(df_all: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["Language", "MWE_norm", "Previous", "Target", "Next", "Label"]
    dup = df_all.duplicated(subset=key_cols, keep=False)
    out = df_all.assign(is_dup=dup).groupby(["Split", "Language"])["is_dup"].mean().reset_index()
    out["dup_%"] = out["is_dup"].apply(pretty_pct)
    return out.drop(columns=["is_dup"])

def minority_sense_rate(devtest_df: pd.DataFrame, type_table: pd.DataFrame) -> pd.DataFrame:
    """
    Token-level difficulty slice: among *seen* MWEs, how often is the gold label the minority sense,
    relative to train idiom_rate? (computed per language separately via type_table)
    """
    g = attach_type_slices(devtest_df, type_table)
    g = g[g["seen_in_train"]].copy()
    if len(g) == 0:
        return pd.DataFrame([{"minority_sense_%": np.nan, "note": "no seen MWEs in this split"}])

    g["majority_label"] = (g["idiom_rate"] >= 0.5).astype(int)
    g["is_minority_sense"] = (g["Label"] != g["majority_label"])
    return pd.DataFrame([{
        "minority_sense_%": round(100 * g["is_minority_sense"].mean(), 2),
        "n_seen_instances": len(g),
    }])

# ---------- main runner ----------
def run_complete_eda(data_dir: Path, settings=("zero_shot", "one_shot")) -> None:
    data_dir = Path(data_dir)

    print(f"EDA data dir: {data_dir}")

    for setting in settings:
        paths = discover_split_paths(data_dir, setting)
        print(f"\n==================== SETTING: {setting} ====================")
        print("Files:", {k: str(v) for k, v in paths.items()})

        train = load_csv(paths["train"]).assign(Split="train")
        dev = load_csv(paths["dev"]).assign(Split="dev") if "dev" in paths else None
        test = load_csv(paths["test"]).assign(Split="test") if "test" in paths else None

        frames = [train]
        if dev is not None: frames.append(dev)
        if test is not None: frames.append(test)
        all_df = pd.concat(frames, ignore_index=True)

        print("\n## Counts + label balance per split/lang")
        print(counts_label_balance(all_df).to_string(index=False))

        print("\n## Context availability (prev/next empty?)")
        print(context_availability(all_df).to_string(index=False))

        print("\n## Span coverage (can we find the MWE in target?)")
        print(span_coverage(all_df).to_string(index=False))

        print("\n## Duplicate/leakage checks (exact duplicates within split/lang)")
        print(duplicate_checks(all_df).to_string(index=False))

        # Build type slices from train
        type_tables_by_lang = build_type_slices_from_train(paths["train"])

        # Train type stats + frequency + ambiguity + type-level length
        for lang, tt in type_tables_by_lang.items():
            print(f"\n--- Type stats | {setting} | {lang} ---")
            print(summarize_type_table(tt).to_string(index=False))

            tmp = tt.copy()
            tmp["mwe_len"] = tmp["MWE_norm"].str.split().str.len()
            print("\nMWE length distribution (type-level):")
            print(tmp["mwe_len"].value_counts().sort_index().to_string())

        # Overlap checks within setting (train vs dev/test)
        if dev is not None:
            print("\n## Split constraint verification: train vs dev (type overlap)")
            print(mwe_overlap(train, dev, "dev").to_string(index=False))
        if test is not None:
            print("\n## Split constraint verification: train vs test (type overlap)")
            print(mwe_overlap(train, test, "test").to_string(index=False))

        # Token-level minority-sense rate (dev/test) per language
        if dev is not None or test is not None:
            devtest = pd.concat([x for x in [dev, test] if x is not None], ignore_index=True)
            for lang, tt in type_tables_by_lang.items():
                dt_lang = devtest[devtest["Language"] == lang].copy()
                if len(dt_lang) == 0:
                    continue
                print(f"\n## Token ambiguity hardness (minority-sense rate) | {setting} | {lang}")
                print(minority_sense_rate(dt_lang, tt).to_string(index=False))

# ----------------- usage -----------------
RAW_DATA_PATH = Path("./data/raw/Data")
run_complete_eda(RAW_DATA_PATH, settings=("zero_shot", "one_shot"))



'''
RAW_DATA_PATH = Path("./data/raw/Data")
train_path = RAW_DATA_PATH / "train_zero_shot.csv"  # or train_one_shot.csv

df = pd.read_csv(train_path)

# PT only
pt = df[df["Language"] == "PT"].copy()

# same logic as your span_coverage: substring match on lowercased target and normalized MWE
pt["Target_norm"] = pt["Target"].fillna("").astype(str).str.lower()
pt["MWE_norm"] = pt["MWE"].astype(str).str.lower().str.split().str.join(" ")

pt["found_in_target"] = pt.apply(lambda r: pt.loc[r.name, "MWE_norm"] in pt.loc[r.name, "Target_norm"], axis=1)

not_found = pt[~pt["found_in_target"]].copy()

print(not_found)
'''


import pandas as pd
from pathlib import Path

def norm_mwe(s: str) -> str:
    return " ".join(str(s).lower().split())

def add_norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MWE_norm"] = df["MWE"].apply(norm_mwe)
    df["Label"] = df["Label"].astype(int)
    return df

RAW = Path("./data/raw/Data")
train_zero = add_norm_cols(pd.read_csv(RAW / "eval.csv"))
#train_one  = add_norm_cols(pd.read_csv(RAW / "train_one_shot.csv"))
train_one = None

# ---- MERGED TRAIN ----
train_merged = pd.concat([train_zero, train_one], ignore_index=True)

# 1) ambiguous types in merged train
nuniq = train_merged.groupby(["Language","MWE_norm"])["Label"].nunique()
amb_types_merged = set(nuniq[nuniq >= 2].index)  # set of (Language, MWE_norm)

# 2) type priors in merged train (for minority-sense)
priors = (train_merged.groupby(["Language","MWE_norm"])["Label"]
          .mean().rename("idiom_rate").reset_index())
priors["majority_label"] = (priors["idiom_rate"] >= 0.5).astype(int)

# helper: mark a split with ambiguity + minority info (based on merged train)
def annotate_with_merged_stats(df_split: pd.DataFrame) -> pd.DataFrame:
    df = add_norm_cols(df_split)
    df = df.merge(priors, on=["Language","MWE_norm"], how="left")
    df["seen_in_merged_train"] = df["idiom_rate"].notna()
    df["is_ambiguous_in_merged_train"] = df[["Language","MWE_norm"]].apply(tuple, axis=1).isin(amb_types_merged)

    # minority-sense only meaningful for seen types
    df["is_minority_sense_in_merged_train"] = (
        df["seen_in_merged_train"] & (df["Label"] != df["majority_label"])
    )
    return df

# Example: filter ambiguous/minority tokens in DEV (or TRAIN/TEST)
dev_df = pd.read_csv(RAW / "train_zero_shot.csv")  # or your dev file
dev_annot = annotate_with_merged_stats(dev_df)
# ambiguous_tokens = dev_annot[dev_annot["is_ambiguous_in_merged_train"]]
minority_tokens  = dev_annot[dev_annot["is_minority_sense_in_merged_train"]]

# how many minority-sense token instances
print("minority token instances:", len(minority_tokens))

# unique MWE types among those minority tokens (overall, across langs)
print("unique (Language,MWE) types in minority slice:",
      minority_tokens[["Language","MWE_norm"]].drop_duplicates().shape[0])

# per language breakdown (optional)
print("\nunique minority types per language:")
print(minority_tokens.groupby("Language")["MWE_norm"].nunique())

unique_minority_mwes = (
    minority_tokens[["Language","MWE_norm"]]
    .drop_duplicates()
    .sort_values(["Language","MWE_norm"])
)

# print(unique_minority_mwes.head(50).to_string(index=False))


top = (minority_tokens.groupby(["Language","MWE_norm"])
       .size()
       .rename("minority_token_count")
       .reset_index()
       .sort_values("minority_token_count", ascending=False))

print(top.head(20).to_string(index=False))

