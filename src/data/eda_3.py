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

def safe_empty_rate(series: pd.Series) -> float:
    s = series.fillna("").astype(str).str.strip()
    return float((s == "").mean())

def pretty_pct(x: float) -> float:
    return round(100 * float(x), 2)

# ---------- core checks on ONE dataframe ----------
def counts_label_balance(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(["Language"])["Label"]
        .agg(n="size", pos_rate="mean")
        .reset_index()
    )
    out["pos_rate_%"] = out["pos_rate"].apply(pretty_pct)
    return out.drop(columns=["pos_rate"])

def context_availability(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lang, g in df.groupby("Language"):
        rows.append({
            "Language": lang,
            "prev_empty_%": pretty_pct(safe_empty_rate(g["Previous"])),
            "next_empty_%": pretty_pct(safe_empty_rate(g["Next"])),
        })
    return pd.DataFrame(rows)

def span_coverage(df: pd.DataFrame) -> pd.DataFrame:
    g = df.copy()
    g["Target_norm"] = g["Target"].fillna("").astype(str).str.lower()
    g["found_in_target"] = g.apply(lambda r: norm_mwe(r["MWE"]) in r["Target_norm"], axis=1)
    out = g.groupby("Language")["found_in_target"].mean().reset_index()
    out["found_in_target_%"] = out["found_in_target"].apply(pretty_pct)
    return out.drop(columns=["found_in_target"])

def duplicate_checks(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["Language", "MWE_norm", "Previous", "Target", "Next", "Label"]
    dup = df.duplicated(subset=key_cols, keep=False)
    out = df.assign(is_dup=dup).groupby("Language")["is_dup"].mean().reset_index()
    out["dup_%"] = out["is_dup"].apply(pretty_pct)
    return out.drop(columns=["is_dup"])

# ---------- type table from the SAME dataframe ----------
def build_type_table(
    df: pd.DataFrame,
    *,
    freq_bins=(0, 1, 5, 10, 20, float("inf")),
    freq_labels=("1", "2-5", "6-10", "11-20", "21+"),
) -> dict[str, pd.DataFrame]:
    """
    Builds per-language type table: type_freq, freq_bin, ambiguity, idiom_rate, mwe_len.
    """
    out = {}
    for lang, df_lang in df.groupby("Language"):
        # frequency per type
        freq = df_lang.groupby("MWE_norm").size().rename("type_freq").reset_index()
        freq["freq_bin"] = pd.cut(
            freq["type_freq"],
            bins=list(freq_bins),
            labels=list(freq_labels),
            include_lowest=True,
        )

        # label behavior per type
        labs = (
            df_lang.groupby("MWE_norm")["Label"]
            .agg(n_unique_labels="nunique", idiom_rate="mean")
            .reset_index()
        )
        labs["is_ambiguous"] = labs["n_unique_labels"].ge(2)

        tt = (
            freq.merge(labs, on="MWE_norm", how="left")
            .sort_values("type_freq", ascending=False)
            .reset_index(drop=True)
        )
        tt["mwe_len"] = tt["MWE_norm"].str.split().str.len()

        out[str(lang)] = tt
    return out

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

# ---------- within-file minority-sense slice ----------
def annotate_within_file_priors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uses THIS file as the reference:
    - idiom_rate per (Language, MWE_norm)
    - majority_label := idiom_rate >= 0.5
    - is_minority_sense := token label differs from majority_label
    """
    df = add_norm_cols(df)

    priors = (
        df.groupby(["Language", "MWE_norm"])["Label"]
        .mean()
        .rename("idiom_rate")
        .reset_index()
    )
    priors["majority_label"] = (priors["idiom_rate"] >= 0.5).astype(int)

    nuniq = df.groupby(["Language", "MWE_norm"])["Label"].nunique()
    amb_types = set(nuniq[nuniq >= 2].index)  # (Language, MWE_norm)

    out = df.merge(priors, on=["Language", "MWE_norm"], how="left")
    out["is_ambiguous_type"] = out[["Language", "MWE_norm"]].apply(tuple, axis=1).isin(amb_types)
    out["is_minority_sense"] = (out["Label"] != out["majority_label"])
    return out

def minority_sense_summary(df_annot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lang, g in df_annot.groupby("Language"):
        rows.append({
            "Language": lang,
            "minority_sense_%": pretty_pct(g["is_minority_sense"].mean()),
            "n_tokens": int(len(g)),
            "n_minority_tokens": int(g["is_minority_sense"].sum()),
            "n_ambiguous_types": int(g[["MWE_norm","is_ambiguous_type"]].drop_duplicates()["is_ambiguous_type"].sum()),
        })
    return pd.DataFrame(rows)

# ---------- main runner: ONE PATH -> ONE DF ----------
def run_single_file_eda(csv_path: Path) -> dict:
    csv_path = Path(csv_path)
    df = add_norm_cols(pd.read_csv(csv_path))

    print(f"EDA file: {csv_path}")
    print(f"Rows: {len(df)} | Languages: {df['Language'].nunique()}")

    print("\n## Counts + label balance per language")
    print(counts_label_balance(df).to_string(index=False))

    print("\n## Context availability (prev/next empty?)")
    print(context_availability(df).to_string(index=False))

    print("\n## Span coverage (can we find the MWE in target?)")
    print(span_coverage(df).to_string(index=False))

    print("\n## Duplicate/leakage checks (exact duplicates within language)")
    print(duplicate_checks(df).to_string(index=False))

    # Type tables
    type_tables = build_type_table(df)
    for lang, tt in type_tables.items():
        print(f"\n--- Type stats | {lang} ---")
        print(summarize_type_table(tt).to_string(index=False))

        print("\nMWE length distribution (type-level):")
        print(tt["mwe_len"].value_counts().sort_index().to_string())

    # Minority sense (within-file)
    df_annot = annotate_within_file_priors(df)
    print("\n## Token ambiguity hardness (minority-sense rate) | within-file priors")
    print(minority_sense_summary(df_annot).to_string(index=False))

    # Also return useful objects for further filtering
    return {
        "df": df,
        "df_annot": df_annot,
        "type_tables": type_tables,
        "tables": {
            "counts_label_balance": counts_label_balance(df),
            "context_availability": context_availability(df),
            "span_coverage": span_coverage(df),
            "duplicate_checks": duplicate_checks(df),
            "minority_sense_summary": minority_sense_summary(df_annot),
        },
    }

# ----------------- usage -----------------
CSV_PATH = Path("./data/raw/Data/dev.csv")  # <- set ONE file here
out = run_single_file_eda(CSV_PATH)

# Example: get top minority-sense MWEs (within this file)
minority_tokens = out["df_annot"][out["df_annot"]["is_minority_sense"]].copy()
top = (minority_tokens.groupby(["Language", "MWE_norm"])
       .size()
       .rename("minority_token_count")
       .reset_index()
       .sort_values("minority_token_count", ascending=False))
print("\nTop minority-sense MWEs:")
print(top.head(20).to_string(index=False))
