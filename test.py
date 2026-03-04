import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from src.config import PATHS

def check_zero_one_shot_overlap(
    df: pd.DataFrame,
    *,
    setting_col: str = "setting",
    zero_label: str = "zero_shot",
    one_label: str = "one_shot",
    mwe_col: str = "MWE",
    id_col: str = "ID",
    language_col: str = "language",
    normalize_mwe: bool = True,
) -> Dict[str, Any]:
    """
    Checks whether zero-shot and one-shot runs are comparable:
      - overlap in example IDs
      - overlap in MWE types
      - overlap per language
      - quick sanity checks (labels consistent on overlapping IDs)

    Returns a dict of summary stats and overlap sets (as lists for easy printing).
    """
    required = {setting_col, mwe_col, id_col}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    z = df[df[setting_col] == zero_label].copy()
    o = df[df[setting_col] == one_label].copy()

    if z.empty or o.empty:
        return {
            "ok": False,
            "reason": f"Empty subset: zero={len(z)} one={len(o)}. Check setting labels.",
            "zero_rows": int(len(z)),
            "one_rows": int(len(o)),
        }

    # Normalize MWEs to avoid casing/whitespace mismatches
    if normalize_mwe:
        z["_mwe"] = z[mwe_col].astype(str).str.strip().str.lower()
        o["_mwe"] = o[mwe_col].astype(str).str.strip().str.lower()
    else:
        z["_mwe"] = z[mwe_col]
        o["_mwe"] = o[mwe_col]

    z_ids = set(z[id_col].dropna().astype(str).unique())
    o_ids = set(o[id_col].dropna().astype(str).unique())
    id_inter = z_ids & o_ids

    z_mwes = set(z["_mwe"].dropna().unique())
    o_mwes = set(o["_mwe"].dropna().unique())
    mwe_inter = z_mwes & o_mwes

    out: Dict[str, Any] = {
        "ok": True,
        "zero_rows": int(len(z)),
        "one_rows": int(len(o)),
        "zero_unique_ids": int(len(z_ids)),
        "one_unique_ids": int(len(o_ids)),
        "id_overlap_count": int(len(id_inter)),
        "id_overlap_rate_zero": float(len(id_inter) / max(1, len(z_ids))),
        "id_overlap_rate_one": float(len(id_inter) / max(1, len(o_ids))),
        "zero_unique_mwe_types": int(len(z_mwes)),
        "one_unique_mwe_types": int(len(o_mwes)),
        "mwe_overlap_count": int(len(mwe_inter)),
        "mwe_overlap_rate_zero": float(len(mwe_inter) / max(1, len(z_mwes))),
        "mwe_overlap_rate_one": float(len(mwe_inter) / max(1, len(o_mwes))),
    }

    # Per-language overlap (if available)
    if language_col in df.columns:
        per_lang = []
        for lang in sorted(set(df[language_col].dropna().astype(str).unique())):
            zl = z[z[language_col].astype(str) == lang]
            ol = o[o[language_col].astype(str) == lang]
            if zl.empty or ol.empty:
                per_lang.append({
                    "language": lang,
                    "zero_rows": int(len(zl)),
                    "one_rows": int(len(ol)),
                    "id_overlap": 0,
                    "mwe_overlap": 0,
                })
                continue

            zl_ids = set(zl[id_col].dropna().astype(str).unique())
            ol_ids = set(ol[id_col].dropna().astype(str).unique())
            zl_m = set((zl["_mwe"]).dropna().unique())
            ol_m = set((ol["_mwe"]).dropna().unique())

            per_lang.append({
                "language": lang,
                "zero_rows": int(len(zl)),
                "one_rows": int(len(ol)),
                "zero_ids": int(len(zl_ids)),
                "one_ids": int(len(ol_ids)),
                "id_overlap": int(len(zl_ids & ol_ids)),
                "zero_mwe_types": int(len(zl_m)),
                "one_mwe_types": int(len(ol_m)),
                "mwe_overlap": int(len(zl_m & ol_m)),
            })
        out["per_language"] = per_lang

    # Label consistency on overlapping IDs (optional but useful)
    if "label" in df.columns and id_inter:
        z_lab = z[z[id_col].astype(str).isin(id_inter)][[id_col, "label"]].copy()
        o_lab = o[o[id_col].astype(str).isin(id_inter)][[id_col, "label"]].copy()
        z_lab[id_col] = z_lab[id_col].astype(str)
        o_lab[id_col] = o_lab[id_col].astype(str)

        merged = z_lab.merge(o_lab, on=id_col, how="inner", suffixes=("_zero", "_one"))
        # If there are duplicates per id, be conservative: check any mismatch
        mism = merged[merged["label_zero"] != merged["label_one"]]
        out["label_mismatch_on_overlapping_ids"] = int(mism[id_col].nunique())
        out["label_overlap_checked_ids"] = int(merged[id_col].nunique())

    # Helpful interpretation flag
    out["comparable_by_id"] = out["id_overlap_count"] > 0
    out["comparable_by_mwe_type"] = out["mwe_overlap_count"] > 0

    return out


def print_overlap_report(overlap: Dict[str, Any]) -> None:
    if not overlap.get("ok", False):
        print("[Overlap check] NOT OK:", overlap.get("reason", ""))
        return

    print("\n[Overlap check] Zero-shot vs One-shot")
    print(f"- Rows: zero={overlap['zero_rows']}  one={overlap['one_rows']}")
    print(f"- Unique IDs: zero={overlap['zero_unique_ids']}  one={overlap['one_unique_ids']}")
    print(f"- ID overlap: {overlap['id_overlap_count']} "
          f"(rate vs zero={overlap['id_overlap_rate_zero']:.3f}, vs one={overlap['id_overlap_rate_one']:.3f})")
    print(f"- Unique MWE types: zero={overlap['zero_unique_mwe_types']}  one={overlap['one_unique_mwe_types']}")
    print(f"- MWE type overlap: {overlap['mwe_overlap_count']} "
          f"(rate vs zero={overlap['mwe_overlap_rate_zero']:.3f}, vs one={overlap['mwe_overlap_rate_one']:.3f})")

    if "label_mismatch_on_overlapping_ids" in overlap:
        print(f"- Label mismatches on overlapping IDs: {overlap['label_mismatch_on_overlapping_ids']} "
              f"(checked {overlap['label_overlap_checked_ids']} overlapping IDs)")

    if "per_language" in overlap:
        print("\nPer-language:")
        for r in overlap["per_language"]:
            print(f"  {r['language']}: id_overlap={r.get('id_overlap',0)}, mwe_overlap={r.get('mwe_overlap',0)} "
                  f"(rows zero={r['zero_rows']}, one={r['one_rows']})")

    print("\nInterpretation:")
    if overlap["comparable_by_id"]:
        print("- ✅ Same examples (IDs) appear in both settings → paired-by-example comparisons are valid.")
    else:
        print("- ⚠️ No shared IDs → you cannot do paired-by-example comparisons across settings.")

    if overlap["comparable_by_mwe_type"]:
        print("- ✅ Some MWE types overlap → type-controlled comparisons are possible (depending on your design).")
    else:
        print("- ⚠️ No shared MWE types → type-controlled overlap analyses won't work as written.")


# ---- Usage in your notebook ----
# --- instead of starting from one combined df, read both csvs and tag them ---
zero_path = PATHS.data_preprocessed / "zero_shot_splits/zero_shot_test.csv"   # <- change
one_path  = PATHS.data_preprocessed / "one_shot_splits/one_shot_test.csv"    # <- change

df_zero = pd.read_csv(zero_path)
df_one  = pd.read_csv(one_path)

df_zero["setting"] = "zero_shot"
df_one["setting"]  = "one_shot"

# combine, then everything else can stay the same
df = pd.concat([df_zero, df_one], ignore_index=True)

overlap = check_zero_one_shot_overlap(df, mwe_col="MWE")  # adjust mwe_col if needed
print_overlap_report(overlap)