"""
“The zero-shot and one-shot test sets contain different instances (no ID overlap), so we cannot do paired-by-example deltas.”

“We therefore report a type-controlled comparison, restricting evaluation to the 32 MWE types shared across both test sets.”

“This isolates changes attributable to the one-shot training regime from changes in the expression inventory.”

That’s defensible and linguist-aligned (type inventory control).
"""


import pandas as pd
from pathlib import Path


def load_and_prepare_meta(meta_csv: Path, *, id_col="id", mwe_col="mwe", lang_col="language"):
    meta = pd.read_csv(meta_csv)
    meta[id_col] = meta[id_col].astype(str).str.strip()
    
    keep = [id_col]
    if mwe_col in meta.columns: keep.append(mwe_col)
    if lang_col in meta.columns: keep.append(lang_col)
    return meta[keep].copy()


def merge_mwe_into_overview(
    overview: pd.DataFrame,
    meta_zero: pd.DataFrame,
    meta_one: pd.DataFrame,
    *,
    id_col="id",
    setting_col="setting",
    zero_label="Zero Shot",
    one_label="One Shot",
    mwe_col="mwe",
):
    ov = overview.copy()
    ov[id_col] = ov[id_col].astype(str).str.strip()

    z = ov[ov[setting_col] == zero_label].merge(
        meta_zero[[id_col, mwe_col]],
        on=id_col,
        how="left",
        validate="one_to_one",
    )
    o = ov[ov[setting_col] == one_label].merge(
        meta_one[[id_col, mwe_col]],
        on=id_col,
        how="left",
        validate="one_to_one",
    )

    out = pd.concat([z, o], ignore_index=True)
    return out