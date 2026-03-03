import pandas as pd
import numpy as np
from pathlib import Path

from typing import Dict, List, Tuple, Union, Sequence

from config import PATHS, Paths
from utils.helper import copy_file, get_ids_by_pair, read_json, write_json


def make_freq_bins(freq: pd.Series) -> pd.Categorical:
    
    edges = [1, 2, 5, 10, 20, 30, 50, 10**9]
    labels = ["1", "2-4", "5-9", "10-19", "20-29", "30-49", "50+"]
    return pd.cut(freq, bins=edges, labels=labels, right=False)


def add_mwe_freq_bin_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mwe_counts = df["MWE"].value_counts()
    df["mwe_freq"] = df["MWE"].map(mwe_counts).astype(int)
    df["mwe_freq_bin"] = make_freq_bins(df["mwe_freq"]).astype(str)  # store as string
    return df


def create_freq_bin_summary(df_binned: pd.DataFrame) -> pd.DataFrame:
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
    """
    Return IDs per frequency bin as simple lists (no language split).
    Keys: "freqbin=1", "freqbin=2-4", ..., "freqbin=50+".
    """
    if bin_col not in df.columns:
        raise ValueError(f"Missing '{bin_col}'. Run add_mwe_freq_bin_cols(df) first.")

    out: Dict[str, List[str]] = {}
    for b, sub in df.groupby(bin_col, dropna=False):
        out[f"freqbin={b}"] = sub[id_col].astype(str).tolist()
    return out


def top_k_mwe_per_lan(df: pd.DataFrame, lang: str):
    
    res = df[df["Language"] == lang]["MWE"].value_counts().head(10).reset_index()
    res.columns = ["MWE", "count"]

    return res


## Ambiguous MWE slice: a) present both as idiom + literal, b) max. 40/60
# label mixture per (language, mwe) type
def identify_potentially_ambiguous_mwe(df: pd.DataFrame, min_total: int=5) -> pd.DataFrame:

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
    Adds/overwrites `slice_col` in csv, marking:
      - rows with ID in hard_ids as "hard"
      - a matched random control sample from the remaining rows as "control"
        (matched on Language + mwe_freq_bin; requires those columns)
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
    add_ambiguous_slices(csv_path=path, hard_ids=hard_ids)
    add_mwe_freq_bin_slice(target_csv=path, df=df_freq_bins)


def create_dataset_for_analysis(data_path: Path, analysis_data_path: Path):
    copy_file(data_path, analysis_data_path)


def run_analysis(setting: str, split_type: str, project_paths: Paths = PATHS):
    data_path = project_paths.data_preprocessed / f"{setting}_splits/{setting}_{split_type}.csv"
    analysis_data_path = project_paths.data_analysis / f"{setting}_{split_type}_analysis.csv"
    slice_ids_path = project_paths.data_analysis / f"{setting}_{split_type}_slice_ids.json"

    df = pd.read_csv(data_path)

    if not analysis_data_path.exists():
        create_dataset_for_analysis(data_path, analysis_data_path)

        df_with_slices, slice_ids = build_slices_and_ids(df, min_total=5)

        # write updated columns back to the analysis CSV
        df_with_slices.to_csv(analysis_data_path, index=False)

        add_ambiguous_slices(csv_path=analysis_data_path, hard_ids=slice_ids["ambiguous_mwe_ids"])

        # save IDs json (contains both ambiguity + freqbin slices)
        write_json(slice_ids_path, slice_ids)







def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    """
    y in {0,1}, p = P(y=1 | x)
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def mean_pred_confidence(p: np.ndarray) -> float:
    """
    confidence of predicted class given p(y=1): max(p, 1-p)
    """
    p = np.asarray(p, dtype=float)
    conf = np.maximum(p, 1.0 - p)
    return float(np.mean(conf))


# ----------------------------
# Core: slice evaluation for one run
# ----------------------------
def evaluate_slices_for_run(
    pred_csv: Path,
    slice_ids: Dict[str, List[str]],
) -> Dict[str, Any]:
    """
    pred_csv must contain: id, label, test_pred, test_proba_literal
    Returns:
      slice_name -> {n, macro_f1, ..., proba_stats{log_loss, mean_pred_conf, ...}}
    """
    pred_df = pd.read_csv(pred_csv)

    pred_df["id"] = pred_df["id"].astype(str)
    pred_df["label"] = pred_df["label"].astype(int)
    pred_df["test_pred"] = pred_df["test_pred"].astype(int)
    pred_df["test_proba_literal"] = pred_df["test_proba_literal"].astype(float)

    pred_by_id = pred_df.set_index("id", drop=False)

    out: Dict[str, Any] = {}
    for slice_name, ids in slice_ids.items():
        ids_set = set(map(str, ids))
        sub = pred_by_id.loc[pred_by_id.index.intersection(ids_set)]

        if len(sub) == 0:
            out[slice_name] = {"n": 0}
            continue

        y = sub["label"].to_numpy()
        preds = sub["test_pred"].to_numpy()
        p = sub["test_proba_literal"].to_numpy()

        metrics = compute_metrics(y, preds)

        proba_stats = {
            "log_loss": log_loss_from_p_literal(y, p),
            "mean_pred_conf": mean_pred_confidence_from_p_literal(p),
            "mean_p_literal": float(np.mean(p)),
            "std_p_literal": float(np.std(p)),
        }

        out[slice_name] = {"n": int(len(sub)), **metrics, "proba_stats": proba_stats}

    return out


# ----------------------------
# Flatten + deltas
# ----------------------------
def flatten_slice_metrics(
    run_dir: str,
    slice_metrics: Dict[str, Any],
) -> pd.DataFrame:
    """
    slice_metrics: output of evaluate_slices_for_run
    Returns long DF with one row per slice.
    """
    rows = []
    for slice_name, m in slice_metrics.items():
        proba = m.get("proba_stats", {}) if isinstance(m, dict) else {}
        cm = m.get("confusion_matrix_values", {}) if isinstance(m, dict) else {}

        rows.append({
            "run_dir": run_dir,
            "slice": slice_name,
            "n": m.get("n", 0),

            "accuracy": m.get("accuracy"),
            "macro_precision": m.get("macro_precision"),
            "macro_recall": m.get("macro_recall"),
            "macro_f1": m.get("macro_f1"),

            "tp": cm.get("tp"),
            "tn": cm.get("tn"),
            "fp": cm.get("fp"),
            "fn": cm.get("fn"),

            "log_loss": proba.get("log_loss"),
            "mean_pred_conf": proba.get("mean_pred_conf"),
            "mean_p_literal": proba.get("mean_p_literal"),
            "std_p_literal": proba.get("std_p_literal"),
        })

    return pd.DataFrame(rows)


def add_deltas_vs_reference(
    df_long: pd.DataFrame,
    *,
    ref_slice: str = "ALL",
    metrics: List[str] = None,
) -> pd.DataFrame:
    """
    Adds delta columns per run: metric - metric(ref_slice).
    Requires df_long to include the reference slice row per run.
    """
    if metrics is None:
        metrics = ["macro_f1", "accuracy", "log_loss", "mean_pred_conf"]

    out = df_long.copy()
    ref = out[out["slice"] == ref_slice][["run_dir"] + metrics].copy()
    ref = ref.rename(columns={m: f"{m}__ref" for m in metrics})

    out = out.merge(ref, on="run_dir", how="left")

    for m in metrics:
        out[f"delta_{m}_vs_{ref_slice}"] = out[m] - out[f"{m}__ref"]

    return out


def add_deltas_between_two_slices(
    df_long: pd.DataFrame,
    *,
    slice_a: str,
    slice_b: str,
    metrics: List[str] = None,
) -> pd.DataFrame:
    """
    Produces one row per run: metric(slice_a) - metric(slice_b).
    Useful for hard vs control, minority vs control, etc.
    """
    if metrics is None:
        metrics = ["macro_f1", "accuracy", "log_loss", "mean_pred_conf"]

    a = df_long[df_long["slice"] == slice_a][["run_dir"] + metrics].copy()
    b = df_long[df_long["slice"] == slice_b][["run_dir"] + metrics].copy()

    a = a.rename(columns={m: f"{m}__{slice_a}" for m in metrics})
    b = b.rename(columns={m: f"{m}__{slice_b}" for m in metrics})

    merged = a.merge(b, on="run_dir", how="inner")

    for m in metrics:
        merged[f"delta_{m}__{slice_a}_minus_{slice_b}"] = merged[f"{m}__{slice_a}"] - merged[f"{m}__{slice_b}"]

    return merged


# ----------------------------
# Batch: all runs
# ----------------------------
def evaluate_all_runs(
    runs_root: Path,
    slice_ids_path: Path,
    out_dir: Path,
    *,
    include_all_reference: bool = True,
    all_slice_name: str = "ALL",
) -> pd.DataFrame:
    """
    For each run folder containing test_predictions.csv:
      - compute slice metrics
      - write run_dir/slice_metrics.json
    Also writes:
      - out_dir/slice_metrics_long.csv
      - out_dir/slice_metrics_with_deltas_vs_ALL.csv
      - (optional) out_dir/deltas__hard_minus_control.csv if those slices exist
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    slice_ids = read_json(slice_ids_path)

    # optionally add ALL slice ids = all ids in a run (computed per run)
    rows_all = []

    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        pred_csv = run_dir / "test_predictions.csv"
        if not pred_csv.exists():
            continue

        # load predictions once
        pred_df = pd.read_csv(pred_csv)
        pred_df["id"] = pred_df["id"].astype(str)

        # build per-run slice ids dict
        run_slice_ids = dict(slice_ids)

        if include_all_reference and all_slice_name not in run_slice_ids:
            run_slice_ids[all_slice_name] = pred_df["id"].tolist()

        # compute
        slice_metrics = evaluate_slices_for_run(pred_csv, run_slice_ids)

        # save per-run json
        write_json(run_dir / "slice_metrics.json", slice_metrics)

        # flatten rows
        df_long = flatten_slice_metrics(run_dir.name, slice_metrics)
        rows_all.append(df_long)

    if not rows_all:
        return pd.DataFrame()

    df_long_all = pd.concat(rows_all, ignore_index=True)
    df_long_all.to_csv(out_dir / "slice_metrics_long.csv", index=False)

    # deltas vs ALL
    if include_all_reference:
        df_with_deltas = add_deltas_vs_reference(df_long_all, ref_slice=all_slice_name)
        df_with_deltas.to_csv(out_dir / f"slice_metrics_with_deltas_vs_{all_slice_name}.csv", index=False)
    else:
        df_with_deltas = df_long_all

    # optional: hard vs control deltas (if you have these slice names)
    # common names from your pipeline:
    #   slice_ambiguous == "hard"/"control" would need to be turned into ID lists if you want them here.
    # If you stored them as ID lists in slice_ids.json, then these will exist.
    if "ambiguous_mwe_ids" in slice_ids and "control_ids" in slice_ids:
        # you can create these keys in your slice-ids creation step if desired
        pass

    return df_with_deltas

