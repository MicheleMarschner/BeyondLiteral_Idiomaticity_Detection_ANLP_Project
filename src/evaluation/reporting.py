import pandas as pd
from pathlib import Path
from typing import Dict, Sequence, Any

from utils.helper import ensure_dir, read_json, write_json


def build_test_predictions(
    ids: Sequence[str],
    preds: Sequence[int],
    gold_labels: Sequence[int],
    proba: Sequence[float]
) -> Dict[str, Any]:
    """Create a record per test example with gold label, prediction, probability, and the MWE string"""
    
    rows = [
        {
            "id": str(id),
            "label": int(y),
            "test_pred": int(preds),
            "test_proba_literal": float(proba)
        }
        for id, y, preds, proba in zip(ids, gold_labels, preds, proba)
    ]
    return rows


def flatten_metrics(metrics):
    """Convert metrics into a flat row to log results and combine them across experiments for later analysis."""
    out = {}

    if "overall" in metrics:
        blocks = [("overall_", metrics["overall"])] + [
            (f"{lang}_", metric) for lang, metric in metrics.get("per_language", {}).items()
        ]
    else:
        blocks = [("", metrics)]

    for prefix, metric in blocks:
        cm = metric.get("confusion_matrix_values", {})
        out.update({
            f"{prefix}accuracy": metric.get("accuracy"),
            f"{prefix}macro_precision": metric.get("macro_precision"),
            f"{prefix}macro_recall": metric.get("macro_recall"),
            f"{prefix}macro_f1": metric.get("macro_f1"),
            f"{prefix}tp": cm.get("tp"),
            f"{prefix}tn": cm.get("tn"),
            f"{prefix}fp": cm.get("fp"),
            f"{prefix}fn": cm.get("fn"),
        })

    return out


def save_artifacts(
    run_dir: Path,
    split_stats: Dict[str, Any],
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    test_predictions: Dict[str, Any]
) -> None:
    """Create the experiment folder and save the results such as metrics and per-example test outputs as JSON or CSV files"""
    
    ensure_dir(run_dir)
    write_json(run_dir / "experiment_config.json", config)
    write_json(run_dir / "metrics.json", metrics)
    pd.DataFrame(split_stats).to_csv(run_dir / "split_stats.csv", index=False)
    pd.DataFrame(test_predictions).to_csv(run_dir / "test_predictions.csv", index=False)
    pd.DataFrame([flatten_metrics(metrics)]).to_csv(run_dir / "metrics.csv", index=False)

    print(f"All files successfully written to {run_dir}")


def flatten_run(run_dir: Path) -> list[dict]:
    cfg = read_json(run_dir / "experiment_config.json")
    metrics = read_json(run_dir / "metrics.json")

    iv = cfg.get("input_variant", {})
    base = {
        "run_dir": run_dir.name,
        "setting": cfg.get("setting"),
        "language_mode": cfg.get("language_mode"),
        "language": cfg.get("language"),  # training language label in config
        "model_family": cfg.get("model_family"),
        "seed": cfg.get("seed"),
        "context": iv.get("context"),
        "features": ",".join(iv.get("features", [])) if isinstance(iv.get("features"), list) else iv.get("features"),
        "include_mwe_segment": iv.get("include_mwe_segment"),
        "transform": iv.get("transform"),
    }

    rows: list[dict] = []

    def add_block(eval_language: str, m: dict):
        cm = m.get("confusion_matrix_values", {})
        rows.append({
            **base,
            "eval_language": eval_language,
            "accuracy": m.get("accuracy"),
            "macro_f1": m.get("macro_f1"),
            "macro_precision": m.get("macro_precision"),
            "macro_recall": m.get("macro_recall"),
            "tp": cm.get("tp"),
            "tn": cm.get("tn"),
            "fp": cm.get("fp"),
            "fn": cm.get("fn"),
        })

    # multilingual metrics
    if isinstance(metrics, dict) and "overall" in metrics:
        add_block("overall", metrics.get("overall", {}))
        for lang, m in metrics.get("per_language", {}).items():
            add_block(str(lang), m)
    else:
        # per-language / cross-lingual runs usually have flat metrics
        add_block(str(cfg.get("language")), metrics)

    return rows


def load_all_runs(runs_root: Path) -> pd.DataFrame:
    all_rows: list[dict] = []
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "experiment_config.json").exists():
            continue
        if not (d / "metrics.json").exists():
            continue
        all_rows.extend(flatten_run(d))
    return pd.DataFrame(all_rows)


def _apply_filters(df: pd.DataFrame, filt: dict) -> pd.DataFrame:
    """
    Filter syntax:
      {"col": "value"}                    -> equality match
      {"col": {"contains": "substr"}}     -> substring match on str(col)
    """
    out = df
    for k, v in (filt or {}).items():
        if isinstance(v, dict) and "contains" in v:
            out = out[out[k].astype(str).str.contains(v["contains"], na=False)]
        else:
            out = out[out[k] == v]
    return out


def ablation_delta(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    metric: str = "macro_f1",
    baseline_filter: dict | None = None,
    variant_filter: dict | None = None,
    eval_languages: tuple[str, ...] = ("overall", "EN", "PT", "GL"),
) -> pd.DataFrame:
    """
    Compute baseline vs variant vs delta for the given metric.

    Returns one row per (group_cols + eval_language).
    """
    base = _apply_filters(df.copy(), baseline_filter or {})
    var = _apply_filters(df.copy(), variant_filter or {})

    base = base[base["eval_language"].isin(eval_languages)]
    var = var[var["eval_language"].isin(eval_languages)]

    base = base[group_cols + ["eval_language", metric]].rename(columns={metric: "baseline"})
    var = var[group_cols + ["eval_language", metric]].rename(columns={metric: "variant"})

    merged = base.merge(var, on=group_cols + ["eval_language"], how="inner")
    merged["delta"] = merged["variant"] - merged["baseline"]

    return merged.sort_values(group_cols + ["eval_language"]).reset_index(drop=True)


def view_per_signal(
    df: pd.DataFrame,
    *,
    metric: str = "macro_f1",
    eval_language: str = "overall",
    train_language: str | None = None,
) -> pd.DataFrame:
    """
    One row per run, filtered by eval_language (overall/EN/PT/GL).
    Optionally filter by training language (only meaningful for per_language mode).
    """
    q = df.copy()
    q = q[q["eval_language"] == eval_language]

    if train_language is not None:
        q = q[q["language"] == train_language]

    cols = [
        "setting", "language_mode", "language", "model_family", "seed",
        "context", "include_mwe_segment", "transform", "features",
        metric
    ]
    return q[cols].sort_values(
        ["setting", "language_mode", "language", "model_family", "context", "features"]
    ).reset_index(drop=True)


def view_context_delta(
    df: pd.DataFrame,
    *,
    baseline_context: str,
    variant_context: str,
    metric: str = "macro_f1",
    eval_language: str = "overall",
    fixed: dict | None = None,
) -> pd.DataFrame:
    """
    Convenience wrapper: delta between two contexts at fixed other settings.
    """
    q = df.copy()
    q = q[q["eval_language"] == eval_language]
    if fixed:
        q = _apply_filters(q, fixed)

    group_cols = ["setting", "language_mode", "language", "model_family", "seed", "features", "transform", "include_mwe_segment"]

    return ablation_delta(
        q,
        group_cols=group_cols,
        metric=metric,
        baseline_filter={"context": baseline_context},
        variant_filter={"context": variant_context},
        eval_languages=(eval_language,),
    )


def load_split_stats_table(runs_root: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "experiment_config.json"
        stats_path = run_dir / "split_stats.csv"
        if not cfg_path.exists() or not stats_path.exists():
            continue

        cfg = read_json(cfg_path)
        stats = pd.read_csv(stats_path)

        # If your split_stats.csv is already a single-row table, this works.
        # If it's multiple rows (e.g., per split), we keep it and attach metadata.
        stats["run_dir"] = run_dir.name
        stats["setting"] = cfg.get("setting")
        stats["language_mode"] = cfg.get("language_mode")
        stats["language"] = cfg.get("language")
        stats["model_family"] = cfg.get("model_family")
        stats["seed"] = cfg.get("seed")

        rows.append(stats)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def make_paper_data_stats(df_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Produces a compact table like:
      setting | language_mode | language | n_train | n_dev | n_test | %literal_train | %literal_dev | %literal_test
    Works if split_stats.csv contains these columns (or similar).
    """
    if df_stats.empty:
        return df_stats

    # Try common column names; adjust if your split_stats.csv uses different names
    col_map_candidates = [
        ("n_train", ["n_train", "train_n", "train_size"]),
        ("n_dev",   ["n_dev", "dev_n", "val_n", "val_size", "valid_n"]),
        ("n_test",  ["n_test", "test_n", "test_size"]),
        ("p_literal_train", ["p_literal_train", "literal_rate_train", "train_literal_rate"]),
        ("p_literal_dev",   ["p_literal_dev", "literal_rate_dev", "val_literal_rate", "dev_literal_rate"]),
        ("p_literal_test",  ["p_literal_test", "literal_rate_test", "test_literal_rate"]),
    ]

    def pick(colnames):
        for c in colnames:
            if c in df_stats.columns:
                return c
        return None

    picks = {dst: pick(srcs) for dst, srcs in col_map_candidates}

    keep = ["setting", "language_mode", "language"]
    for dst, src in picks.items():
        if src is not None:
            df_stats[dst] = df_stats[src]
            keep.append(dst)

    # De-duplicate (you may have identical split_stats across multiple runs/seeds/models)
    out = df_stats[keep].drop_duplicates().sort_values(keep[:3]).reset_index(drop=True)
    return out



import pandas as pd

def split_stats(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    g = df.groupby("Language")["label"].value_counts(dropna=False).unstack(fill_value=0)
    g = g.rename(columns={0: "n_idiom", 1: "n_literal"})
    g["n"] = g["n_idiom"] + g["n_literal"]
    g["label_counts"] = g.apply(lambda r: {0: int(r["n_idiom"]), 1: int(r["n_literal"])}, axis=1)
    g = g.reset_index()
    g["split"] = split_name
    return g[["split", "Language", "n", "label_counts", "n_idiom", "n_literal"]]


def global_stats(df: pd.DataFrame) -> dict:
    counts = df["label"].value_counts().to_dict()
    return {"n": int(len(df)), "label_counts": {int(k): int(v) for k, v in counts.items()}}

# usage: global_stats(train_df)



# usage:
# train_df, dev_df, test_df = load_data_splits(...)
# stats = pd.concat([split_stats(train_df,"train"), split_stats(dev_df,"dev"), split_stats(test_df,"test")], ignore_index=True)
# stats.to_csv("data_stats.csv", index=False)

# --- usage ---
# from src.config import PATHS
# stats_long = load_split_stats_table(PATHS.runs)
# stats_long.to_csv(PATHS.results / "split_stats_long.csv", index=False)
# paper_stats = make_paper_data_stats(stats_long)
# paper_stats.to_csv(PATHS.results / "data_stats_table.csv", index=False)
# print(paper_stats)