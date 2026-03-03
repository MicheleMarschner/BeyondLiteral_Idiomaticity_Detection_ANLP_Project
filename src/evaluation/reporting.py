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


