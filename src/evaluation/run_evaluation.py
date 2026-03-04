from pathlib import Path
import pandas as pd 

from config import PATHS
from evaluation.reporting import extract_run_base
from utils.helper import ensure_dir, read_json


def flatten_run(experiment_dir: Path) -> list[dict]:
    """Flattens one run into metric rows (overall + per-language if available), with base metadata attached"""

    exp_config = read_json(experiment_dir / "experiment_config.json")
    metrics = read_json(experiment_dir / "metrics.json")
    base = extract_run_base(experiment_dir)

    rows = []

    if isinstance(metrics, dict) and "overall" in metrics:
        overall_metrics = metrics.get("overall", {})
        overall_cm = overall_metrics.get("confusion_matrix_values", {})

        rows.append({
            **base,
            "eval_language": "overall",
            "accuracy": overall_metrics.get("accuracy"),
            "macro_f1": overall_metrics.get("macro_f1"),
            "macro_precision": overall_metrics.get("macro_precision"),
            "macro_recall": overall_metrics.get("macro_recall"),
            "tp": overall_cm.get("tp"),
            "tn": overall_cm.get("tn"),
            "fp": overall_cm.get("fp"),
            "fn": overall_cm.get("fn"),
        })

        for eval_lang, lang_metrics in metrics.get("per_language", {}).items():
            lang_cm = lang_metrics.get("confusion_matrix_values", {})
            rows.append({
                **base,
                "eval_language": str(eval_lang),
                "accuracy": lang_metrics.get("accuracy"),
                "macro_f1": lang_metrics.get("macro_f1"),
                "macro_precision": lang_metrics.get("macro_precision"),
                "macro_recall": lang_metrics.get("macro_recall"),
                "tp": lang_cm.get("tp"),
                "tn": lang_cm.get("tn"),
                "fp": lang_cm.get("fp"),
                "fn": lang_cm.get("fn"),
            })

    else:
        flat_metrics = metrics if isinstance(metrics, dict) else {}
        flat_cm = flat_metrics.get("confusion_matrix_values", {})

        rows.append({
            **base,
            "eval_language": str(exp_config.get("language")),
            "accuracy": flat_metrics.get("accuracy"),
            "macro_f1": flat_metrics.get("macro_f1"),
            "macro_precision": flat_metrics.get("macro_precision"),
            "macro_recall": flat_metrics.get("macro_recall"),
            "tp": flat_cm.get("tp"),
            "tn": flat_cm.get("tn"),
            "fp": flat_cm.get("fp"),
            "fn": flat_cm.get("fn"),
        })

    return rows


def load_all_runs(experiments_root: Path) -> pd.DataFrame:
    """Loads and concatenates flattened metric rows for all experiments"""

    all_rows: list[dict] = []
    for experiment_dir in sorted(experiments_root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        
        all_rows.extend(flatten_run(experiment_dir))
    return pd.DataFrame(all_rows)


def create_evaluation_overview(experiments_root, results_root) -> pd.DataFrame:
    """Creates overview table containing base metadata + metrics for all experiments"""

    df = load_all_runs(experiments_root)
    ensure_dir(results_root)

    # Save master long table
    df.to_csv(results_root / "master_metrics_long.csv", index=False)

    return df

def run_evaluation():
    """Runs evaluation reporting and writes the master metrics table to the results directory"""
    overview_df = create_evaluation_overview(experiments_root=PATHS.runs, results_root=PATHS.results)

