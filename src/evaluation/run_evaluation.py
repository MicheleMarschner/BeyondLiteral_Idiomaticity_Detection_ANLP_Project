from pathlib import Path
import pandas as pd 

from config import PATHS
from utils.helper import ensure_dir, read_json



def flatten_run(run_dir: Path) -> list[dict]:
    exp_config = read_json(run_dir / "experiment_config.json")
    metrics = read_json(run_dir / "metrics.json")

    input_variant = exp_config["input_variant"]
    base = {
        "run_dir": run_dir.name,
        "setting": exp_config["setting"],
        "language_mode": exp_config["language_mode"],
        "language": exp_config["language"],
        "model_family": exp_config["model_family"],
        "seed": exp_config["seed"],
        "context": input_variant["context"],
        "features": ",".join(input_variant["features"]),
        "include_mwe_segment": input_variant["include_mwe_segment"],
        "transform": input_variant["transform"],
    }

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


def create_evaluation_overview(experiments_root, results_root) -> pd.DataFrame:
    df = load_all_runs(experiments_root)
    ensure_dir(results_root)

    # Save master long table (single seed -> one row per run per eval_language)
    df.to_csv(results_root / "master_metrics_long.csv", index=False)

    return df

def run_evaluation():
    overview_df = create_evaluation_overview(experiments_root=PATHS.runs, results_root=PATHS.results)

