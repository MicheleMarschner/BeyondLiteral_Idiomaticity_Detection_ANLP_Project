from pathlib import Path

from pandas import pd 
from src.utils.helper import ensure_dir, read_json



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
        "features": ",".join(input_variant.get("features", [])),
        "include_mwe_segment": input_variant.get("include_mwe_segment"),
        "transform": input_variant.get("transform"),
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
        add_block(str(exp_config.get("language")), metrics)

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

def run_evaluation(experiments_root, results_root):
    overview_df = create_evaluation_overview(experiments_root, results_root)

