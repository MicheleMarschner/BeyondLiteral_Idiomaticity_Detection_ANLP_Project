from pathlib import Path
import pandas as pd

from typing import Dict, Any

from config import PATHS, Paths
from utils.helper import read_json, ensure_dir


def _format_mean_std(mean, std):
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


def create_overview_per_experiment_config(selected_config: Dict[str, Any], experiments_root: Path, results_root: Path) -> None:
    """
    Create and save an overview table for one experiment config (setting/language/input_variant),
    comparing model families aggregated over seeds
    """

    setting = selected_config.get('setting')
    language = selected_config.get('language')
    input_variant = selected_config.get('input_variant')

    rows = []

    # scan all run folders and collect metrics for runs that match the selected experiment configs
    for experiment_dir in sorted(experiments_root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        
        experiment_config_path = experiment_dir / "experiment_config.json"
        experiment_config = read_json(experiment_config_path)

        # match: same language, setting, input_variant
        if experiment_config.get('setting') != setting:
            continue
        if experiment_config.get('language') != language:
            continue
        if experiment_config.get('input_variant') != input_variant:
            continue
        
        metrics_path = experiment_dir / "metrics.json"
        metrics = read_json(metrics_path)

        # store one row per run
        rows.append({
            "run_dir": experiment_dir.name,
            "model_family": experiment_config.get('model_family'),
            "seed": experiment_config.get('seed'),
            "macro_f1": metrics.get('macro_f1'),
            "macro_precision": metrics.get('macro_precision'),
            "macro_recall": metrics.get('macro_recall'),
        })

    df = pd.DataFrame(rows)
    
    if df.empty:
        return
    
    # ensure metric columns are numeric
    for col in ["macro_f1", "macro_precision", "macro_recall"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # aggregate over seeds per model family
    agg = (
        df.groupby("model_family", dropna=False)
          .agg(
              n_runs=('run_dir', 'count'), 
              n_seeds=('seed', pd.Series.nunique),
              macro_f1_mean=('macro_f1', 'mean'),
              macro_f1_std=('macro_f1', 'std'),
              macro_precision_mean=('macro_precision', 'mean'),
              macro_precision_std=('macro_precision', 'std'),
              macro_recall_mean=('macro_recall', 'mean'),
              macro_recall_std=('macro_recall', 'std'),
          )
          .reset_index()
    )

    # create a wide "overview" table: metric rows × model columns with 
    # rows: macro_f1 / macro_precision / macro_recall
    # columns: each model_family
    # cells: formatted "mean ± std"
    metrics = ["macro_f1", "macro_precision", "macro_recall"]
    overview_df = pd.DataFrame(index=metrics)

    for model_family, row in agg.set_index('model_family').iterrows():
        for m in metrics:
            overview_df.loc[m, model_family] = _format_mean_std(row[f"{m}_mean"], row[f"{m}_std"])

    # sort columns by macro_f1_mean desc
    order = (
        agg.sort_values("macro_f1_mean", ascending=False, na_position="last")["model_family"]
           .astype(str)
           .tolist()
    )
    overview_df = overview_df.reindex(columns=order)

    # save overview table
    out_dir = results_root / "tables"
    ensure_dir(out_dir)
    out_path = out_dir / f"evaluation_overview_{setting}__{language}__{input_variant}.csv"
    overview_df.reset_index(names="metric").to_csv(out_path, index=False)


def create_overview_per_model(
    experiments_root: Path,
    model_family: str,
    results_root: Path
) -> None:
    """Create and save an overview table for all runs belonging to a model family, sorted by macro_F1 score"""

    rows = []

    # scan all run folders and collect metrics for runs that match the selected model family
    for experiment_dir in sorted(experiments_root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        
        experiment_config_path = experiment_dir / "experiment_config.json"
        experiment_config = read_json(experiment_config_path)
        
        # match: model family
        if experiment_config.get('model_family') != model_family:
            continue

        metrics_path = experiment_dir / "metrics.json"
        if metrics_path.exists():
            metrics = read_json(metrics_path)

        # store one row per run
        row = {
            "setting": experiment_config.get('setting'),
            "language_mode": experiment_config.get('language_mode'),
            "language": experiment_config.get('language'),
            "input_variant": experiment_config.get('input_variant'),
            "seed": experiment_config.get('seed'),
            "model_family": experiment_config.get('model_family'),
            
            "macro_f1": metrics.get('macro_f1'),
            "macro_precision": metrics.get('macro_precision'),
            "macro_recall": metrics.get('macro_recall')
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    
    # ensure metric columns are numeric
    for col in ["macro_f1", "macro_precision", "macro_recall"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(by=["macro_f1"], ascending=False, na_position="last").reset_index(drop=True)
    
    # save overview table
    out_dir = results_root / "tables"
    ensure_dir(out_dir)
    out_path = out_dir / f"evaluation_overview_{model_family}.csv"
    df.to_csv(out_path, index=False)

    
def run_evaluation(paths: Paths=PATHS, model_family=None, experiment_config=None):
    """Aggregates and saves results in an evaluation overview table"""
    
    if model_family is not None:
        create_overview_per_model(model_family, paths.runs, paths.results)
    
    if experiment_config is not None:
        create_overview_per_experiment_config(experiment_config, paths.runs, paths.results)