from pathlib import Path
from itertools import product
import importlib.util
import os
import argparse

from typing import Dict, Any

from evaluation.metrics import compute_metrics, make_predictions, compute_metrics_per_language
from utils.helper import set_seeds, create_experiment_dir, ensure_dirs
from evaluation.reporting import save_artifacts, build_test_predictions
from training import get_model
from data.data import load_data_splits, build_inputs_for_splits, compute_and_check_split_stats
from config import Paths, PATHS
from models.factory import get_model_runner
from logger.wandb_logger import (
    init_wandb_run,
    update_wandb_split_stats_summary,
    log_wandb_final_metrics,
    log_wandb_artifacts,
    finish_wandb_run,
)


def parse_args():
    """Parse command-line arguments for local or Slurm-based experiment execution."""
    parser = argparse.ArgumentParser(description="Run experiment template locally or with Slurm array tasks.")
    parser.add_argument(
        "--experiments_path",
        type=str,
        required=True,
        help="Path to the Python experiment template file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing experiment directories.",
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Run only the config selected by SLURM_ARRAY_TASK_ID.",
    )
    return parser.parse_args()


def load_experiment_template(file: Path):
    """Load experiment setup from a Python config file by importing and executing it"""
    # Import an import spec from the file path given as argument
    spec = importlib.util.spec_from_file_location("exp", str(file))

    # Create a new empty module object from that spec (nothing executed yet).
    module = importlib.util.module_from_spec(spec)

    # Execute the module code (i.e., run the .py file) which makes the top-level 
    # variables available via `module.EXPERIMENTS`
    spec.loader.exec_module(module)

    # Return the experiment object(s) defined in the experiments file.
    return module.EXPERIMENTS


def expand_template(experiments_template):
    """Expand an experiment template into a list of concrete runs (one per combination of settings)"""
    runs = []
    for setting, lang, input_variant, model_family, seed in product(
        experiments_template.settings,
        experiments_template.languages, 
        experiments_template.input_variant, 
        experiments_template.model_families, 
        experiments_template.seeds
    ):
        runs.append({
            "setting": setting,
            "language_mode": experiments_template.language_mode,
            "language": lang,
            "input_variant": input_variant,
            "model_family": model_family,
            "seed": seed,
        })
    return runs


def select_run_configs_for_execution(run_configs, use_slurm: bool):
    """Return either all run configs or the single config selected by Slurm array task id."""
    if not use_slurm:
        return run_configs

    total_runs = len(run_configs)
    task_id_raw = os.environ.get("SLURM_ARRAY_TASK_ID")

    if task_id_raw is None:
        raise ValueError("Missing SLURM_ARRAY_TASK_ID although --slurm was set.")

    task_id = int(task_id_raw)

    if task_id < 0 or task_id >= total_runs:
        raise IndexError(
            f"SLURM_ARRAY_TASK_ID={task_id} out of range for {total_runs} runs."
        )

    selected_config = run_configs[task_id]
    print(f"[slurm] running task_id={task_id} of {total_runs}")

    return [selected_config]


def run_single_experiment(experiment_config: Dict[str, Any], paths: Paths=PATHS, overwrite: bool=False) -> None:
    """Run one experiment end-to-end: run and evaluate experiment and save all artifacts (config, predictions, metrics) to its experiment run folder"""
   
    experiment_dir = create_experiment_dir(experiment_config, paths.runs, overwrite)
    wandb_run = init_wandb_run(experiment_config, experiment_dir)

    try: 
        set_seeds(experiment_config['seed'])

        runner = get_model_runner(experiment_config['model_family'])

        train_df, val_df, test_df = load_data_splits(experiment_config, paths.data_preprocessed)
        split_stats, is_too_small, reasons = compute_and_check_split_stats(train_df, val_df, test_df, experiment_config['language'])
        
        update_wandb_split_stats_summary(
            wandb_run,
            split_stats,
            is_too_small,
            reasons,
        )
        
        if is_too_small:
            print(f"[skip] experiment | " + " ; ".join(reasons))
            return None
        
        train_data, val_data, test_data = build_inputs_for_splits(train_df, val_df, test_df, experiment_config)

        model, best_params = get_model(
            experiment_config, 
            experiment_dir, 
            train_data, 
            val_data, 
            runner, 
            wandb_run
        )

        _, test_loader, _ = runner.prepare_features(
            params=best_params,
            config=experiment_config,
            train_df=train_data,
            test_df=test_data
        )
        test_proba = runner.predict_proba(model, test_loader)
        test_preds = make_predictions(test_proba)

        test_predictions = build_test_predictions(test_data['ID'], test_preds, test_data['label'], test_proba)

        if experiment_config["language_mode"] == "multilingual":
            metrics = compute_metrics_per_language(
                gold_labels=test_data["label"],
                preds=test_preds,
                languages=test_data["Language"],
                threshold=0.5,
            )
        else:
            metrics = compute_metrics(test_data["label"], test_preds)  

        save_artifacts(
            run_dir=experiment_dir,
            config=experiment_config,
            split_stats=split_stats,
            test_predictions=test_predictions,      
            metrics=metrics,         
        )

        log_wandb_final_metrics(wandb_run, metrics)
        log_wandb_artifacts(wandb_run, experiment_dir)

    finally:
        finish_wandb_run(wandb_run)


def run_experiments(
    experiments_path: Path,
    overwrite: bool=False,
    use_slurm: bool=False,
):
    """Run all experiments defined in the template file and save the results per experiment"""
    template = load_experiment_template(experiments_path)
    run_configs = expand_template(template)   # produces a cartesian-product experiment grid
    run_configs = select_run_configs_for_execution(run_configs, use_slurm)

    ensure_dirs(PATHS)

    for config in run_configs:
        try:
            res = run_single_experiment(config, PATHS, overwrite)

            if res == None:             # if experiment got skipped
                continue

        except AssertionError as e:
            print("Assertion error: ", e)
    
    print("All experiments successfully run")


def main():
    """Parse command-line arguments and run experiments."""
    args = parse_args()
    run_experiments(
        experiments_path=Path(args.experiments_path),
        overwrite=args.overwrite,
        use_slurm=args.slurm,
    )


if __name__ == "__main__":
    main()