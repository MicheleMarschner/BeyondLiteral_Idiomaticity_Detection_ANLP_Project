from pathlib import Path
from itertools import product
import importlib.util

from evaluation import compute_metrics, make_predictions
from utils.helper import set_seeds, create_experiment_dir, ensure_dirs
from utils.reporting import save_artifacts, build_test_predictions
from training import get_model
from data import load_data_splits
from config import Paths, PATHS
from models.factory import get_model_runner


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


def run_single_experiment(experiment_config, paths: Paths=PATHS, overwrite=False):
    """Run one experiment end-to-end: run and evaluate experiment and save all artifacts (config, predictions, metrics) to its experiment run folder"""
    experiment_dir = create_experiment_dir(experiment_config, paths.runs, overwrite)
    set_seeds(experiment_config['seed'])

    runner = get_model_runner(experiment_config['model_family'])

    train_df, val_df, test_df = load_data_splits(experiment_config, paths.data_preprocessed)
    train_preprocessed, val_preprocessed, test_preprocessed = runner.preprocessing(
        experiment_config, 
        train_df, 
        val_df, 
        test_df
    )
  
    train_loader, val_loader, test_loader = runner.prepare_inputs(
        train_preprocessed, 
        val_preprocessed, 
        test_preprocessed, 
        experiment_config
    )

    model = get_model(experiment_config, experiment_dir, train_loader, val_loader, runner)

    test_proba = runner.predict_proba(model, test_loader)
    test_preds = make_predictions(test_proba)

    test_predictions = build_test_predictions(test_preds, test_preprocessed['Label'], test_proba, test_preprocessed["MWE"])

    metrics = compute_metrics(test_preprocessed['Label'], test_preds)

    save_artifacts(
        run_dir=experiment_dir,
        config=experiment_config,
        test_predictions=test_predictions,      
        metrics=metrics,         
    )


def run_experiments(experiments_path, overwrite=False):
    """Run all experiments defined in the template file and save the results per experiment"""
    template = load_experiment_template(experiments_path)
    run_configs = expand_template(template)   # produces a cartesian-product experiment grid
    for config in run_configs:
        try:
            ensure_dirs(PATHS)
            run_single_experiment(config, PATHS, overwrite)
        except AssertionError as e:
            print("Assertion error: ", e)
    
    print("All experiments successfully run")