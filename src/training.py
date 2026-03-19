from torch import nn
from pathlib import Path
import joblib
from transformers import (AutoModelForSequenceClassification, Trainer, TrainingArguments)
import pandas as pd
from typing import Dict, Union, Tuple, Any

from utils.helper import ensure_dir, write_json, read_json
from logger.wandb_logger import (
    update_wandb_best_params, 
    update_wandb_best_curves_summary, 
    update_wandb_best_result_summary,
    log_wandb_tuning_results_table
)


def _load_model(model_family: str, model_path: Path, best_params: Dict[str, str]) -> Union[nn.Module, Any]:
    """Load existing model from checkpoint"""

    if model_family.startswith("logreg"):
            model = joblib.load(model_path)
    else:
        # Load model with model weights and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        best_params["tokenizer_source"] = str(model_path)
        
        # Freeze model layers
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        args = TrainingArguments(
            output_dir=str(model_path),
            per_device_eval_batch_size=int(best_params.get("batch_size", 8)),
            report_to=["none"],
        )

        model = Trainer(model=model, args=args)

    return model


def get_model(
    experiment_config: Dict[str, Any], 
    experiment_dir: Path, 
    train_data: pd.DataFrame, 
    dev_data: pd.DataFrame, 
    runner: Any,
    wandb_run: Any = None
) -> Tuple[Union[nn.Module, Any], Dict[str, Any]]:
    """
    Load existing model checkpoint and results if available; otherwise train and save the 
    best model for this config
    """
    model_family = experiment_config['model_family']
    if model_family.startswith("logreg"):
        model_path = experiment_dir / f"{model_family}.joblib"
    else:
        model_path = experiment_dir / model_family

    # Check for existing artifacts
    if not model_path.exists():
        print("Model not found. Training...")
        
        # Trigger full training pipeline
        model, tuning_results, best_params, best_curves = runner.tune(
            config=experiment_config,
            model_path=model_path,
            train_df=train_data,
            dev_df=dev_data,
            wandb_run=wandb_run
        )

        # save artifacts locally
        tuning_results.sort(key=lambda d: d.get("best_dev_macro_f1", float("-inf")), reverse=True)
        best_result = tuning_results[0] if tuning_results else None
        
        ensure_dir(experiment_dir)
        write_json(experiment_dir / "best_params.json", best_params)
        write_json(experiment_dir / "tuning_results.json", tuning_results)
        pd.DataFrame(tuning_results).to_csv(experiment_dir / "tuning_results.csv", index=False)
        write_json(experiment_dir / "learning_curves.json", best_curves)

        # save artifacts to wandb
        update_wandb_best_params(wandb_run, best_params)
        update_wandb_best_curves_summary(wandb_run, best_curves)
        update_wandb_best_result_summary(wandb_run, best_result)
        log_wandb_tuning_results_table(wandb_run, tuning_results)

    else:    # Re-instantiate a new model and load the best weights (Frozen state)
        print("Loading existing model...")

        best_params = read_json(experiment_dir / "best_params.json")
        model = _load_model(model_family, model_path, best_params)
    
    return model, best_params