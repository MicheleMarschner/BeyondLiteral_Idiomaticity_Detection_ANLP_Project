from torch import nn
from pathlib import Path
import joblib
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import pandas as pd
from typing import Dict, Union, Tuple, Any

from utils.helper import ensure_dir, write_json, read_json


def get_model(
    experiment_config: Dict[str, Any], 
    experiment_dir: Path, 
    train_data: pd.DataFrame, 
    val_data: pd.DataFrame, 
    runner: Any
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
        model, tuning_results, best_params, best_curves = runner.tune(experiment_config, model_path, train_data, val_data)

        tuning_results.sort(key=lambda d: d.get("dev_macro_f1", float("-inf")), reverse=True)
        ensure_dir(experiment_dir)
        write_json(experiment_dir / "best_params.json", best_params)
        write_json(experiment_dir / "tuning_results.json", tuning_results)
        pd.DataFrame(tuning_results).to_csv(experiment_dir / "tuning_results.csv", index=False)
        write_json(experiment_dir / "learning_curves.json", best_curves)

    else:    # Re-instantiate a clean model architecture and load the best weights (Frozen state)
        print("Loading existing model...")

        best_params = read_json(experiment_dir / "best_params.json")

        if model_family.startswith("logreg"):
            model = joblib.load(model_path)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            # tokenizer = AutoTokenizer.from_pretrained(model_path)

            for p in model.parameters():
                p.requires_grad = False

            model.eval()
    
    return model, best_params
    