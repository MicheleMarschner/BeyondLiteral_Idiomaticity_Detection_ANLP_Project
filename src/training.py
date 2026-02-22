import torch
from torch import nn
from pathlib import Path
import joblib
from transformers import AutoModelForSequenceClassification, Trainer

import pandas as pd
from typing import Dict, Union, Tuple, Any


from config import DEVICE
from utils.helper import ensure_dir, write_json, read_json



'''
def load_model_checkpoint(model: Union[nn.Module, Any], model_path: Path, device=DEVICE) -> Union[nn.Module, Any]:
    THE FUNCTION IS CURRENTLY UNUSED AS WE NO LONGER USE CUSTOM .pth CHECKPOINTS FOR OUR MODELS.
    HuggingFace Transformers are saved using `save_pretrained()` and are loaded via `from_pretrained()` instead.

    """Load a saved model, freeze parameters and set it to inference mode"""

    if model_path.suffix == ".joblib":
        return joblib.load(model_path)
    
    if model_path.suffix == ".pth":
        # Load serialized weights
        state_dict = torch.load(model_path, weights_only=True, map_location=device)     # map_location ensures compatibility if moving between CPU/GPU
        model.load_state_dict(state_dict, strict=True)      # Enforce exact structural match between the architecture and the saved weights

        # Freeze all parameters
        for p in model.parameters():
            p.requires_grad = False

        model.eval()

        return model
    
    raise ValueError(f"Unsupported checkpoint type: {model_path.suffix}")
'''

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
        # sklearn models are saved as single .joblib files
        model_path = experiment_dir / f"{model_family}.joblib"
    else:
        # HuggingFace models are saved as directories via save_pretrained()
        model_path = experiment_dir / model_family

    # Check for existing artifacts
    if not model_path.exists():
        print("Model not found. Training...")
        
        # Trigger full training pipeline
        model, tuning_results, best_params, best_curves = runner.tune(experiment_config, model_path, train_data, val_data)

        tuning_results.sort(key=lambda d: d.get("val_macro_f1", float("-inf")), reverse=True)
        ensure_dir(experiment_dir)
        write_json(experiment_dir / "best_params.json", best_params)
        write_json(experiment_dir / "tuning_results.json", tuning_results)
        pd.DataFrame(tuning_results).to_csv(experiment_dir / "tuning_results.csv", index=False)
        write_json(experiment_dir / "learning_curves.json", best_curves)

    else:    # Re-instantiate a clean model architecture and load the best weights (Frozen state)
        print("Loading existing model...")

        best_params = read_json(experiment_dir / "best_params.json")

        if model_family.startswith("logreg"):
            # load sklearn model
            model = joblib.load(model_path)
        else:
            # load HuggingFace model directory
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.eval()
    
    return model, best_params
    