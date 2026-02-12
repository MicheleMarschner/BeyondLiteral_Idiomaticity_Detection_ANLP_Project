import torch
from torch import nn
from pathlib import Path
import joblib

import pandas as pd
from typing import Dict, Union, Tuple, Any

from src.config import DEVICE
from src.utils.helper import ensure_dir, write_json, read_json

def load_model_checkpoint(model: Union[nn.Module, Any], model_path: Path, device=DEVICE) -> Union[nn.Module, Any]:
    """Load a saved model, freeze parameters and set it to inference mode"""

    if model_path.suffix == ".joblib":
        return joblib.load(model_path)
    
    if model_path.suffix == ".pth":
        # Load serialized weights;
        state_dict = torch.load(model_path, weights_only=True, map_location=device)     # map_location ensures compatibility if moving between CPU/GPU
        model.load_state_dict(state_dict, strict=True)      # Enforce exact structural match between the architecture and the saved weights

        # Freeze all parameters
        for p in model.parameters():
            p.requires_grad = False

        model.eval()

        return model
    
    raise ValueError(f"Unsupported checkpoint type: {model_path.suffix}")


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
    ext = ".joblib" if model_family.startswith("logreg") else ".pth"

    ## check model according to experiment config name
    ## !TODO: what should be the model name
    model_path = Path(f"{experiment_dir}/{model_family}{ext}")

    # Check for existing artifacts (Lazy Loading)
    if not model_path.exists():
        
        # Trigger full training pipeline
        model, tuning_results, best_params = runner.tune(experiment_config, model_path, train_data, val_data)

        tuning_results.sort(key=lambda d: d.get("val_macro_f1", float("-inf")), reverse=True)
        ensure_dir(experiment_dir)
        write_json(experiment_dir / "best_params.json", best_params)
        write_json(experiment_dir / "tuning_results.json", tuning_results)

    else:    # Re-instantiate a clean model architecture and load the best weights (Frozen state)
        print("Load model...")

        best_params = read_json(experiment_dir / "best_params.json")

        model = runner.initialize(best_params, experiment_config['seed'], experiment_config['model_family'])
        model = load_model_checkpoint(model, model_path)

    return model, best_params
    