from typing import Dict, Tuple, Optional, Mapping, Iterable, Any
import numpy as np
from torch import nn

from pathlib import Path
import pandas as pd


class ModelRunner:

    def prepare_features(
        self, 
        params: Dict[str, Any], 
        config: Dict[str, Any], 
        train_df: Optional[pd.DataFrame]=None, 
        val_df: Optional[pd.DataFrame]=None, 
        test_df: Optional[pd.DataFrame]=None, 
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Input: 
            train_data/val_data/test_df: Pandas DataFrames containing the already modified input-variant examples according to the experiment `config` and labels

            config: Experiment configuration dict (setting, language, input_variant, model_family, seed)

        1. Prepare train/val/test for the transformer models: tokenize inputs according to params
        2. Setup data Loaders. Use config['seed'] for deterministic shuffling of the train data

        Output:
            train/val/test data loader
        """
        pass

        # return train_loader, val_loader, test_loader
    

    def initialize(self, params: Dict[str, Any], seed: int, model_family: str) -> nn.Module :
        """
        Initializes a model instance with a given set of parameters

        Input: set of parameters
        Output: Model Instance (.to(DEVICE)) where DEVICE is a constant created during environment setup (from config import DEVICE)
        """
        pass
        
        # return model.to(DEVICE)
    
    
    def tune(
        self,
        config: Dict[str, Any],
        model_path: Path,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        threshold: float = 0.5,
    ) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
        """
        Entry point for hyperparameter tuning and training.

        Run a sweep over `param_grid` which needs to be created, call prepare_features (tokenize inputs, create dataloaders) and train a model for each parameter setting on `train_loader`,
        evaluate on `val_loader` using the chosen threshold, and select the best configuration
        (by the macro F1 score). The best model is then saved to `model_path` and returned along with a summary of 
        all tried settings.

        Use the seed in the config (config['seed]) for reproducible training runs and deterministic train data shuffling

        Inputs:
            config: Experiment configuration (model_family, seed, input_variant, etc.)
            model_path: path where best checkpoint is saved
            train_df: PandaDataframes
            val_df: PandaDataframes
            param_grid: List of hyperparameter dicts to try (each dict defines one run). Needs to be defined first
            threshold: Decision threshold to convert probabilities into 0/1 predictions for validation metrics. Currently not changed

        Outputs:
            best_model: Trained model instance with the best hyperparameter setting.
            results: List of dicts, one per tried setting, containing params + validation score(s).
            best_params: Dict of the selected best hyperparameters (the entry from `param_grid` with highest score, including for the tokenizer).
        """
        pass

        # for params in params_grid:
        #    train_loader, val_loader, _ = self.prepare_features(params, config, train_df=train_df, val_df=val_df)
        #    model = self.initialize(config, params)
        #    model, val_score = model.train(model, train_loader, val_loader, config['seed'], params) or self.train(model, train_loader, val_loader, config['seed'], params)
        #    results.append({**params, "val_score": float(val_score)})
        #    if val_score > best_score:
        #        ....

        # return best_model, results, best_params
    

    def predict_proba(self, model: nn.Module, dataloader: Iterable[Mapping[str, Any]]) -> np.ndarray:
        """
        Compute positive-class probabilities for a transformer over a dataloader. (Wrapper function)
        Input:
            model: The transformer model instance
            dataloader: iterable over batches of val/test data

        Output: 
            proba: positive-class probabilities for all examples in `dataloader`.
        """
        pass

        # return proba