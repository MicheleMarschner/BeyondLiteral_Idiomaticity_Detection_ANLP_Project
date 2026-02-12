from typing import Dict, List, Any
import numpy as np


class ModelRunner:

    def prepare_features(self, params, config, train_df=None, val_df=None, test_df=None):
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
    

    def initialize(self, params, seed, model_family):
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
        model_path,
        train_df,
        val_df,
        threshold: float = 0.5,
    ):
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
    

    def predict_proba(self, model, dataloader) -> np.ndarray:
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