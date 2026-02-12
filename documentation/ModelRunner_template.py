from typing import Dict, List, Any
import numpy as np


class ModelRunner:

    def preprocessing(self, config, train_data, val_data, test_data):
        """
        Prepare train/val/test for the transformer models
            - extract input-variant samples and labels from the DataFrames
            - tokenize inputs
            
        Inputs:
            config: Experiment configuration dict (setting, language, input_variant, model_family, seed)
            train_data/val_data/test_data: Pandas DataFrames containing the already modified input-variant examples according to the experiment `config` and labels


        Outputs:
            train_data/val_data/test_data: tokenized representations and labels as datasets
   
        """

        # return train_ds, val_ds, test_ds


    def prepare_inputs(self, train_ds, val_ds, test_ds, config):
        """
        Input: 
            train/val/test_ds: Datasets
            config: Experiment configuration dict (setting, language, input_variant, model_family, seed)

        Setup data Loaders. Use config['seed'] for deterministic shuffling of the train data

        Output:
            train/val/test data loader
        """
        pass

        # return train_loader, val_loader, test_loader
    

    def initialize(self, params):
        """
        Initializes a model instance with a given set of parameters

        Input: set of parameters
        Output: Model Instance (.to(DEVICE)) where DEVICE is a constant created during environment setup (from config import DEVICE)
        """
        pass
        
        # return model.to(DEVICE)
    
    
    def fit(
        self,
        config: Dict[str, Any],
        model_path,
        train_loader,
        val_loader,
        param_grid: List[Dict[str, Any]],
        threshold: float = 0.5,
    ):
        """
        Entry point for hyperparameter tuning and training.

        Run a sweep over `param_grid` which needs to be created, train a model for each parameter setting on `train_loader`,
        evaluate on `val_loader` using the chosen threshold, and select the best configuration
        (by the macro F1 score). The best model is then saved to `model_path` and returned along with a summary of 
        all tried settings.

        Use the seed in the config (config['seed]) for reproducible training runs

        Inputs:
            config: Experiment configuration (model_family, seed, input_variant, etc.)
            model_path: path where best checkpoint is saved
            train_loader: Train Data Loader
            val_loader: Val Data Loader
            param_grid: List of hyperparameter dicts to try (each dict defines one run). Needs to be defined first
            threshold: Decision threshold to convert probabilities into 0/1 predictions for validation metrics. Currently not changed

        Outputs:
            best_model: Trained model instance with the best hyperparameter setting.
            results: List of dicts, one per tried setting, containing params + validation score(s).
            best_params: Dict of the selected best hyperparameters (the entry from `param_grid` with highest score).
        """
        pass

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