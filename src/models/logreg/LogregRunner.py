import numpy as np
import joblib
from sklearn.model_selection import ParameterGrid
from pathlib import Path

from typing import Dict, Tuple, Any
import pandas as pd

from src.models.logreg.model import LogisticRegression
from utils.helper import set_seeds
from models.logreg.featurize import build_featurizer
from models.logreg.param_grid import tfidf_param_grid, word2vec_param_grid



class LogRegRunner:
    def prepare_features(self, 
        params: Dict[str, Any], 
        config: Dict[str, Any], 
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fit a featurizer on train text and transform train/test into feature matrices"""
        
        set_seeds(config['seed'])

        featurizer = build_featurizer(config['model_family'], params)
        
        X_train, y_train = train_df['input'], train_df['label'].astype(int) # changed label_col default to "label" according to the data files, was "Label" before
        X_test, y_test   = test_df['input'],  test_df['label'].astype(int)  # changed label_col default to "label" according to the data files, was "Label" before

        X_train = featurizer.fit_transform(X_train)
        X_test  = featurizer.transform(X_test)

        return (X_train, y_train), (X_test, y_test), featurizer


    def initialize(self, params: Dict[str, Any], seed: int, model_family: str) -> LogisticRegression:
        """Create a LogisticRegression instance from hyperparameters"""

        print("learning_rate", params["learning_rate"])
        print("num_iterations", params["num_iterations"])
        print("lambda_reg", params["lambda_reg"])

        model = LogisticRegression(
            learning_rate=params.get("learning_rate", 0.01),
            num_iterations=params.get("num_iterations", 1000),
            lambda_reg=params.get("lambda_reg", 0.0)
        ) 
        return model
    
    def tune(self, 
        config: Dict[str, Any],
        model_path: Path,
        train_df: pd.DataFrame,
        dev_df: pd.DataFrame,
        threshold: float=0.5
    ) -> Tuple[LogisticRegression, Dict[str, Any], Dict[str, Any]]:
        """Grid-search hyperparameters, save the best model bundle, and return best model and other results"""
        
        model_family = config["model_family"]

        # choose the grid based on the experiment config
        if model_family == "logreg_tfidf":
            param_grid = tfidf_param_grid
        elif model_family == "logreg_word2vec":
            param_grid = word2vec_param_grid
        else:
            raise ValueError(f"Unknown model_family: {model_family}")
        
        total = len(list(ParameterGrid(param_grid)))

        results = []
        bundle = {}
        best_dev_macro_f1_overall = -1.0
        best_model = None
        best_params = None
        best_featurizer = None
        best_curves = None

        # Run through hyperparameter grid
        for i, params in enumerate(ParameterGrid(param_grid)):
            print()
            train_data, dev_data, featurizer = self.prepare_features(params=params, config=config, train_df=train_df, test_df=dev_df)
            X_train, y_train = (train_data[0], train_data[1])
            X_dev, y_dev = (dev_data[0], dev_data[1])
            
            model = self.initialize(params, config['seed'], config['model_family'])

            best_dev_macro_f1, best_train_macro_f1, loss_curves = model.fit(config, X_train, y_train, X_dev, y_dev)

            results.append({
                **params,
                "best_dev_macro_f1": best_dev_macro_f1,
                "best_train_macro_f1": best_train_macro_f1,
            })

            if best_dev_macro_f1 > best_dev_macro_f1_overall:
                best_dev_macro_f1_overall = best_dev_macro_f1
                best_model = model
                best_params = dict(params)
                best_featurizer = featurizer
                best_curves = loss_curves
            
            # display progress of grid search
            if (i+1) % 50 == 0 or i == 0:
                print(f"[tune] {i+1}/{total} ({(i+1)/total:.1%})")

        if best_model is None:
            raise RuntimeError("Tuning failed: no valid parameter combination produced a trained model.")

        bundle = {
            "model": best_model,                 # contains weights+bias
            "featurizer": best_featurizer,
            "best_params": best_params,
            "threshold": threshold,
            "macro_f1": best_dev_macro_f1_overall,
        }
        joblib.dump(bundle, model_path)

        return best_model, results, best_params, best_curves


    def predict_proba(self, model: LogisticRegression, X: Any) -> np.ndarray:
        """Wrapper to get literal-class probabilities from a model"""
        
        if isinstance(X, tuple) and len(X) >= 1:
            X = X[0]

        proba_literal = model.predict_proba(X)

        return proba_literal