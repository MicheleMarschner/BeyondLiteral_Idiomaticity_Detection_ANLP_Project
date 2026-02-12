from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
import numpy as np
import joblib
from pathlib import Path
import pandas as pd

from typing import Any, Tuple, Dict, Optional

from src.evaluation import compute_metrics
from src.models.logreg.featurize import TfidfWeightedWord2VecVectorizer
from src.models.logreg.param_grid import tfidf_param_grid, word2vec_param_grid


class LogRegRunner:

    def prepare_features(
        self, 
        params: Dict[str, Any], 
        config: Dict[str, Any], 
        train_df: Optional[pd.DataFrame]=None, 
        val_df: Optional[pd.DataFrame]=None, 
        test_df: Optional[pd.DataFrame]=None
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        "featurize"
        X_train, y_train = train_df['Input'], train_df['Label'].astype(int)
        X_val, y_val     = val_df['Input'],   val_df['Label'].astype(int)
        X_test, y_test   = test_df['Input'],  test_df['Label'].astype(int)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
        

    def initialize(self, params: Dict[str, Any], seed: int, model_family: str) -> Pipeline:
        """Instantiate a LogisticRegression model pipeline from config with sklearn"""

        if model_family == "logreg_word2vec":
            feat_step = ("w2v_tfidf", TfidfWeightedWord2VecVectorizer(
                vector_size=params.get("vector_size", 200),
                window=params.get("window", 5),
                min_count=params.get("min_df", 2),
                sg=1,
                negative=params.get("negative", 10),
                epochs=params.get("epochs", 10),
                workers=0,

                tfidf_min_df=params.get("min_df", 2),
                tfidf_max_df=params.get("max_df", 0.95),
                tfidf_norm=None,                 # keep None for weighting
                max_features=params.get("max_features", None),
                fallback=params.get("fallback", "mean"),
            ))
        elif model_family == "logreg_tfidf":
            feat_step = ("tfidf", TfidfVectorizer(
                analyzer="word",
                ngram_range=params.get("ngrams", (1, 2)),
                min_df=params.get("min_df", 2),
                max_df=params.get("max_df", 0.95),
                norm=params.get("norm", "l2"),
                max_features=params.get("max_features", 50000),
                lowercase=True,
            ))
        else:
            raise ValueError(f"Unknown model_family: {model_family}")

        clf = ("clf", LogisticRegression(
            max_iter=params.get("num_iterations", 2000),
            C=params.get("C", 1.0),
            solver=params.get("solver", "liblinear"),
            class_weight=params.get("class_weight", None),
            # only meaningful for elasticnet + saga; otherwise keep None
            l1_ratio=params.get("l1_ratio", None),
            random_state=seed,
        ))

        return Pipeline([feat_step, clf])
    
    
    
    def tune(
        self,
        config: Dict[str, Any],
        model_path: Path,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        threshold: float = 0.5,
    ) -> Tuple[Pipeline, Dict[str, Any], Dict[str, Any]]:
        
        model_family = config["model_family"]

        # choose the grid based on the experiment config
        if model_family == "logreg_tfidf":
            param_grid = tfidf_param_grid
        elif model_family == "logreg_word2vec":
            param_grid = word2vec_param_grid
        else:
            raise ValueError(f"Unknown model_family: {model_family}")


        results = []
        best_score = -1.0
        best_model = None
        best_params = None

        # Run through hyperparameter grid
        for params in ParameterGrid(param_grid):
            train_data, val_data, _ = self.prepare_features(params=params, config=config, train_df=train_df, val_df=val_df)
            X_train, y_train = (train_data[0], train_data[1])
            X_val, y_val = (val_data[0], val_data[1])

            model = self.initialize(params, config['seed'], config['model_family'])

            model.fit(X_train, y_train)
            
            val_proba = self.predict_proba(model, X_val)
            val_preds = (np.asarray(val_proba) >= threshold).astype(int)

            metrics = compute_metrics(val_preds, y_val)
            macro_f1 = metrics['macro_f1']

            results.append({**params, "val_score": macro_f1})

            if macro_f1 > best_score:
                best_score = macro_f1
                best_model = model
                best_params = dict(params)
                joblib.dump(model, model_path)

        if best_model is None:
            raise RuntimeError("Tuning failed: no valid parameter combination produced a trained model.")

        return best_model, results, best_params
    

    def predict_proba(self, model: Pipeline, X: Any) -> np.ndarray:

        if isinstance(X, tuple) and len(X) >= 1:
            X = X[0]
            
        if not isinstance(X, np.ndarray):
            X = X.astype(str).to_numpy()

        return model.predict_proba(X)[:, 1]  # numpy array (N,)