import numpy as np
import joblib
from sklearn.model_selection import ParameterGrid
from pathlib import Path

from typing import Dict, Tuple, Any
import pandas as pd

from src.utils.helper import set_seeds
from src.models.logreg_bare_metal.featurize import build_featurizer
from src.evaluation import compute_metrics
from src.models.logreg_bare_metal.param_grid import tfidf_param_grid, word2vec_param_grid


class LogisticRegression:
    def __init__(self, 
        learning_rate: float=0.01, 
        num_iterations: int=1000, 
        lambda_reg: float=0.0
    ) -> None:
        """Binary logistic regression trained with gradient descent and L2 regularization"""
        
        self.lr = learning_rate
        self.num_iter = num_iterations
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
    

    def _sigmoid(self, z: Any) -> np.ndarray:
        """Converts scores into probabilities (binary classification)"""
        z = np.clip(z, -250, 250)               # Clip values to avoid overflow in exp
        return 1.0 / (1.0 + np.exp(-z))

    def _compute_loss(self, y: Any, y_pred: Any) -> float:
        """Compute binary cross-entropy loss with L2 Regularization"""
        m = len(y)
        epsilon = 1e-15                          # Add epsilon to prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Standard Log Loss: Cross Entropy
        loss = - (1 / m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        
        # Add L2 Regularization term (excluding bias)
        reg = (self.lambda_reg / (2 * m)) * np.sum(np.square(self.weights))
        
        return loss + reg

    def fit(self, 
        config: Dict[str, Any], 
        X_train: Any, 
        y_train: Any, 
        X_val: Any, 
        y_val: Any, 
        threshold: float=0.5
    ) -> Tuple[list[float], list[float], float]:
        """Train the model; return tracked losses and best validation macro-F1"""
        
        set_seeds(config['seed'])
        
        n_samples, n_features = X_train.shape
        
        # Initialize parameters
        patience = 50
        bad = 0
        best_f1 = -1.0
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        best_weights = self.weights.copy()
        best_bias = float(self.bias)
        best_epoch = 0
        train_losses = []
        val_losses = []

        # Train Loop
        for epoch in range(self.num_iter):
            #z = np.dot(X_train, self.weights) + self.bias     
            #y_proba = self._sigmoid(z)                   

            z = X_train @ self.weights              # Linear Prediction (z = wx + b)
            z = np.asarray(z).ravel() + self.bias
            y_proba = self._sigmoid(z)              # Compute Score

            # Gradient of negative log-likelihood (averaged)
            #grad_likelihood = np.dot(X_train.T, (y_proba - y_train)) / n_samples 
            #grad_reg = (self.lambda_reg / n_samples) * self.weights
            #gradient = grad_likelihood + grad_reg

            error = np.asarray(y_proba).ravel() - np.asarray(y_train).ravel()   # (n_samples,)

            grad_likelihood = (X_train.T @ error) / n_samples                   # -> (n_features,)
            grad_likelihood = np.asarray(grad_likelihood).ravel()               # force 1D

            grad_reg = (self.lambda_reg / n_samples) * self.weights             # (n_features,)
            gradient = grad_likelihood + grad_reg    
            
            # Derivative of Loss for bias
            #db = np.sum(y_proba - y_train) / n_samples
            db = np.mean(error)

            # Update Parameters
            self.weights -= self.lr * gradient
            self.bias -= self.lr * db
            
            # Tracking Loss
            if epoch % 50 == 0:
                train_loss = self._compute_loss(y_train, y_proba)
                train_losses.append(train_loss)

                # val loss
                #z_val = np.dot(X_val, self.weights) + self.bias
                #val_proba = self._sigmoid(z_val)
                z_val = X_val @ self.weights
                z_val = np.asarray(z_val).ravel() + self.bias
                val_proba = self._sigmoid(z_val)
                val_preds = (val_proba >= threshold).astype(int)
                val_loss = self._compute_loss(y_val, val_proba)
                val_losses.append(val_loss)

                print(f"Iteration {epoch}: train {train_loss:.6f} | val {val_loss:.6f}")
                
                metrics = compute_metrics(val_preds, y_val)
                macro_f1 = metrics['macro_f1']

                if macro_f1 > best_f1:
                    best_f1 = macro_f1
                    best_weights = self.weights.copy()
                    best_bias = float(self.bias)
                    best_epoch = epoch
                    bad = 0

                else:
                    bad += 1
                    if bad >= patience:
                        self.weights = best_weights
                        self.bias = best_bias
                        print(f"Early stopping at epoch {epoch} (best val restored).")
                        break
        
        return best_f1, {
            "train_loss": train_losses,
            "val_loss": val_losses,
            "best_epoch": best_epoch,
        }
    

    def predict_proba(self, X: Any) -> np.ndarray:
        """Computes predicted probabilities for the positive class"""
        #score = np.dot(X, self.weights) + self.bias
        #return self._sigmoid(score)
        score = X @ self.weights
        score = np.asarray(score).ravel() + self.bias
        return self._sigmoid(score)


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
        
        X_train, y_train = train_df['Input'], train_df['Label'].astype(int)
        X_test, y_test   = test_df['Input'],  test_df['Label'].astype(int)

        X_train = featurizer.fit_transform(X_train)
        X_test  = featurizer.transform(X_test)

        print("N_train:", X_train.shape[0])
        print("vocab size:", len(featurizer.vocab_))
        print("avg nnz/doc train:", X_train.nnz / X_train.shape[0])
        print("empty train docs:", (X_train.getnnz(axis=1) == 0).sum())
        print("empty val/test docs:", (X_test.getnnz(axis=1) == 0).sum())


        return (X_train, y_train), (X_test, y_test), featurizer


    def initialize(self, params: Dict[str, Any], seed: int, model_family: str) -> LogisticRegression:
        """Create a LogisticRegression instance from hyperparameters"""

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
        val_df: pd.DataFrame,
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

        results = []
        bundle = {}
        best_f1 = -1.0
        best_model = None
        best_params = None
        best_featurizer = None
        best_curves = None

        # Run through hyperparameter grid
        for params in ParameterGrid(param_grid):
            train_data, val_data, featurizer = self.prepare_features(params=params, config=config, train_df=train_df, test_df=val_df)
            X_train, y_train = (train_data[0], train_data[1])
            X_val, y_val = (val_data[0], val_data[1])
            
            model = self.initialize(params, config['seed'], config['model_family'])

            best_val_f1, loss_curves = model.fit(config, X_train, y_train, X_val, y_val)

            results.append({**params, "val_macro_f1": best_val_f1})

            if best_val_f1 > best_f1:
                best_f1 = best_val_f1
                best_model = model
                best_params = dict(params)
                best_featurizer = featurizer
                best_curves = loss_curves
            
            if best_model is None:
                raise RuntimeError("Tuning failed: no valid parameter combination produced a trained model.")

        
        bundle = {
            "model": best_model,                 # contains weights+bias
            "featurizer": best_featurizer,       # crucial if you used TF-IDF / vocab
            "best_params": best_params,
            "threshold": threshold,
            "macro_f1": best_f1,
        }
        joblib.dump(bundle, model_path)

        return best_model, results, best_params, best_curves


    def predict_proba(self, model: LogisticRegression, X: Any) -> np.ndarray:
        """Wrapper to get positive-class probabilities from a model"""
        
        if isinstance(X, tuple) and len(X) >= 1:
            X = X[0]

        p = model.predict_proba(X)
        print("proba min/mean/max:", p.min(), p.mean(), p.max())
        print("positives @0.5:", (p >= 0.5).sum(), "/", len(p))

        return p