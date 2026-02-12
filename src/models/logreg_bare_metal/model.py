import numpy as np
import joblib
from sklearn.model_selection import ParameterGrid

from typing import Dict, Tuple, Any
import pandas as pd
from tqdm import tqdm

from src.utils.helper import set_seeds
from src.models.logreg_bare_metal.featurize import build_featurizer
from src.evaluation import compute_metrics
from src.models.logreg_bare_metal.param_grid import tfidf_param_grid, word2vec_param_grid


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, lambda_reg=0.0):
        """
        Args:
            learning_rate (float): Step size for gradient descent.
            num_iterations (int): Number of epochs.
            lambda_reg (float): L2 regularization strength (0.0 = no regularization).
        """
        self.lr = learning_rate
        self.num_iter = num_iterations
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
    

    def _sigmoid(self, z):
        """
        Sigmoid function: calculates probabilities 
        """
        z = np.clip(z, -250, 250)               # Clip values to avoid overflow in exp
        return 1.0 / (1.0 + np.exp(-z))

    def _compute_loss(self, y, y_pred):
        """
        Binary Cross Entropy Loss with L2 Regularization.
        """
        m = len(y)
        epsilon = 1e-15                          # Add epsilon to prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Standard Log Loss: Cross Entropy
        loss = - (1 / m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        
        # Add L2 Regularization term (excluding bias)
        reg = (self.lambda_reg / (2 * m)) * np.sum(np.square(self.weights))
        
        return loss + reg

    def fit(self, config: Dict[str, Any], X_train, y_train, X_val, y_val, threshold=0.5):
        """Trains the weights of the model using Gradient Descent"""
        set_seeds(config['seed'])
        
        n_samples, n_features = X_train.shape
        
        # Initialize parameters
        patience = 5
        bad = 0
        best_f1 = -1.0
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        best_weights = self.weights.copy()
        best_bias = float(self.bias)
        train_losses = []
        val_losses = []

        # Train Loop
        for epoch in tqdm(range(self.num_iter)):
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

            print("bias", self.bias)
            
            # Tracking Loss
            if epoch % 100 == 0:
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
                    bad = 0

                else:
                    bad += 1
                    if bad >= patience:
                        self.weights = best_weights
                        self.bias = best_bias
                        print(f"Early stopping at epoch {epoch} (best val restored).")
                        break
        
        return train_losses, val_losses, best_f1
    
    def predict_proba(self, X):
        """Returns probability of positive class"""
        #score = np.dot(X, self.weights) + self.bias
        #return self._sigmoid(score)
        score = X @ self.weights
        score = np.asarray(score).ravel() + self.bias
        return self._sigmoid(score)


class LogRegRunner:

    def prepare_features(
        self, 
        params: Dict[str, Any], 
        config: Dict[str, Any], 
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        "featurize"
        featurizer = build_featurizer(config['model_family'], params, config['seed'])
        
        X_train, y_train = train_df['Input'], train_df['Label'].astype(int)
        X_test, y_test   = test_df['Input'],  test_df['Label'].astype(int)

        X_train = featurizer.fit_transform(X_train)
        X_test  = featurizer.transform(X_test)

        return (X_train, y_train), (X_test, y_test), featurizer


    def initialize(self, params, seed, model_family) -> LogisticRegression:
        """Initialize an instance of the LogisticRegression model with the hyperparameters from config"""

        model = LogisticRegression(
            learning_rate=params.get("learning_rate", 0.01),
            num_iterations=params.get("num_iterations", 1000),
            lambda_reg=params.get("lambda_reg", 0.0)
        ) 
        return model
    
    
    def tune(self, config, model_path, train_df, val_df, threshold=0.5):
        
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

        # Run through hyperparameter grid
        for params in ParameterGrid(param_grid):
            train_data, val_data, featurizer = self.prepare_features(params=params, config=config, train_df=train_df, test_df=val_df)
            X_train, y_train = (train_data[0], train_data[1])
            X_val, y_val = (val_data[0], val_data[1])
            
            model = self.initialize(params, config['seed'], config['model_family'])

            train_losses, val_losses, best_val_f1 = model.fit(config, X_train, y_train, X_val, y_val)

            results.append({**params, "val_macro_f1": best_val_f1})

            if best_val_f1 > best_f1:
                best_f1 = best_val_f1
                best_model = model
                best_params = dict(params)
                best_featurizer = featurizer
            
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

        return best_model, results, best_params


    def predict_proba(self, model, X) -> np.ndarray:
        if isinstance(X, tuple) and len(X) >= 1:
            X = X[0]

        return model.predict_proba(X)