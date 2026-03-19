import numpy as np
from typing import Dict, Tuple, Any

from utils.helper import set_seeds
from evaluation.metrics import compute_metrics


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
        X_dev: Any, 
        y_dev: Any, 
        threshold: float=0.5
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Train the model; return tracked losses and best validation macro-F1"""
        
        set_seeds(config['seed'])
        
        n_samples, n_features = X_train.shape
        
        # Initialize parameters
        patience = 50
        bad = 0
        best_dev_macro_f1 = -1.0
        best_train_macro_f1 = -1.0
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        best_weights = self.weights.copy()
        best_bias = float(self.bias)
        best_epoch = 0
        train_losses = []
        dev_losses = []
        dev_macro_f1s = []
        steps = []

        # Train Loop
        for epoch in range(self.num_iter):
            #z = np.dot(X_train, self.weights) + self.bias     
            #y_proba = self._sigmoid(z)                   

            z = X_train @ self.weights              # Linear Prediction (z = wx + b)
            z = np.asarray(z).ravel() + self.bias
            y_proba = self._sigmoid(z)              # Compute Score

            error = np.asarray(y_proba).ravel() - np.asarray(y_train).ravel()   # (n_samples,)

            grad_likelihood = (X_train.T @ error) / n_samples                   # -> (n_features,)
            grad_likelihood = np.asarray(grad_likelihood).ravel()               # force 1D

            grad_reg = (self.lambda_reg / n_samples) * self.weights             # (n_features,)
            gradient = grad_likelihood + grad_reg    
            
            # Derivative of Loss for bias
            db = np.mean(error)

            # Update Parameters
            self.weights -= self.lr * gradient
            self.bias -= self.lr * db
            
            # Tracking Loss
            if epoch % 50 == 0:
                train_loss = self._compute_loss(y_train, y_proba)
                train_losses.append(train_loss)
                steps.append(epoch)

                # dev loss
                z_dev = X_dev @ self.weights
                z_dev = np.asarray(z_dev).ravel() + self.bias
                dev_proba = self._sigmoid(z_dev)
                dev_preds = (dev_proba >= threshold).astype(int)
                dev_loss = self._compute_loss(y_dev, dev_proba)
                dev_losses.append(dev_loss)

                print(f"Iteration {epoch}: train {train_loss:.6f} | dev {dev_loss:.6f}")
                
                metrics = compute_metrics(y_dev, dev_preds)
                dev_macro_f1 = metrics['macro_f1']
                dev_macro_f1s.append(dev_macro_f1)

                train_preds = (y_proba >= threshold).astype(int)
                train_metrics = compute_metrics(y_train, train_preds)
                train_macro_f1 = train_metrics["macro_f1"]

                if dev_macro_f1 > best_dev_macro_f1:
                    best_dev_macro_f1 = dev_macro_f1
                    best_train_macro_f1 = train_macro_f1
                    best_weights = self.weights.copy()
                    best_bias = float(self.bias)
                    best_epoch = epoch
                    bad = 0

                else:
                    bad += 1
                    if bad >= patience:
                        self.weights = best_weights
                        self.bias = best_bias
                        print(f"Early stopping at epoch {epoch} (best dev restored).")
                        break
        
        return best_dev_macro_f1, float(best_train_macro_f1), {
            "train_steps": steps,
            "train_loss": train_losses,
            "dev_steps": steps,
            "dev_loss": dev_losses,
            "dev_macro_f1": dev_macro_f1s,
            "best_step": best_epoch,
        }
    

    def predict_proba(self, X: Any) -> np.ndarray:
        """Computes predicted probabilities for the literal class"""
        #score = np.dot(X, self.weights) + self.bias
        #return self._sigmoid(score)
        score = X @ self.weights
        score = np.asarray(score).ravel() + self.bias
        return self._sigmoid(score)