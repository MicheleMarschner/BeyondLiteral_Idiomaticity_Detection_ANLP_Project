import numpy as np
import joblib
from sklearn.model_selection import ParameterGrid

from src.utils.helper import set_seeds
from src.models.logreg_bare_metal.featurize import build_featurizer
from src.evaluation import compute_metrics
from src.models.logreg_bare_metal.param_grid import tfidf_param_grid, word2vec_param_grid


class BareMetalLogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, lambda_reg=0.0, verbose=False):
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

    def fit(self, config: Dict[str, Any], X, y):
        """Trains the weights of the model using Gradient Descent"""
        set_seeds(config.seed)
        
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        # Train Loop
        for epoch in range(self.num_iter):
            z = np.dot(X, self.weights) + self.bias     # Linear Prediction (z = wx + b)
            y_pred = self._sigmoid(z)                   # Compute Score

            # Gradient of negative log-likelihood (averaged)
            grad_likelihood = np.dot(X.T, (y_pred - y)) / n_samples 
            grad_reg = (self.lambda_reg / n_samples) * self.weights
            gradient = grad_likelihood + grad_reg
            
            # Derivative of Loss for bias
            db = np.sum(y_pred - y) / n_samples

            # Update Parameters
            self.weights -= self.lr * gradient
            self.bias -= self.lr * db
            
            # Tracking Loss
            if epoch % 100 == 0:
                current_loss = self._compute_loss(y, y_pred)
                self.loss_history.append(current_loss)
                print(f"Iteration {epoch}: Loss {current_loss:.4f}")

    
    def predict_proba(self, X):
        """Returns probability of positive class"""
        score = np.dot(X, self.weights) + self.bias
        return self._sigmoid(score)


class LogRegRunner:

    def prepare_features(self, params, config: Dict[str, Any], train_df=None, val_df=None, test_df=None):
        "featurize"
        X_train, y_train = train_df['Input'], train_df['Label'].astype(int)
        X_val, y_val     = val_df['Input'],   val_df['Label'].astype(int)
        X_test, y_test   = test_df['Input'],  test_df['Label'].astype(int)

        featurizer = build_featurizer(config['model_family'], params, config['seed'])
        X_train = featurizer.fit_transform(X_train)
        X_val   = featurizer.transform(X_val)
        X_test  = featurizer.transform(X_test)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


    def initialize(self, config: Dict[str, Any], params) -> BareMetalLogisticRegression:
        """Initialize an instance of the LogisticRegression model with the hyperparameters from config"""

        model = BareMetalLogisticRegression(
            learning_rate=params.get("learning_rate", 0.01),
            num_iterations=params.get("num_iterations", 1000),
            lambda_reg=params.get("lambda_reg", 0.0)
        )
        return model
    
    
    def fit(self, config, model_path, train_data, val_data, threshold=0.5):
        
        X_train, y_train = (train_data[0], train_data[1])
        X_val, y_val = (val_data[0], val_data[1])
        
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


    def predict_proba(self, model, X_test) -> np.ndarray:
        return model.predict_proba(X_test)  # numpy array (N,)