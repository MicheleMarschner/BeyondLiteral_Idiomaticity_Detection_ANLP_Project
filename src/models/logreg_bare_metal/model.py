import numpy as np

from utils.helper import set_seeds, get_cols_from_df
from helper import build_vocab, buildw2i

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

    def train(self, config, X, y):
        """
        Trains the model using Gradient Descent.
        
        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,) containing 0 or 1
        """
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
        """
        Returns probability of positive class.
        """
        score = np.dot(X, self.weights) + self.bias
        return self._sigmoid(score)

    def predict(self, X, threshold=0.5):
        """
        Returns class labels (0 or 1).
        """
        y_pred_proba = self.predict_proba(X)
        return np.array([1 if i > threshold else 0 for i in y_pred_proba])


class LogRegRunner:
    def preprocessing(config, train_data, val_data, test_data):
        
        ## preprocessing such as lowercasing, punctuation etc.

        return train_data, val_data, test_data


    def prepare_inputs(self, train_df, val_df, test_df, config):
        "vectorize"

        inputs, labels, mwes = get_cols_from_df(train_df, ['Inputs', 'Label', 'Mwe'])
        
        vocab = build_vocab(inputs)
        w2i = buildw2i(vocab)
        
        label_set = sorted({y for _, y in train_data})
        label2i = {label: idx for idx, label in enumerate(label_set)}

        n = len(data)
        d = len(w2i)
        X = np.zeros((n, d), dtype=np.float32)
        Y = np.zeros(n, dtype=np.int64)

        for i, (tokens, label) in enumerate(data):
            for t in tokens:
                j = w2i.get(t)
                if j is not None:
                    X[i, j] += 1.0
            Y[i] = label2i[label]
        return X, Y


    def initialize(self, config, params) -> LogisticRegression:
        """
        Create the bare-metal LogisticRegression model from config.
        """

        model = LogisticRegression(
            learning_rate=params.get("learning_rate", 0.01),
            num_iterations=params.get("num_iterations", 1000),
            lambda_reg=params.get("lambda_reg", 0.0)
        )
        return model
    
    def fit():
        pass

    def predict_proba(self, model, X_test):
        return model.predict_proba(X_test)  # numpy array (N,)