import numpy as np

class LinearRegression:
    """
    En veldig enkel binær klassifikator med sigmoid og binær kryssentropi.
    (Navnet 'LinearRegression' beholdt for kompatibilitet.)
    """

    def __init__(self, learning_rate, epochs, threshold):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold
        self.weights = None  # (n_features,)
        self.bias = None     # skalar
        self.losses = []
        self.train_accuracies = []

    # --- Grunnleggende byggesteiner ---

    def sigmoid_function(self, x):
        # Vanlig sigmoid. Dette holder i praksis for moderate x.
        return 1.0 / (1.0 + np.exp(-x))

    def _compute_loss(self, y, y_pred):
        # Binær kryssentropi. Clip for å unngå log(0).
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1 - eps)
        m = len(y)
        return - (1.0 / m) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def compute_gradients(self, X, y, y_pred):
        # batch-gradienter for vekter og bias
        m = X.shape[0]
        error = y_pred - y               # (m,)
        grad_w = (X.T @ error) / m       # (n_features,)
        grad_b = np.sum(error) / m       # skalar
        return grad_w, grad_b

    def update_parameters(self, grad_w, grad_b):
        # enkel gradient descent
        self.weights -= self.learning_rate * grad_w
        self.bias    -= self.learning_rate * grad_b

    def accuracy(self, y_true, y_pred_labels):
        return float(np.mean(y_true == y_pred_labels))

    # --- Trening og prediksjon ---

    def fit(self, X, y):
        """
        X: (m, n) matrise
        y: (m,) vektor med 0/1
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        m, n = X.shape
        if y.shape[0] != m:
            raise ValueError("X og y må ha samme antall rader.")
        if not set(np.unique(y)).issubset({0.0, 1.0}):
            raise ValueError("y må være binær (0/1).")

        # init
        self.weights = np.zeros(n, dtype=float)
        self.bias = 0.0
        self.losses.clear()
        self.train_accuracies.clear()

        for _ in range(self.epochs):
            # 1) lineær kombinasjon + sigmoid
            lin = X @ self.weights + self.bias          # (m,)
            y_pred = self.sigmoid_function(lin)         # (m,)

            # 2) loss 
            loss = self._compute_loss(y, y_pred)
            self.losses.append(loss)

            # 3) gradienter
            grad_w, grad_b = self.compute_gradients(X, y, y_pred)

            # 4) oppdater parametre
            self.update_parameters(grad_w, grad_b)

            # enkel trenings-accuracy logging på nåværende prediksjoner
            preds = (y_pred >= self.threshold).astype(int)
            self.train_accuracies.append(self.accuracy(y.astype(int), preds))

    def predict_proba(self, X):
        if self.weights is None or self.bias is None:
            raise RuntimeError("Kall fit(X, y) før predict_proba.")
        X = np.asarray(X, dtype=float)
        lin = X @ self.weights + self.bias
        return self.sigmoid_function(lin)

    def predict(self, X):
        """Returnerer klassetagger {0,1} basert på terskel."""
        X = np.asarray(X, dtype=float)
        lin = X @ self.weights + self.bias
        y_pred = self.sigmoid_function(lin)
        return (y_pred >= self.threshold).astype(int)
