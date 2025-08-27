import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.losses = []

    def _compute_loss(self, y, y_pred):
        """
        MSE (Mean Squared Error) = (1/(2m)) * sum((y_pred - y)^2)
        Faktor 1/2 gjør at derivatet blir penere (2 kanselleres).
        """
        m = y.shape[0]
        error = y_pred - y
        return (1.0 / (2*m)) * np.sum(error ** 2)

    def compute_gradients(self, X, y, y_pred):
        """
        For lineær regresjon (y_hat = Xw + b) blir gradientene:
          grad_w = (1/m) * X^T · (y_pred - y)
          grad_b = (1/m) * sum(y_pred - y)
        """
        m = X.shape[0]
        error = (y_pred - y)               # (m,)
        grad_w = (X.T @ error) / m         # (n,)
        grad_b = np.sum(error) / m         # skalar
        return grad_w, grad_b

    def update_parameters(self, grad_w, grad_b):
        self.weights -= self.learning_rate * grad_w
        self.bias    -= self.learning_rate * grad_b

    def fit(self, X, y):
        """
        Tilpasser lineær modell på formen y_hat = Xw + b
        - X kan være 1D (Series/array) eller 2D (DataFrame/array)
        - y er kontinuerlig målvariabel (1D)
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)          # tving 2D: (m, 1)
        y = np.asarray(y).ravel()          # tving 1D: (m,)

        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0.0

        for _ in range(self.epochs):
            # Forward
            y_pred = X @ self.weights + self.bias   # lineær modell

            # Logg loss før oppdatering (konsistent)
            loss = self._compute_loss(y, y_pred)
            self.losses.append(loss)

            # Backward og oppdater
            grad_w, grad_b = self.compute_gradients(X, y, y_pred)
            self.update_parameters(grad_w, grad_b)

        return self

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y_pred = X @ self.weights + self.bias
        return y_pred.ravel()
