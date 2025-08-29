import numpy as np

class LinearRegression():
    
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.losses = []
        self.train_accuracies = []

    def sigmoid_function(self, x):
        """Konverterer lineær output til sannsynlighet mellom 0 og 1"""
        return 1 / (1 + np.exp(-x))

    def compute_loss(self, y_true, y_pred):
        """Regner ut hvor mye modellen tar feil (binary cross entropy loss)"""
        # Unngå log(0) ved å begrense verdiene
        y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        
        m = len(y_true)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def compute_gradients(self, X, y_true, y_pred):
        """Regner ut gradientene for å oppdatere vekter og bias"""
        m = len(y_true)
        error = y_pred - y_true
        
        # Gradient for vekter
        grad_weights = (X.T @ error) / m
        # Gradient for bias
        grad_bias = np.mean(error)
        
        return grad_weights, grad_bias

    def update_parameters(self, grad_weights, grad_bias):
        """Oppdaterer vekter og bias med gradientene"""
        self.weights -= self.learning_rate * grad_weights
        self.bias -= self.learning_rate * grad_bias

    def accuracy(self, y_true, y_pred):
        """Regner ut hvor mange prosent av prediksjonene som er riktige"""
        return np.mean(y_true == y_pred)

    def fit(self, X, y):
        """Trener modellen på dataene"""
        X_poly = np.column_stack([
            X,                    # Original features
            X[:, 0] * X[:, 1],   # x0 * x1 (interaksjon)
            X[:, 0]**2,          # x0^2 (kvadratisk)
            X[:, 1]**2,          # x1^2 (kvadratisk)
            np.abs(X[:, 0]),     # |x0|
            np.abs(X[:, 1])      # |x1|
        ])
        
        # Initialiser vekter og bias
        self.weights = np.zeros(X_poly.shape[1])
        self.bias = 0
        
        # Tren modellen i flere epoker
        for epoch in range(self.epochs):
            # 1. Gjør prediksjoner
            linear_output = X_poly @ self.weights + self.bias
            y_pred = self.sigmoid_function(linear_output)
            
            # 2. Regn ut gradientene
            grad_weights, grad_bias = self.compute_gradients(X_poly, y, y_pred)
            
            # 3. Oppdater parametere
            self.update_parameters(grad_weights, grad_bias)
            
            # 4. Lagre loss og accuracy for plotting
            if epoch % 50 == 0:
                loss = self.compute_loss(y, y_pred)
                predictions = (y_pred > 0.5).astype(int)
                acc = self.accuracy(y, predictions)
                
                self.losses.append(loss)
                self.train_accuracies.append(acc)

    def predict(self, X):
        """Gjør prediksjoner på nye data"""
        # Bruk samme feature engineering som i training
        X_poly = np.column_stack([
            X,                    # Original features
            X[:, 0] * X[:, 1],   # x0 * x1 (interaksjon)
            X[:, 0]**2,          # x0^2 (kvadratisk)
            X[:, 1]**2,          # x1^2 (kvadratisk)
            np.abs(X[:, 0]),     # |x0|
            np.abs(X[:, 1])      # |x1|
        ])
        
        linear_output = X_poly @ self.weights + self.bias
        y_pred = self.sigmoid_function(linear_output)
        return (y_pred > 0.5).astype(int)