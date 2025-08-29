import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    """
    En enkel lineær regresjon implementasjon for kontinuerlige verdier.
    Bruker Mean Squared Error loss og gradient descent.
    """

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None  # (n_features,)
        self.bias = None     # skalar
        self.losses = []
        self.X_mean = None   # For normalisering
        self.X_std = None    # For normalisering

    def _compute_loss(self, y, y_pred):
        """Mean Squared Error loss for regresjon"""
        m = len(y)
        # Bruk np.clip for å unngå overflow
        diff = np.clip(y - y_pred, -1e10, 1e10)
        return np.mean(diff ** 2)

    def compute_gradients(self, X, y, y_pred):
        """Gradienter for Mean Squared Error loss"""
        m = X.shape[0]
        error = np.clip(y_pred - y, -1e10, 1e10)  # Clip error for å unngå overflow
        grad_w = (X.T @ error) / m       # (n_features,)
        grad_b = np.sum(error) / m       # skalar
        return grad_w, grad_b

    def update_parameters(self, grad_w, grad_b):
        """Oppdaterer vekter og bias med gradient descent"""
        # Clip gradienter for å unngå overflow
        grad_w = np.clip(grad_w, -1e10, 1e10)
        grad_b = np.clip(grad_b, -1e10, 1e10)
        
        self.weights -= self.learning_rate * grad_w
        self.bias -= self.learning_rate * grad_b

    def normalize_data(self, X):
        """Normaliserer dataene for bedre numerisk stabilitet"""
        if self.X_mean is None:
            self.X_mean = np.mean(X, axis=0)
            self.X_std = np.std(X, axis=0)
            # Unngå divisjon med 0
            self.X_std = np.where(self.X_std == 0, 1, self.X_std)
        
        return (X - self.X_mean) / self.X_std

    def fit(self, X, y):
        """
        Trener modellen på dataene
        X: (m, n) matrise med features
        y: (m,) vektor med target verdier
        """
        # Konverter til numpy arrays
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        
        # Hvis X er 1D, gjør den til 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        m, n = X.shape
        if y.shape[0] != m:
            raise ValueError("X og y må ha samme antall rader.")

        # Normaliser dataene
        X_normalized = self.normalize_data(X)

        # Initialiser parametere
        self.weights = np.zeros(n, dtype=float)
        self.bias = 0.0
        self.losses.clear()

        # Gradient descent
        for _ in range(self.epochs):
            # 1) Lineær prediksjon (ingen sigmoid for regresjon)
            y_pred = X_normalized @ self.weights + self.bias

            # 2) Beregn loss
            loss = self._compute_loss(y, y_pred)
            self.losses.append(loss)

            # 3) Beregn gradienter
            grad_w, grad_b = self.compute_gradients(X_normalized, y, y_pred)

            # 4) Oppdater parametere
            self.update_parameters(grad_w, grad_b)

    def predict(self, X):
        """Gjør prediksjoner på nye data"""
        if self.weights is None or self.bias is None:
            raise RuntimeError("Kall fit(X, y) før predict.")
        
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Normaliser testdata med samme parametere som treningsdata
        X_normalized = (X - self.X_mean) / self.X_std
            
        return X_normalized @ self.weights + self.bias

    def plot_training_progress(self):
        """Plotter MSE over epochs for å vise treningsprosessen"""
        if not self.losses:
            print("Ingen treningsdata tilgjengelig. Kjør fit() først.")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot MSE over epochs
        epochs = range(1, len(self.losses) + 1)
        plt.plot(epochs, self.losses, 'b-', linewidth=2, label='MSE')
        
        # Legg til grid og labels
        plt.grid(True, alpha=0.3)
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
        plt.title('Treningsprosess: MSE over tid', fontsize=14)
        plt.legend()
        
        # Vis start- og sluttverdier
        plt.text(0.02, 0.98, f'Start MSE: {self.losses[0]:.4f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        plt.text(0.02, 0.92, f'Slutt MSE: {self.losses[-1]:.4f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # Print statistikk
        print(f"Treningsstatistikk:")
        print(f"Start MSE: {self.losses[0]:.6f}")
        print(f"Slutt MSE: {self.losses[-1]:.6f}")
        print(f"Reduksjon: {((self.losses[0] - self.losses[-1]) / self.losses[0] * 100):.2f}%")
        print(f"Antall epochs: {len(self.losses)}")
