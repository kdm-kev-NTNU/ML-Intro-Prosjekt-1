import numpy as np

class LinearRegression():
    
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights, self.bias = None, None 
        self.losses, self.train_accuracies = [], []


    def sigmoid_function(self, x):
        return 1 / (1 + np.exp(-x))


    def _compute_loss(self, y, y_pred):
        eps = 1e-12                      
        

        y_pred = np.clip(y_pred, eps, 1 - eps)
        m = y.shape[0]
        return - (1.0 / m) * (
            np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        )
        


    def compute_gradients(self, X, y, y_pred):
        """
        Regner ut gradientene (hvor mye vi skal justere vekter og bias).
        X: (m, n) matrise med datapunkter
        y: (m,)   sanne labels (0 eller 1)
        y_pred:   sannsynligheter fra sigmoid

        Formelen er basert på utledningen av loss:
          grad_w = (1/m) * X^T · (y_pred - y)
          grad_b = (1/m) * sum(y_pred - y)
        """
        m = X.shape[0]                  # antall datapunkter
        error = y_pred - y              # forskjellen mellom prediksjon og fasit

        grad_w = (X.T @ error) / m      
        grad_b = np.sum(error) / m      
        return grad_w, grad_b
    

    def update_parameters(self, grad_w, grad_b):
        self.weights -= self.learning_rate * grad_w
        self.bias    -= self.learning_rate * grad_b


    def accuracy(self, true_values, predictions):
        return np.mean(true_values == predictions)


    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # Gradient Descent
        for _ in range(self.epochs):
            lin_model = np.matmul(self.weights, X.transpose()) + self.bias 
            y_pred = self.sigmoid_function(lin_model)
            grad_w, grad_b = self.compute_gradients(X, y, y_pred) 
            self.update_parameters(grad_w, grad_b)

            loss = self._compute_loss(y, y_pred)  
            pred_to_class = [1 if _y > 0.5 else 0 for _y in y_pred] 
            self.train_accuracies.append(self.accuracy(y, pred_to_class))
            self.losses.append(loss)


    
    def predict(self, X):
        lin_model = np.matmul(X, self.weights) + self.bias
        y_pred = self.sigmoid_function(lin_model)
        return [1 if _y > 0.5 else 0 for _y in y_pred]

        




