import numpy as np

class LinearRegression():
    
    def __init__(self, learning_rate=0.1, epochs=100):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights, self.bias = None, None #Dette er ikke en tuple, det er bare en mer lettvint måte å skrive attributter på
        self.losses, self.train_accuracies = [], []


    def sigmoid_function(self, x):
        pass

    def _compute_loss(self, y, y_pred):
        pass

    def compute_gradients(self, x, y, y_pred):
        pass

    def update_parameters(self, grad_w, grad_b):
        pass

    def accuracy(true_values, predictions):
        return np.mean(true_values == predictions)

        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        # TODO: Implement
        raise NotImplementedError("The fit method is not implemented yet.")
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        # TODO: Implement
        raise NotImplementedError("The predict method is not implemented yet.")





