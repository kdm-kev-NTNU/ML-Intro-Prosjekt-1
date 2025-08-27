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

        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
    def fit(self, X, y):
        self.weights = np.zeros(x.shape[1]) # x.shape = antall datapunkter, antall features
        self.bias = 0
        # Gradient Descent
        for _ in range(self.epochs): #Epochs: antall ganger treningsdataene sendes gjennom læringsalgoritmen. Nok en hyperparameter
            lin_model = np.matmul(self.weights, x.transpose()) + self.bias # her brukes det tranpose slik at matrimultiplikasjonen skal være på riktig form (m x n) * (n x p)
            y_pred = self.sigmoid_function(lin_model) #her begrenses de verdiene som lin modellen kan lage, mellom [0, 1] slik at det 
            grad_w, grad_b = self.compute_gradients(x, y, y_pred)
            self.update_parameters(grad_w, grad_b)

            loss = self._compute_loss(y, y_pred)  #her regnes ut loss - f.eks. via cross-entropi loss
            pred_to_class = [1 if _y > 0.5 else 0 for _y in y_pred] #her er klassifiseringsterskelen 0.5 er den større en det blir det 1 ellers 0. 
            self.train_accuracies.append(accuracy(y, pred_to_class))
            self.losses.append(loss)



        # TODO: Implement
        raise NotImplementedError("The fit method is not implemented yet.")

        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """ 
    def predict(self, X):

        




