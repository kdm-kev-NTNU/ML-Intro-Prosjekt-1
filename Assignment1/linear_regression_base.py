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
        return 1 / (1 + np.exp(-x))

    def _compute_loss(self, y, y_pred):
        pass

    def compute_gradients(self, x, y, y_pred):
        pass

    def update_parameters(self, grad_w, grad_b):
        pass

    def accuracy(self, true_values, predictions):
        return np.mean(true_values == predictions)

        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features) - datapunkter 
            y (array<m>): a vector of floats - target 
        """
        #Tenk på X som datapunkter og Y som kolonnen av Targets
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1]) # X.shape = antall datapunkter, antall features
        self.bias = 0
        #Her har vi z = Sigma  Wi*Xi + b - som da er lin reg formelen. 
        #men fordi vi initialiserer alt i fit. så starter det som 0. 
        # Hvor X er featurene. og y er targeten

        # Gradient Descent
        for _ in range(self.epochs): #Epochs: antall ganger treningsdataene sendes gjennom læringsalgoritmen. Nok en hyperparameter
            lin_model = np.matmul(self.weights, X.transpose()) + self.bias # her brukes det tranpose slik at matrimultiplikasjonen skal være på riktig form (m x n) * (n x p)
            #Det er også her hvor lin.reg blir implementert
            y_pred = self.sigmoid_function(lin_model) #her forvandles de verdiene som lin modellen kan lage, mellom [0, 1] slik at det samsvar med binærklassifisering
            grad_w, grad_b = self.compute_gradients(X, y, y_pred)
            self.update_parameters(grad_w, grad_b)

            loss = self._compute_loss(y, y_pred)  #her regnes ut loss - f.eks. via cross-entropi loss
            pred_to_class = [1 if _y > 0.5 else 0 for _y in y_pred] #her er klassifiseringsterskelen 0.5 er den større en det blir det 1 ellers 0 - dette her er bare for å legge endringen av presisjon under treningen.
            # ikke for å faktisk predikere. 
            self.train_accuracies.append(self.accuracy(y, pred_to_class))
            self.losses.append(loss)


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
        lin_model = np.matmul(X, self.weights) + self.bias
        y_pred = self.sigmoid_function(lin_model)
        return [1 if _y > 0.5 else 0 for _y in y_pred]

        




