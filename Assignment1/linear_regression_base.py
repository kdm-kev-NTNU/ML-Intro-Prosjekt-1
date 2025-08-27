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


    #Regner ut loss for alle datapunktene og så finner den gjennomsnittet av det

    """
    y       = [1, 0, 1]
    y_pred  = [0.9, 0.3, 0.2]   # sannsynligheter fra sigmoid

    """
    def _compute_loss(self, y, y_pred):
        # Vi setter en veldig liten verdi 'eps' for å unngå log(0),
        # som ellers ville gitt -inf og ødelagt beregningene
        eps = 1e-12                      
        
        # Sørg for at sannsynlighetene y_pred aldri blir nøyaktig 0 eller 1.
        # np.clip setter en nedre grense på eps og en øvre grense på 1 - eps.
        y_pred = np.clip(y_pred, eps, 1 - eps)
        
        # Antall datapunkter (m = number of samples)
        m = y.shape[0]
        
        # Binær kryssentropi / negativ log-likelihood:
        # Formelen er:
        # L = -1/m * Σ [ y_i * log(y_pred_i) + (1 - y_i) * log(1 - y_pred_i) ]
        # alt etter sigma = formel 2 i word, mens alt i alt = formel 3 i word. 
        # (-1/m ) - gjør det slik at man finner det for gjennomsnittet av alle datapunkter
        #
        # Intuisjon:
        #  - Når y=1 → bidraget blir -log(y_pred)
        #  - Når y=0 → bidraget blir -log(1 - y_pred)
        # Dette straffer modellen når sannsynligheten for riktig klasse er lav.
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

        grad_w = (X.T @ error) / m      # gradient for vekter
        grad_b = np.sum(error) / m      # gradient for bias
        return grad_w, grad_b
    

    def update_parameters(self, grad_w, grad_b):
        """
        Oppdaterer vekter og bias med gradient descent.
        Vi flytter oss "motsatt" av gradienten for å minske loss.
        """
        self.weights -= self.learning_rate * grad_w
        self.bias    -= self.learning_rate * grad_b


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


        #Her så er gradient bergning før loss, fordi man har allerede formelen på forhånd, men matematisk så utledes gradienter basert på tapsfunksjonen. 

        # Gradient Descent
        for _ in range(self.epochs): #Epochs: antall ganger treningsdataene sendes gjennom læringsalgoritmen. Nok en hyperparameter
            lin_model = np.matmul(self.weights, X.transpose()) + self.bias # her brukes det tranpose slik at matrimultiplikasjonen skal være på riktig form (m x n) * (n x p)
            #Det er også her hvor lin.reg blir implementert
            y_pred = self.sigmoid_function(lin_model) #her forvandles de verdiene som lin modellen kan lage, mellom [0, 1] slik at det samsvar med binærklassifisering
            grad_w, grad_b = self.compute_gradients(X, y, y_pred)  #Her regner man ut gradienter. 
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

        




