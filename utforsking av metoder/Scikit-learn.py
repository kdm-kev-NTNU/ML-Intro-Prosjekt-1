#Scikit-learn bibliotek 

#eksempel for effektiv implementering av lineær regresjon

from sklearn.linear_model import LinearRegression
import numpy as np

x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # må være 2D
y = np.array([2, 4, 5, 4, 5])

model = LinearRegression().fit(x, y)
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
