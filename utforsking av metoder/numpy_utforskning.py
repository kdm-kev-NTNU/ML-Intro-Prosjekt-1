import numpy as np

#np.mean eksempel
y_true = np.array([0, 1, 1, 0, 1])   # true labels
y_pred = np.array([0, 1, 0, 0, 1])   # predicted label

accuracy = np.mean(y_true == y_pred)
print(accuracy)  #betyr at det er 80% likhet mellom true labels og det som ble predicted



#np.matmul

#fungerer her som en dot product fordi det begge vektorene er i 1D. men det gjør kort sagt matrise multiplikasjon
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

result = np.matmul(a, b)
print(result)  # 32 -- 1*4 + 2*5 + 3*6 = 32



#np.zeros

# np.zeros(n) lager en vetkor med n nuller
# X.shape[1] = antall kolonner

"""
X = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [10.0, 11.0, 12.0]
])

"""

# Her kommer det et eksempel på disse prinsippene i bruk

# 4 samples, 3 features
X = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0],
    [10.0, 11.0, 12.0]
])

y = np.array([0, 1, 0, 1])  # labels

#Initiering av weights og bias
weights = np.zeros(X.shape[1])  # -> [0. 0. 0.]
bias = 0.0

"""

Her
- X.shape[1] = 3 - features
- Weights representerer hvor mye hver feature skal telle med i modellen
- bias er startpunktet intercept


"""

#Etter en oppdatering
grad_w = np.array([0.1, -0.2, 0.05])
grad_b = -0.1
learning_rate = 0.1

weights = weights - learning_rate * grad_w
bias = bias - learning_rate * grad_b

print(weights)  # [-0.01  0.02 -0.005]
print(bias)     # 0.01


