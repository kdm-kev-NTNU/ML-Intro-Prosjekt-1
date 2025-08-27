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