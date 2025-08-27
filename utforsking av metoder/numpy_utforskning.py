import numpy as np

y_true = np.array([0, 1, 1, 0, 1])   # true labels
y_pred = np.array([0, 1, 0, 0, 1])   # predicted label

accuracy = np.mean(y_true == y_pred)
print(accuracy)  #betyr at det er 80% likhet mellom true labels og det som ble predicted

