from sklearn.linear_model import SGDRegressor
import numpy as np

# Dataset
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8]])
y = np.array([2,4,6,8,10,12,14,16])

# Mini-Batch Training
model = SGDRegressor(max_iter=1000, random_state=1)

batch_size = 2

for i in range(0, len(X), batch_size):
    model.partial_fit(X[i:i+batch_size], y[i:i+batch_size])

# Prediction
pred = model.predict([[9]])

print("Prediction =", round(pred[0],2))
