from sklearn.linear_model import SGDRegressor
import numpy as np

X = np.array([[1],[2],[3],[4],[5]])
y = np.array([2,4,6,8,10])

model = SGDRegressor(max_iter=1000, random_state=1)
model.fit(X, y)

pred = model.predict([[6]])

print("Prediction =", round(pred[0],2))
