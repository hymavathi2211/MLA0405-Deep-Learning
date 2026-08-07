from sklearn.neural_network import MLPRegressor
import numpy as np

# Input data
X = np.array([[0],
              [1],
              [2],
              [3],
              [4],
              [5]])

# Output data
Y = np.array([0, 2, 4, 6, 8, 10])

# Create Neural Network with one hidden layer
model = MLPRegressor(hidden_layer_sizes=(5,),
                     activation='relu',
                     solver='adam',
                     max_iter=2000,
                     random_state=1)

# Train the model
model.fit(X, Y)

# Predict outputs
pred = model.predict(X)

print("Actual Output")
print(Y)

print("\nPredicted Output")
print(np.round(pred,2))

print("\nPrediction for Input = 6")
print(np.round(model.predict([[6]]),2))
