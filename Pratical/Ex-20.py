import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# Multi-class data
np.random.seed(1)
X = np.random.randn(150, 2)
y = np.array([0]*50 + [1]*50 + [2]*50)

# Neural Network
model = MLPClassifier(
    hidden_layer_sizes=(2, 2),
    activation='identity',
    learning_rate_init=0.01,
    max_iter=500,
    random_state=1
)

model.fit(X, y)

print("Accuracy:", model.score(X, y))

# Visualization
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("Multi-Class Neural Network")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
