import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# Circular data
np.random.seed(1)
X = np.random.randn(200, 2)
y = (X[:, 0]**2 + X[:, 1]**2 > 1).astype(int)

# Neural Network
model = MLPClassifier(
    hidden_layer_sizes=(3, 3),
    activation='identity',
    learning_rate_init=0.03,
    max_iter=500,
    random_state=1
)

model.fit(X, y)

print("Accuracy:", model.score(X, y))

# Plot
plt.scatter(X[:,0], X[:,1], c=y)
plt.title("Circular Data Classification")
plt.show()
