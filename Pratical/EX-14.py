import numpy as np
import matplotlib.pyplot as plt

# Data
X = np.array([1,2,3,4,5])
Y = np.array([2,4,5,4,5])

# Initialize
m, b = 0, 0
lr = 0.01
n = len(X)

# Gradient Descent
for i in range(1000):
    Y_pred = m*X + b
    dm = (-2/n) * np.sum(X * (Y - Y_pred))
    db = (-2/n) * np.sum(Y - Y_pred)
    m -= lr * dm
    b -= lr * db

print("Slope =", round(m,2))
print("Intercept =", round(b,2))

# Plot
plt.scatter(X, Y, color="blue")
plt.plot(X, m*X+b, color="red")
plt.title("Linear Regression using Gradient Descent")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
