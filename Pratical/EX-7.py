import numpy as np
import matplotlib.pyplot as plt
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
x = np.arange(-5, 5, 0.1)
plt.plot(x, sigmoid(x), color='pink')
plt.title("Visualization of the Sigmoid Function")
plt.xlabel("x")
plt.ylabel("Sigmoid(x)")
plt.grid(True)

plt.show()
