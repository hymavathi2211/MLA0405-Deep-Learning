import numpy as np 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
def true_fun(X):
    return np.cos(1.5 * np.pi * X)
np.random.seed(0)
X = np.sort(np.random.rand(30))
y = true_fun(X) + np.random.randn(30) * 0.1
degrees = [1, 4, 15]
plt.figure(figsize=(14, 5))
for i, degree in enumerate(degrees):
    plt.subplot(1, 3, i + 1)
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("lr", LinearRegression())
    ])
    model.fit(X[:, np.newaxis], y)
    scores = cross_val_score(
        model, X[:, np.newaxis], y,
        scoring="neg_mean_squared_error", cv=10
    )
    X_test = np.linspace(0, 1, 100)
    plt.plot(X_test, model.predict(X_test[:, np.newaxis]), label="Model")
    plt.plot(X_test, true_fun(X_test), label="True")
    plt.scatter(X, y, color="red", marker="^")
    plt.title(f"Degree {degree}\nMSE={-scores.mean():.2e}")
    plt.xlim(0, 1)
    plt.ylim(-2, 2)
    plt.legend()
plt.show()
