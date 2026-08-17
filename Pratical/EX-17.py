import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# Create dataset
X, y = make_classification(n_samples=100, n_features=2,
                           n_informative=2, n_redundant=0,
                           random_state=42)

# Train model
model = LogisticRegression()
model.fit(X, y)

# Plot data
plt.scatter(X[:,0], X[:,1], c=y, cmap='coolwarm')

# Decision boundary
coef = model.coef_[0]
intercept = model.intercept_[0]
x = [X[:,0].min(), X[:,0].max()]
y_line = [-(coef[0]*i + intercept)/coef[1] for i in x]

plt.plot(x, y_line, 'k--')
plt.title("Linear Separability")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
