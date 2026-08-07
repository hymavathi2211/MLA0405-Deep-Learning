import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Actual labels
actual = np.array([
    'Dog', 'Dog', 'Dog', 'Not Dog', 'Dog',
    'Not Dog', 'Dog', 'Dog', 'Not Dog', 'Not Dog'
])

# Predicted labels
predicted = np.array([
    'Dog', 'Not Dog', 'Dog', 'Not Dog', 'Dog',
    'Dog', 'Dog', 'Dog', 'Not Dog', 'Not Dog'
])

# Create confusion matrix
conf_matrix = confusion_matrix(actual, predicted)

# Plot heatmap
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt='g',
    cmap='RdPu',
    xticklabels=['Dog', 'Not Dog'],
    yticklabels=['Dog', 'Not Dog']
)

plt.xlabel("Predicted", fontsize=16)
plt.ylabel("Actual", fontsize=16)
plt.title("Confusion Matrix", fontsize=20)

plt.show()
