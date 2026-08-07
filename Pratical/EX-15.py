import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read Image
img = cv2.imread("C:/Users/chekk/Downloads/teddy bear.jpg")

# Convert to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Reshape pixels
Z = np.float32(img.reshape((-1,3)))

# K-Means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4
_, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Segmented Image
center = np.uint8(center)
result = center[label.flatten()]
result = result.reshape(img.shape)

# Display
plt.subplot(121)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(122)
plt.imshow(result)
plt.title("Segmented")
plt.axis("off")

plt.show()
