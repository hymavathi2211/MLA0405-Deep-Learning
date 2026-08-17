import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read Image
img = cv2.imread(r"C:/Users/chekk/Downloads/dog.jpg")

# Convert to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Thresholding
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Morphological Closing
kernel = np.ones((3,3), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Display
plt.subplot(131)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(132)
plt.imshow(thresh, cmap="gray")
plt.title("Threshold")
plt.axis("off")

plt.subplot(133)
plt.imshow(closing, cmap="gray")
plt.title("Segmented")
plt.axis("off")

plt.show()
