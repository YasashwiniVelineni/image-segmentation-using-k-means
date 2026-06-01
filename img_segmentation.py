import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load image
image = cv2.imread("image.jpg")

if image is None:
    print("Image not found!")
    exit()

# Convert BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape image into pixel array
pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Number of clusters
k = 4

# Apply K-Means
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(pixel_values)

# Get cluster centers
centers = np.uint8(kmeans.cluster_centers_)

# Recreate segmented image
segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape(image.shape)

# Display images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title(f"Segmented Image (K={k})")
plt.axis("off")

plt.show()

# Save result
cv2.imwrite(
    "segmented_output.jpg",
    cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
)

print("Segmented image saved as segmented_output.jpg")
