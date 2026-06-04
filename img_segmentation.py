import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

image = cv2.imread("image.jpg")

if image is None:
    print("Image not found!")
    exit()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(pixel_values)

centers = np.uint8(kmeans.cluster_centers_)

segmented_data = centers[labels.flatten()]
segmented_image = segmented_data.reshape(image.shape)

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

cv2.imwrite(
    "segmented_output.jpg",
    cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
)

print("Segmented image saved as segmented_output.jpg")
