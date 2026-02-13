import cv2
import numpy as np

# Read image in grayscale
img = cv2.imread("contact_lens.tif", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Image not found!")
    exit()

# Apply Sobel in X and Y direction
sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

# Gradient magnitude
sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)

# Normalize to 0-255
sobel_mag = np.uint8(255 * sobel_mag / np.max(sobel_mag))

# Show results
cv2.imshow("Original", img)
cv2.imshow("Sobel Gradient", sobel_mag)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Save output
cv2.imwrite("sobel_output.jpg", sobel_mag)
