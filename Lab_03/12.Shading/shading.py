import cv2
import numpy as np

# Read image in grayscale
img = cv2.imread("hickson.tif", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Image not found!")
    exit()

# (b) Gaussian lowpass filtering
# Kernel size and sigma chosen similar to book example
blur = cv2.GaussianBlur(img, (31, 31), 5)

# Convert to float and scale to [0,1]
blur_norm = blur.astype(np.float32) / 255.0

# (c) Thresholding (choose threshold between 0 and 1)
T = 0.4
binary = np.zeros_like(blur_norm)
binary[blur_norm >= T] = 1.0

# Convert for display (0 or 255)
binary_disp = (binary * 255).astype(np.uint8)

# Show results
cv2.imshow("Original (a)", img)
cv2.imshow("Gaussian Lowpass (b)", blur)
cv2.imshow("Thresholded (c)", binary_disp)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Save outputs
cv2.imwrite("gaussian_filtered.jpg", blur)
cv2.imwrite("thresholded.jpg", binary_disp)
