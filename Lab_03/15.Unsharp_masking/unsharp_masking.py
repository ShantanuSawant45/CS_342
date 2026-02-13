import cv2
import numpy as np

# Read original image in grayscale
img = cv2.imread("dipxe.tif", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Image not found!")
    exit()

# (b) Gaussian blur: 31x31 kernel, sigma = 5
blur = cv2.GaussianBlur(img, (31, 31), 5)

# (c) Mask = Original - Blurred
mask = img.astype(np.float32) - blur.astype(np.float32)

# (d) Unsharp masking (k = 1)
k1 = 1
unsharp = img.astype(np.float32) + k1 * mask

# (e) Highboost filtering (k = 4.5)
k2 = 4.5
highboost = img.astype(np.float32) + k2 * mask

# Clip values to [0,255]
unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)
highboost = np.clip(highboost, 0, 255).astype(np.uint8)

# Show all results
cv2.imshow("Original (a)", img)
cv2.imshow("Blurred (b)", blur)
cv2.imshow("Mask (c)", cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
cv2.imshow("Unsharp Masking k=1 (d)", unsharp)
cv2.imshow("Highboost k=4.5 (e)", highboost)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Save outputs
cv2.imwrite("blurred.jpg", blur)
cv2.imwrite("mask.jpg", cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX))
cv2.imwrite("unsharp.jpg", unsharp)
cv2.imwrite("highboost.jpg", highboost)
