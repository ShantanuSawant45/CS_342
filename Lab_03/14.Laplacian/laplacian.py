import cv2
import numpy as np

# Read image in grayscale
img = cv2.imread("moon.tif", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Image not found!")
    exit()

# (a) Blur the image (simulate blurred moon)
blur = cv2.GaussianBlur(img, (5, 5), 1.0)

# Laplacian kernel Fig 3.45(a)
kernel1 = np.array([[0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]], dtype=np.float32)

# Laplacian kernel Fig 3.45(b)
kernel2 = np.array([[1, 1, 1],
                    [1, -8, 1],
                    [1, 1, 1]], dtype=np.float32)

# (b) Laplacian image using kernel1
laplacian1 = cv2.filter2D(blur, cv2.CV_32F, kernel1)

# (c) Sharpen using Eq (3-54) with c = -1  ->  g = f - c*laplacian
c = -1
sharpen1 = blur - c * laplacian1

# (d) Sharpen using kernel2
laplacian2 = cv2.filter2D(blur, cv2.CV_32F, kernel2)
sharpen2 = blur - c * laplacian2

# Convert to uint8
laplacian1 = cv2.normalize(laplacian1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
sharpen1 = np.clip(sharpen1, 0, 255).astype(np.uint8)
sharpen2 = np.clip(sharpen2, 0, 255).astype(np.uint8)

# Show results
cv2.imshow("Blurred (a)", blur)
cv2.imshow("Laplacian (b)", laplacian1)
cv2.imshow("Sharpened kernel1 (c)", sharpen1)
cv2.imshow("Sharpened kernel2 (d)", sharpen2)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Save outputs
cv2.imwrite("blurred.jpg", blur)
cv2.imwrite("laplacian.jpg", laplacian1)
cv2.imwrite("sharpen1.jpg", sharpen1)
cv2.imwrite("sharpen2.jpg", sharpen2)
