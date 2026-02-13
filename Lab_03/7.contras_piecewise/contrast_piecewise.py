import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("img06.tif", cv2.IMREAD_GRAYSCALE)

r_min, r_max = img.min(), img.max()

contrast_stretched = ((img - r_min) / (r_max - r_min)) * 255
contrast_stretched = contrast_stretched.astype(np.uint8)

_, thresholded = cv2.threshold(contrast_stretched, 120, 255, cv2.THRESH_BINARY)

plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.title("Input")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Contrast Stretched")
plt.imshow(contrast_stretched, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Thresholded")
plt.imshow(thresholded, cmap='gray')
plt.axis('off')

plt.show()

cv2.imwrite("fig3_10_b_input.png", img)
cv2.imwrite("fig3_10_c_contrast_stretched.png", contrast_stretched)
cv2.imwrite("fig3_10_d_thresholded.png", thresholded)
