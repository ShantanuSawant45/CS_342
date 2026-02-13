from PIL import Image
import numpy as np
# Read grayscale image
img = np.array(Image.open("input.tif").convert("L"))
# Define 3x3 averaging (lowpass) filter
kernel = np.ones((3, 3)) / 9
# Pad image
padded = np.pad(img, 1, mode='edge')
filtered = np.zeros_like(img)
# Apply lowpass filter
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        region = padded[i:i+3, j:j+3]
        filtered[i, j] = np.sum(region * kernel)
Image.fromarray(filtered.astype(np.uint8)).show()
