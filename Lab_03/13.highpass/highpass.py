from PIL import Image
import numpy as np

# Read grayscale image
img = np.array(Image.open("input.tif").convert("L"))
# High-pass filter kernel
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

# Pad image
padded = np.pad(img, 1, mode='edge')
filtered = np.zeros_like(img)

# Apply high-pass filter
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        region = padded[i:i+3, j:j+3]
        filtered[i, j] = np.sum(region * kernel)

# Clip and display
filtered = np.clip(filtered, 0, 255).astype(np.uint8)
Image.fromarray(filtered).show()
