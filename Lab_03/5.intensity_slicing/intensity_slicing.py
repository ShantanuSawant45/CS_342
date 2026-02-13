from PIL import Image
import numpy as np
img = np.array(Image.open("kidney.tif").convert("L"))
# Define intensity range to highlight
lower, upper = 100, 180
# Intensity-level slicing
sliced = np.zeros_like(img)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if lower <= img[i, j] <= upper:
            sliced[i, j] = 255
        else:
            sliced[i, j] = 0
# Display output image
Image.fromarray(sliced).show()
