from PIL import Image
import numpy as np
img = np.array(Image.open("Fig0310(b).tif").convert("L"))
r1, r2 = 70, 180
s1, s2 = 0, 255
# Piecewise linear contrast stretching
stretched = np.zeros_like(img)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        r = img[i, j]
        if r < r1:
            stretched[i, j] = (s1 / r1) * r
        elif r <= r2:
            stretched[i, j] = ((s2 - s1) / (r2 - r1)) * (r - r1) + s1
        else:
            stretched[i, j] = ((255 - s2) / (255 - r2)) * (r - r2) + s2
Image.fromarray(stretched.astype(np.uint8)).show()
