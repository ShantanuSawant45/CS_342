from PIL import Image
import numpy as np
img = np.array(Image.open("skeleton.tif").convert("L"))
# 1) Smoothing (Lowpass: 3x3 average)
lp_kernel = np.ones((3,3)) / 9
pad = np.pad(img, 1, mode='edge')
smooth = np.zeros_like(img)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        smooth[i,j] = np.sum(pad[i:i+3, j:j+3] * lp_kernel)
# 2) Laplacian (High-frequency components)
lap_kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])
pad2 = np.pad(smooth, 1, mode='edge')
lap = np.zeros_like(img)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        lap[i,j] = np.sum(pad2[i:i+3, j:j+3] * lap_kernel)
# 3) Combine (Sharpening)
enhanced = smooth + lap
enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
Image.fromarray(enhanced).show()
