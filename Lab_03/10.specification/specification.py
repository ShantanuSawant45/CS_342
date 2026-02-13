from PIL import Image
import numpy as np
img = np.array(Image.open("input.tif").convert("L"))
mean = np.mean(img)
std = np.std(img)
desired_mean = 128
desired_std = 64
# Histogram statistics enhancement
enhanced = (img - mean) * (desired_std / std) + desired_mean
enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
Image.fromarray(enhanced).show()