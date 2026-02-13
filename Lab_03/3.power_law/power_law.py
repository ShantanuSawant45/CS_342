from PIL import Image
import numpy as np

# Read grayscale image
img = np.array(Image.open("fig0307(a).tif").convert("L"))
img = np.array(Image.open("fig0307(a)).tif").convert("L"))
# Normalize to [0,1]
r = img / 255.0

# Power-law (gamma) transformation
gamma = 2.5
s = r ** gamma

# Scale back to [0,255]
s = (s * 255).astype(np.uint8)

Image.fromarray(s).show()
