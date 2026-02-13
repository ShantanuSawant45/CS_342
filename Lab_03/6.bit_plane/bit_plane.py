from PIL import Image
import numpy as np
img = np.array(Image.open("dollar.tif").convert("L"))

# Select bit plane (0 = LSB, 7 = MSB)
bit_plane = 7

# Extract bit plane
sliced = (img >> bit_plane) & 1

# Scale to [0, 255] for display
sliced = sliced * 255
Image.fromarray(sliced.astype(np.uint8)).show()
