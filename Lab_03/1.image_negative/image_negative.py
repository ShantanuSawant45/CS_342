from PIL import Image
import numpy as np

img = Image.open("img01.tif").convert("L")   # .tif file
negative = 255 - np.array(img)

Image.fromarray(negative).show()