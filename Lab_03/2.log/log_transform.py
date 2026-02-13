from PIL import Image
import numpy as np

# Read grayscale image
img = np.array(Image.open("fig0305(a).tif").convert("L"))

# Fourier Transform
F = np.fft.fft2(img)
F_shift = np.fft.fftshift(F)

# Magnitude spectrum
magnitude = np.abs(F_shift)

# Log transformation (Eq. 3-4)
log_spectrum = np.log(1 + magnitude)

# Scale to [0, 255]
log_spectrum = 255 * log_spectrum / np.max(log_spectrum)

Image.fromarray(log_spectrum.astype(np.uint8)).show()
