
from PIL import Image
import numpy as np

src = np.array(Image.open("mars_dark.tif").convert("L"))
ref = np.array(Image.open("mars_ref.tif").convert("L"))

def hist_match(src, ref):
    src_hist, _ = np.histogram(src.flatten(), 256, [0, 256])
    ref_hist, _ = np.histogram(ref.flatten(), 256, [0, 256])
    
    src_cdf = np.cumsum(src_hist) / src.size
    ref_cdf = np.cumsum(ref_hist) / ref.size
    
    lut = np.interp(src_cdf, ref_cdf, np.arange(256))
    
    return lut[src].astype(np.uint8)

out = hist_match(src, ref)

Image.fromarray(out).show()


