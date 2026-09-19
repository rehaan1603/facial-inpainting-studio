import numpy as np
from scipy.ndimage import laplace

def quality(rgb, region):
    if not region.any():return {'sharpness':0.,'exposure':0.,'quality':0.}
    gray=np.asarray(rgb,dtype=float).mean(2)
    # Exclude patch boundaries: filtering happens on the original full frame.
    variance=float(laplace(gray)[region].var())
    sharpness=variance/(variance+100.)
    pixels=gray[region];exposure=float(np.mean((pixels>12)&(pixels<243)))
    return {'sharpness':sharpness,'exposure':exposure,'quality':.75*sharpness+.25*exposure}
