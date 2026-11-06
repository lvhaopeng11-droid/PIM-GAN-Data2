import numpy as np
from scipy.ndimage import sobel

def compute_characteristic_width_um(binary_mask, pixel_scale_um=0.058):
    area = np.sum(binary_mask)
    if area == 0:
        return 0.0
    dx = sobel(binary_mask.astype(float), axis=1)
    dy = sobel(binary_mask.astype(float), axis=0)
    peri = np.sum(np.sqrt(dx**2 + dy**2))
    if peri == 0:
        return 0.0
    w_px = (2.0 * area) / peri
    return float(w_px * pixel_scale_um)
