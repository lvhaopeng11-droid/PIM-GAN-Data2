import numpy as np
from skimage.filters import threshold_otsu

def segment_phase_otsu(gray_img):
    img = np.asarray(gray_img, dtype=np.float32)
    if img.max() > 1.0:
        img /= 255.0
    val = threshold_otsu(img)
    mask = (img > val).astype(np.uint8)
    return mask, val

def calculate_training_tref(image_list):
    thresholds = []
    for img in image_list:
        _, val = segment_phase_otsu(img)
        thresholds.append(val)
    return float(np.median(thresholds)) if thresholds else 0.5
