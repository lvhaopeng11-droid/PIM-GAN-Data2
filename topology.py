import numpy as np
from skimage.morphology import skeletonize
from skimage.measure import euler_number

def compute_euler_characteristic(binary_mask):
    return float(euler_number(binary_mask, connectivity=2))

def compute_skeleton_length_um(binary_mask, pixel_scale_um=0.058):
    skel = skeletonize(binary_mask > 0)
    length = np.sum(skel) * pixel_scale_um
    total_area = binary_mask.shape[0] * binary_mask.shape[1] * (pixel_scale_um**2)
    return float(length / total_area)
