import numpy as np

def compute_phase_area_fraction_2d(binary_mask):
    return float(np.mean(binary_mask) * 100.0)
