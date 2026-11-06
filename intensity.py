import numpy as np

def normalize_zero_one_to_tanh(gray01):
    a = np.asarray(gray01, dtype=np.float32)
    return a * 2.0 - 1.0

def denormalize_tanh_to_zero_one(tanh_image):
    a = np.asarray(tanh_image, dtype=np.float32)
    return (a + 1.0) / 2.0
