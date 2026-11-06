import numpy as np
from PIL import Image

def extract_regular_crops(image_np, crop_size=425, n_cols=4, n_rows=3):
    H, W = image_np.shape[:2]
    crops = []
    coords = []
    for r in range(n_rows):
        for c in range(n_cols):
            x0 = c * crop_size
            y0 = r * crop_size
            if x0 + crop_size <= W and y0 + crop_size <= H:
                crop = image_np[y0:y0+crop_size, x0:x0+crop_size]
                crops.append(crop)
                coords.append((x0, y0))
    return crops, coords

def resize_crop_to_network_dim(crop_np, target_size=(512, 512)):
    img = Image.fromarray(crop_np)
    resized = img.resize(target_size, Image.BILINEAR)
    return np.array(resized)
