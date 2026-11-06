import numpy as np

def augment_crop_twofold(crop_np):
    augmented = []
    # 原始
    augmented.append(crop_np.copy())
    # 水平翻转 + 微小旋转
    flipped = np.fliplr(crop_np)
    augmented.append(flipped)
    return augmented
