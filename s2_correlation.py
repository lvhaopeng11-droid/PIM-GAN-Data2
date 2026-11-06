import numpy as np

def compute_two_point_correlation(binary_mask, max_r=50):
    H, W = binary_mask.shape
    s2 = np.zeros(max_r)
    counts = np.zeros(max_r)
    
    # 径向二点相关统计
    for r in range(1, max_r):
        # 水平与垂直采样近似各向同性相关
        h_corr = np.mean(binary_mask[:, :-r] * binary_mask[:, r:])
        v_corr = np.mean(binary_mask[:-r, :] * binary_mask[r:, :])
        s2[r] = (h_corr + v_corr) / 2.0
    s2[0] = np.mean(binary_mask)
    return s2
