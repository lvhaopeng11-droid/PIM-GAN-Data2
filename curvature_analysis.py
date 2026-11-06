import numpy as np

def correlate_attention_with_curvature(attn_map, curvature_map):
    return float(np.corrcoef(attn_map.flatten(), curvature_map.flatten())[0, 1])
