import numpy as np
from scipy.stats import t

def calculate_5seed_run_stats(metric_values, conf=0.95):
    arr = np.asarray(metric_values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    crit = float(t.ppf((1 + conf) / 2, n - 1))
    margin = crit * sd / np.sqrt(n)
    return {
        "mean": mean,
        "sd": sd,
        "ci_lower": mean - margin,
        "ci_upper": mean + margin
    }
