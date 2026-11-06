import numpy as np
from scipy.stats import t

def welch_t_test_and_ci(exp_arr, gen_arr, conf=0.95):
    a = np.asarray(exp_arr, dtype=float)
    b = np.asarray(gen_arr, dtype=float)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se2 = va/na + vb/nb
    df = se2**2 / ((va/na)**2 / (na - 1) + (vb/nb)**2 / (nb - 1))
    diff = float(a.mean() - b.mean())
    crit = float(t.ppf((1 + conf)/2, df))
    return {
        "mean_diff": diff,
        "ci_lower": diff - crit * np.sqrt(se2),
        "ci_upper": diff + crit * np.sqrt(se2),
        "p_value": float(2 * t.sf(abs(diff / np.sqrt(se2)), df))
    }
