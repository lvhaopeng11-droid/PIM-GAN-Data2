import pandas as pd

def export_table_s7_summary():
    data = {
        "Time (min)": [30, 60, 120, 240],
        "Exp_Vf (%)": ["8.40±0.35", "7.55±0.31", "5.12±0.21", "4.10±0.18"],
        "Gen_Vf (%)": ["8.40±0.32", "7.62±0.28", "5.24±0.19", "4.15±0.16"],
        "Exp_w (um)": [2.55, 2.41, 1.98, 1.86],
        "Gen_w (um)": [2.58, 2.38, 2.02, 1.84]
    }
    return pd.DataFrame(data)
