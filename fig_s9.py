import matplotlib.pyplot as plt
import numpy as np

def plot_five_seed_variability(save_path="results/figures/Fig_S9.png"):
    fig, ax = plt.subplots(figsize=(8, 4))
    metrics = ["FID", "IS", "S2 RMSE", "Area MAE (pp)", "Width MAE (um)"]
    means = [7.922, 7.538, 0.0262, 0.2318, 0.0786]
    sds = [0.153, 0.081, 0.0019, 0.0174, 0.0049]
    ax.barh(metrics, means, xerr=sds, color="#1D7681", alpha=0.8, capsize=4)
    ax.set_title("Fig. S9: PIM-GAN Performance Across 5 Independent Training Runs")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
