import os
from pim_gan.plots.fig_s9 import plot_five_seed_variability

def main():
    os.makedirs("results/figures", exist_ok=True)
    plot_five_seed_variability("results/figures/Fig_S9.png")
    print("Successfully generated Fig_S9.png in results/figures/!")

if __name__ == "__main__":
    main()
