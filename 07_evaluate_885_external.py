import torch
from pim_gan.models.generator import PIMGANGenerator
from pim_gan.evaluation.area_fraction import compute_phase_area_fraction_2d
from pim_gan.evaluation.characteristic_width import compute_characteristic_width_um

def main():
    print("=== 885 °C External Unseen Temperature Blind Evaluation ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gen = PIMGANGenerator().to(device)
    gen.eval()
    times = [30, 60, 90, 120, 150, 180, 210, 240]
    dummy_init = torch.zeros(1, 1, 512, 512, device=device)
    
    with torch.no_grad():
        for t in times:
            T_t = torch.tensor([885.0], device=device)
            t_t = torch.tensor([float(t)], device=device)
            out, _, _ = gen(dummy_init, T_t, t_t)
            mask = (out.squeeze().cpu().numpy() > 0.0).astype(int)
            area = compute_phase_area_fraction_2d(mask)
            w = compute_characteristic_width_um(mask)
            print(f"Holding Time: {t:3d} min | Pred Area Fraction: {area:5.2f}% | Width: {w:.3f} um")

if __name__ == "__main__":
    main()
