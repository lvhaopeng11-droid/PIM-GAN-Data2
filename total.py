import torch
import torch.nn as nn
from .adversarial import AdversarialBCEWithLogitsLoss
from .vgg19_gram import VGG19GramTextureLoss
from ..physics.phase_score import get_continuous_phase_score
from ..physics.soft_area_fraction import compute_soft_area_fraction
from ..physics.jmak import jmak_dissolution_residual
from ..physics.lsw import lsw_coarsening_residual
from ..physics.kinetic_switch import get_kinetic_smoothing_weight

class CompositePIMGANLoss(nn.Module):
    def __init__(self, lambda_adv=1.0, lambda_tex=10.0, lambda_phys=5.0, T_ref=0.5):
        super().__init__()
        self.lambda_adv = lambda_adv
        self.lambda_tex = lambda_tex
        self.lambda_phys = lambda_phys
        self.T_ref = T_ref
        
        self.adv = AdversarialBCEWithLogitsLoss()
        self.tex = VGG19GramTextureLoss()

    def forward(self, gen_img, real_img, fake_logits, kinetic_params, t_hold, calphad_A_eq):
        l_adv = self.adv(fake_logits, target_is_real=True)
        l_tex = self.tex(gen_img, real_img)
        
        P_delta = get_continuous_phase_score(gen_img, self.T_ref)
        A_delta = compute_soft_area_fraction(P_delta)
        
        n, log_k, k_lsw = kinetic_params
        l_jmak = torch.mean(jmak_dissolution_residual(A_delta, t_hold, n, log_k))
        l_lsw = torch.mean(lsw_coarsening_residual(P_delta, A_delta, t_hold, k_lsw))
        
        omega = get_kinetic_smoothing_weight(A_delta, calphad_A_eq)
        l_phys = torch.mean((1.0 - omega) * l_jmak + omega * l_lsw)
        
        total = self.lambda_adv * l_adv + self.lambda_tex * l_tex + self.lambda_phys * l_phys
        return total, {"adv": l_adv.item(), "tex": l_tex.item(), "phys": l_phys.item()}
