import torch
import torch.optim as optim
from ..models.generator import PIMGANGenerator
from ..models.discriminator import PIMGANDiscriminator
from ..losses.total import CompositePIMGANLoss
from .checkpoints import save_checkpoint

class PIMGANTrainer:
    def __init__(self, dataloader, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.loader = dataloader
        self.device = device
        self.gen = PIMGANGenerator().to(device)
        self.disc = PIMGANDiscriminator().to(device)
        self.opt_g = optim.Adam(self.gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.opt_d = optim.Adam(self.disc.parameters(), lr=2e-4, betas=(0.5, 0.999))
        self.criterion = CompositePIMGANLoss().to(device)

    def train_epoch(self, epoch):
        self.gen.train()
        self.disc.train()
        total_g, total_d = 0.0, 0.0
        
        for batch in self.loader:
            x_init = batch["initial"].to(self.device)
            x_final = batch["final"].to(self.device)
            T = batch["T"].to(self.device)
            t = batch["t"].to(self.device)
            
            # 1. 训练判别器 D
            self.opt_d.zero_grad()
            gen_final, _, _ = self.gen(x_init, T, t)
            
            p_real = torch.cat([x_init, x_final], dim=1)
            p_fake = torch.cat([x_init, gen_final.detach()], dim=1)
            
            l_real = torch.nn.functional.binary_cross_entropy_with_logits(
                self.disc(p_real), torch.ones_like(self.disc(p_real))
            )
            l_fake = torch.nn.functional.binary_cross_entropy_with_logits(
                self.disc(p_fake), torch.zeros_like(self.disc(p_fake))
            )
            l_d = (l_real + l_fake) * 0.5
            l_d.backward()
            self.opt_d.step()
            
            # 2. 训练生成器 G
            self.opt_g.zero_grad()
            gen_final, kinetic_params, _ = self.gen(x_init, T, t)
            p_fake = torch.cat([x_init, gen_final], dim=1)
            f_logits = self.disc(p_fake)
            
            l_g, _ = self.criterion(gen_final, x_final, f_logits, kinetic_params, t, torch.tensor(0.052, device=self.device))
            l_g.backward()
            self.opt_g.step()
            
            total_g += l_g.item()
            total_d += l_d.item()
            
        print(f"Epoch {epoch:03d} -> Loss_G: {total_g/len(self.loader):.4f} | Loss_D: {total_d/len(self.loader):.4f}")
