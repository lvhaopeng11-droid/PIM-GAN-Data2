import torch
import torch.nn as nn

class ProcessMLP(nn.Module):
    def __init__(self, d_model=512, C=20.0):
        super().__init__()
        self.C = C
        self.R = 8.314e-3
        self.raw_Q = nn.Parameter(torch.tensor(235.6, dtype=torch.float32))
        
        self.mlp_query = nn.Sequential(
            nn.Linear(4, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, d_model)
        )
        
        self.mlp_kinetic = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)
        )

    def forward(self, T_celsius, t_min):
        T_K = T_celsius + 273.15
        t_safe = torch.clamp(t_min, min=1.0)
        Q_eff = torch.clamp(self.raw_Q, 150.0, 350.0)
        
        arrhenius = torch.exp(-Q_eff / (self.R * T_K))
        larson_miller = (T_K / 1000.0) * (torch.log10(t_safe) + self.C)
        T_norm = (T_celsius - 900.0) / 150.0
        t_norm = (t_safe - 60.0) / 120.0
        
        X_phys = torch.stack([T_norm, t_norm, arrhenius, larson_miller], dim=-1)
        query = self.mlp_query(X_phys)
        
        k_params = self.mlp_kinetic(X_phys)
        n = 1.0 + 1.5 * torch.sigmoid(k_params[:, 0:1])
        log_k = k_params[:, 1:2]
        k_lsw = torch.exp(k_params[:, 2:3])
        
        return query, (n, log_k, k_lsw), Q_eff
