# models/conditional_unet.py

import torch
import torch.nn as nn
from denoising_diffusion_pytorch import Unet

class ConditionalUnet(nn.Module):
    def __init__(self, image_channels=3, base_dim=64, embed_dim=256):
        super().__init__()
        
        self.channels = image_channels  # needed for diffusion
        self.self_condition = False     # needed for diffusion

        # New UNet will take concatenated (x_noisy + x_clean)
        self.unet = Unet(
            dim=base_dim,
            dim_mults=(1, 2, 4),
            channels=image_channels * 2,  # <-- Important: input 6 channels now
            out_dim=image_channels
        )
        
        # Conditioning network: model embedding (ResNet params)
        self.film = nn.Sequential(
            nn.Linear(embed_dim, image_channels * 2),
            nn.ReLU()
        )

    def forward(self, x_noisy, x_clean, t, cond):
        """
        Inputs:
            x_noisy: noised image
            x_clean: clean image (original input)
            t: timestep tensor
            cond: ResNet embedding (batch_size, embed_dim)
        """
        # Concatenate x_noisy and x_clean along channel dimension
        x_input = torch.cat([x_noisy, x_clean], dim=1)  # (B, 6, H, W)

        # FiLM conditioning
        gamma_beta = self.film(cond)  # (B, 2*C)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        # Run Unet
        out = self.unet(x_input, t)  # (B, C, H, W)

        return (1 + gamma) * out + beta