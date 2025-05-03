import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]  # (B, half_dim)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # (B, dim)
        return emb


class ConditionalUNet(nn.Module):
    def __init__(self, param_dim=256, embed_dim=64):
        """
        Args:
            param_dim: Dimensionality of the parameter vector.
            embed_dim: Dimension of the embedding used for both timestep and param.
        """
        super().__init__()
        self.embed_dim = embed_dim

        # Embedding layers
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )

        self.param_proj = nn.Sequential(
            nn.Linear(param_dim, embed_dim),
            nn.ReLU()
        )

        # Input channels: 3 (x) + 1 (clean_cond) + embed_dim (param) + embed_dim (time)
        in_channels = 3 + 1 + embed_dim + embed_dim

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x, t, clean_img, param_vec):
        """
        Args:
            x: Noisy image (B, 3, H, W)
            t: Timestep tensor (B,)
            clean_img: Clean image (B, 3, H, W)
            param_vec: ResNet parameter embedding vector (B, param_dim)
        Returns:
            Predicted noise (B, 3, H, W)
        """
        B, _, H, W = x.shape

        # Param embedding → map
        param_embed = self.param_proj(param_vec)  # (B, embed_dim)
        param_map = param_embed.view(B, -1, 1, 1).expand(-1, -1, H, W)

        # Timestep embedding → map
        t_embed = self.time_mlp(t)  # (B, embed_dim)
        t_map = t_embed.view(B, -1, 1, 1).expand(-1, -1, H, W)

        # Clean image conditioning (take first channel)
        clean_cond = clean_img[:, :1]  # (B, 1, H, W)

        # Concatenate all conditioning inputs
        x_input = torch.cat([x, clean_cond, param_map, t_map], dim=1)  # (B, in_channels, H, W)

        # Forward through the CNN
        x = F.relu(self.conv1(x_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.out_conv(x)