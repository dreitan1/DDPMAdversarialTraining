
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalUNet(nn.Module):
    def __init__(self, param_dim=256, embed_dim=64):
        """
        Args:
            param_dim: Dimensionality of the vector condition.
            embed_dim: Dimension to embed the parameter before fusion.
        """
        super().__init__()
        self.param_fc = nn.Linear(param_dim, embed_dim)  # Embed param vector

        # Input: 3 (noisy image) + 1 (clean image channel) + 1 (param embedding) = 5
        self.conv1 = nn.Conv2d(5, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x, t, clean_img, param):
        """
        Args:
            x: Noisy image (B, 3, H, W), where H=W=32.
            t: Timestep (not used in this version).
            param: Vector condition (B, param_dim)

        Returns:
            Predicted noise (B, 3, H, W)
        """
        B, _, H, W = x.shape
        param_embed = self.param_fc(param)                          # (B, embed_dim)
        param_map = param_embed[:, 0].view(B, 1, 1, 1).expand(-1, 1, H, W)  # Use only 1st dim

        clean_cond = clean_img[:, 0:1]  # Grayscale conditioning or first channel
        x_input = torch.cat([x, clean_cond, param_map], dim=1)     # (B, 5, H, W)

        x = F.relu(self.conv1(x_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.out_conv(x)
