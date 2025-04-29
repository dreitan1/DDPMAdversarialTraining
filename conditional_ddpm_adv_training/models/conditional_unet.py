
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalUNet(nn.Module):
    def __init__(self, cond_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.fc_condition = nn.Linear(cond_dim, 256 * 8 * 8)

        self.deconv1 = nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 3, kernel_size=3, padding=1)

    def forward(self, x_noisy, cond_embed):
        h1 = F.relu(self.conv1(x_noisy))  # (B, 64, 32, 32)
        h2 = F.relu(self.conv2(h1))       # (B, 128, 32, 32)
        h3 = F.relu(self.conv3(h2))        # (B, 256, 32, 32)

        cond_spatial = self.fc_condition(cond_embed).view(-1, 256, 8, 8)
        cond_spatial = F.interpolate(cond_spatial, size=(32, 32), mode='bilinear')

        h3_cond = torch.cat([h3, cond_spatial], dim=1)  # (B, 512, 32, 32)

        u1 = F.relu(self.deconv1(h3_cond))              # (B, 128, 64, 64)
        h2_resized = F.interpolate(h2, size=u1.shape[-2:], mode='bilinear', align_corners=False)
        u1 = torch.cat([u1, h2_resized], dim=1)          # (B, 256, 64, 64)

        u2 = F.relu(self.deconv2(u1))                    # (B, 64, 128, 128)
        h1_resized = F.interpolate(h1, size=u2.shape[-2:], mode='bilinear', align_corners=False)
        u2 = torch.cat([u2, h1_resized], dim=1)           # (B, 128, 128, 128)

        out = self.deconv3(u2)                           # (B, 3, 128, 128)
        out = F.interpolate(out, size=(32, 32), mode='bilinear', align_corners=False)  # resize back to CIFAR-10 size

        return out
