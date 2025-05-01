import torch
import torch.nn as nn

class ResNetEmbedder(nn.Module):
    def __init__(self, input_dim, out_dim=256):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, out_dim)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.compress(x)
    