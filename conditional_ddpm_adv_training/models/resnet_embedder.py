import torch
import torch.nn as nn

class ResNetEmbedder(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(ResNetEmbedder, self).__init__()
        self.compress = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, out_dim)
        )

    def forward(self, x):
        return self.compress(x)