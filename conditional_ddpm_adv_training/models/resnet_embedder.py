# models/resnet_embedder.py

import torch
import torch.nn as nn

class ResNetEmbedder(nn.Module):
    """
    Extracts a trainable embedding from ResNet parameters.
    """
    def __init__(self, resnet, embed_dim=256):
        super().__init__()
        self.resnet = resnet
        self.embed_dim = embed_dim
        
        # Create an MLP to map flattened parameters to embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(self.get_total_params(), embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def get_total_params(self):
        """
        Count total number of parameters in ResNet model.
        """
        total = 0
        for p in self.resnet.parameters():
            total += p.numel()
        return total

    def forward(self):
        """
        Returns a (1, embed_dim) tensor representing the ResNet.
        """
        params = []
        for p in self.resnet.parameters():
            params.append(p.flatten())

        params = torch.cat(params)  # (total_params,)
        params = params.unsqueeze(0)  # (1, total_params)
        embed = self.mlp(params)
        return embed