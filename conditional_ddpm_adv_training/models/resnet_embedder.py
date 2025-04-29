import torch
import torch.nn as nn

class ResNetEmbedder(nn.Module):
    """
    Converts the final linear layer of a ResNet18 model into a 256-dim condition embedding.
    This avoids high memory use from embedding all parameters.
    """
    def __init__(self, resnet_model=None, embed_dim=256):
        super().__init__()
        self.resnet = resnet_model
        self.embed_dim = embed_dim
        self.proj = nn.Linear(512 * 10 + 10, embed_dim)  # 512 weights x 10 classes + 10 biases

    def forward(self):
        assert self.resnet is not None, "resnet_model must be set before calling forward()"
        weight = self.resnet.linear.weight.view(-1)
        bias = self.resnet.linear.bias.view(-1)
        flat = torch.cat([weight, bias], dim=0).to(self.proj.weight.device)
        return self.proj(flat)