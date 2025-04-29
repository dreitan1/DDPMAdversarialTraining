
import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianDiffusion(nn.Module):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        self.register_buffer('betas', torch.linspace(beta_start, beta_end, timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_cumprod = self.alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = (1. - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)

        return sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise

    def p_losses(self, x_start, t, cond_embed, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t, noise)
        noise_pred = self.model(x_noisy, cond_embed)

        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def p_sample(self, x, t, cond_embed):
        beta_t = self.betas[t]
        sqrt_one_minus_alpha_cumprod = (1. - self.alphas_cumprod[t]).sqrt()
        sqrt_recip_alpha = (1. / self.alphas[t]).sqrt()

        noise_pred = self.model(x, cond_embed)

        model_mean = sqrt_recip_alpha * (x - beta_t / sqrt_one_minus_alpha_cumprod * noise_pred)

        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + beta_t.sqrt() * noise

    @torch.no_grad()
    def sample(self, x_shape, cond_embed, device):
        x = torch.randn(x_shape, device=device)

        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((x_shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t, cond_embed)

        return x
