
import torch
import torch.nn as nn
import torch.nn.functional as F
    
class GaussianDiffusion(nn.Module):
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod = self.alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus = (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        return sqrt_alphas_cumprod * x_start + sqrt_one_minus * noise

    def p_losses(self, x_start, t, clean_img, param):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        noise_pred = self.model(x_noisy, t, clean_img, param)
        return F.mse_loss(noise_pred, noise)
    
    @torch.no_grad()
    def predict_image(self, clean_img, param_vec):
        """
        Generate a new image conditioned on a clean image and a parameter vector.

        Args:
            clean_img (Tensor): shape (B, 3, H, W)
            param_vec (Tensor): shape (B, 256)

        Returns:
            Tensor: generated image (B, 3, H, W)
        """
        B, C, H, W = clean_img.shape
        x = torch.randn_like(clean_img).to(clean_img.device)

        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((B,), t, device=clean_img.device, dtype=torch.long)
            noise_pred = self.model(x, t_tensor, clean_img, param_vec)

            beta = self.betas[t]
            alpha = self.alphas[t]
            alpha_hat = self.alphas_cumprod[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / alpha.sqrt()) * (x - (1 - alpha) / (1 - alpha_hat).sqrt() * noise_pred) + beta.sqrt() * noise

        return x
    
