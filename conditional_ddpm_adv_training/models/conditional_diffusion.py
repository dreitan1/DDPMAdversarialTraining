
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
    
class GaussianDiffusion2(nn.Module):
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

    def p_losses(self, x_start, t, cond_clean, cond_embed, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t, noise)
        noise_pred = self.model(x_noisy, cond_embed)

        return F.mse_loss(noise_pred, noise)


    def ddpm_loss(self, x_start, cond, t, noise=None):
        """
        DDPM 'simple' L2 loss used in the Palette paper.

        Args:
            model: the denoising model ε_θ(x_t, t, cond)
            x_start: the clean image (x₀)
            cond: conditioning input (e.g., grayscale, masked image)
            t: timestep tensor (B,)
            noise: optional Gaussian noise ε ~ N(0, 1)

        Returns:
            L2 loss between predicted and true noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # Get the noised image x_t based on x₀ and timestep t
        alpha_cumprod = self.get_alpha_cumprod_schedule(t, x_start.device)
        sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alpha_cumprod).view(-1, 1, 1, 1)
        
        x_t = sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise

        # Predict noise using the model
        pred_noise = self.model(x_t, cond)

        # L2 loss between predicted and actual noise
        return F.mse_loss(pred_noise, noise)

    def get_alpha_cumprod_schedule(self, t, device):
        """
        Returns alpha_bar(t), the cumulative product of (1 - beta) up to timestep t.
        This should be precomputed in your diffusion setup.
        """
        # Dummy beta schedule for illustration (cosine/beta-linear preferred)
        betas = torch.linspace(1e-4, 0.02, 1000, device=device)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        return alphas_cumprod[t]

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
