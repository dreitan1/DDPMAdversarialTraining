# models/conditional_diffusion.py

import torch
import torch.nn as nn
from denoising_diffusion_pytorch import GaussianDiffusion as BaseGaussianDiffusion

class ConditionalGaussianDiffusion(BaseGaussianDiffusion):
    """
    Extends the original GaussianDiffusion to handle conditioning (cond).
    """

    def __init__(self, model, image_size, timesteps=1000, sampling_timesteps=None):
        super().__init__(
            model=model,
            image_size=image_size,
            timesteps=timesteps,
            sampling_timesteps=sampling_timesteps,
            objective='pred_noise'
        )

    def p_losses(self, x_start, t, cond, noise=None, x_cond=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Call updated UNet
        model_out = self.model(x_noisy, x_cond, t, cond)  # <-- Pass x_cond (clean image)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'Unknown objective {self.objective}')

        loss = nn.functional.mse_loss(model_out, target)
        return loss

    @torch.no_grad()
    def sample(self, batch_size=16, cond=None):
        """
        Sampling function that passes cond into model at each step.
        """
        device = next(self.model.parameters()).device
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        img = torch.randn(shape, device=device)

        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(img, torch.full((batch_size,), i, device=device, dtype=torch.long), cond=cond)

        return img

    @torch.no_grad()
    def p_sample(self, x, t, cond=None):
        """
        Single reverse step in DDPM.
        """
        model_mean, _, model_log_variance = self.q_posterior(x_start=self.model(x, t, cond), x_t=x, t=t)
        noise = torch.randn_like(x) if (t > 0).all() else 0.0
        return model_mean + (0.5 * model_log_variance).exp() * noise