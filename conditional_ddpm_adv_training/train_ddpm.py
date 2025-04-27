# train_ddpm.py

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from models.conditional_unet import ConditionalUnet
from models.conditional_diffusion import ConditionalGaussianDiffusion
from models.resnet_embedder import ResNetEmbedder
from models.resnet20 import ResNet20

# ========== Device Setup ==========
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
print(f"Using device: {device}")

# ========== CIFAR-10 Loader ==========
def get_cifar10(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    return trainloader

# ========== Train Function ==========
def train_diffusion(resnet, embedder, diffusion_model, dataloader, epochs=100, lr=1e-4, save_every=10, resume_epoch=None):
    optimizer = optim.Adam(diffusion_model.parameters(), lr=lr)

    diffusion_model.train()
    resnet.eval()  # Freeze ResNet during diffusion training

    start_epoch = 0

    # ====== Resume from checkpoint ======
    if resume_epoch is not None:
        ckpt_path = f'checkpoints_ddpm/diffusion_epoch{resume_epoch}.pth'
        if os.path.exists(ckpt_path):
            print(f"Resuming diffusion model from {ckpt_path}")
            diffusion_model.load_state_dict(torch.load(ckpt_path, map_location=device))
            start_epoch = resume_epoch
        else:
            print(f"Warning: Checkpoint {ckpt_path} not found. Starting from scratch.")

    for epoch in range(start_epoch, epochs):
        pbar = tqdm(dataloader, desc=f"Diffusion Epoch {epoch+1}/{epochs}")
        epoch_loss = 0.0

        for x_clean, _ in pbar:
            x_clean = x_clean.to(device)
            batch_size = x_clean.size(0)

            # Extract dynamic ResNet embedding
            embed = embedder()
            embed = embed.repeat(batch_size, 1)

            # Sample random timestep
            t = torch.randint(0, diffusion_model.num_timesteps, (batch_size,), device=device).long()

            # Predict noise and compute MSE loss
            loss = diffusion_model.p_losses(x_start=x_clean, t=t, cond=embed, x_cond=x_clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=epoch_loss / (pbar.n + 1))

        # Save checkpoints
        os.makedirs('checkpoints_ddpm', exist_ok=True)
        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            torch.save(diffusion_model.state_dict(), f'checkpoints_ddpm/diffusion_epoch{epoch+1}.pth')
            print(f"Checkpoint saved at epoch {epoch+1}.")

# ========== Main ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resnet-path', type=str, required=True, help='Path to pretrained ResNet model')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save-every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--resume-epoch', type=int, default=None, help='Resume training from this epoch checkpoint')
    args = parser.parse_args()

    # Load data
    dataloader = get_cifar10(batch_size=args.batch_size)

    # Load pretrained ResNet
    resnet = ResNet20().to(device)
    resnet.load_state_dict(torch.load(args.resnet_path, map_location=device))
    resnet.eval()

    # Build models
    unet = ConditionalUnet(image_channels=3, base_dim=64, embed_dim=256).to(device)
    diffusion = ConditionalGaussianDiffusion(model=unet, image_size=32, timesteps=1000).to(device)
    embedder = ResNetEmbedder(resnet).to(device)

    # Train
    train_diffusion(
        resnet, embedder, diffusion, dataloader,
        epochs=args.epochs, lr=args.lr,
        save_every=args.save_every,
        resume_epoch=args.resume_epoch
    )