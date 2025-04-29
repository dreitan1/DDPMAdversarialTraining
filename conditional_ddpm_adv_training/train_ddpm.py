import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from models.conditional_unet import ConditionalUNet
from models.conditional_diffusion import GaussianDiffusion

# ====== Argument parser ======
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--save-every', type=int, default=10, help='Save checkpoint every N epochs')
args = parser.parse_args()

# ====== Device setup ======
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ====== Configurations ======
batch_size = args.batch_size
learning_rate = args.lr
num_epochs = args.epochs
save_every = args.save_every
cond_dim = 256

# ====== Dataset: CIFAR-10 ======
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

num_workers = 0 if device.type == 'mps' else 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# ====== Initialize Model ======
cond_unet = ConditionalUNet(cond_dim=cond_dim).to(device)
diffusion_model = GaussianDiffusion(model=cond_unet, timesteps=1000).to(device)

optimizer = optim.Adam(diffusion_model.parameters(), lr=learning_rate)

# Dummy embedder for now (random embeddings)
def get_random_condition(batch_size, cond_dim, device):
    return torch.randn(batch_size, cond_dim, device=device)

# ====== Training Loop ======
start_epoch = 0
os.makedirs('checkpoints_ddpm', exist_ok=True)

for epoch in range(start_epoch, num_epochs):
    diffusion_model.train()
    pbar = tqdm(trainloader, desc=f"Diffusion Training Epoch {epoch+1}/{num_epochs}")
    epoch_loss = 0.0

    for x_clean, _ in pbar:
        x_clean = x_clean.to(device)
        batch_size = x_clean.size(0)

        # Random condition for now
        cond_embed = get_random_condition(batch_size, cond_dim, device)

        # Sample random timestep t
        t = torch.randint(0, diffusion_model.timesteps, (batch_size,), device=device).long()

        # Compute loss
        loss = diffusion_model.p_losses(x_start=x_clean, t=t, cond_embed=cond_embed)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=epoch_loss / (pbar.n + 1))

    # Save checkpoint
    if (epoch + 1) % save_every == 0 or (epoch + 1) == num_epochs:
        torch.save(diffusion_model.state_dict(), f'checkpoints_ddpm/diffusion_epoch{epoch+1}.pth')
        print(f"Checkpoint saved at epoch {epoch+1}.")

print("Training completed.")
