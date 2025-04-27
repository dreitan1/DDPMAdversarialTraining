# adv_train_resnet.py

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    return trainloader

# ========== Adversarial Training ==========
def adv_train_resnet(resnet, embedder, diffusion_model, dataloader, epochs=30, lr=1e-3, save_every=10):
    optimizer = optim.SGD(resnet.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    resnet.train()
    diffusion_model.eval()

    for epoch in range(epochs):
        pbar = tqdm(dataloader, desc=f"Adversarial Epoch {epoch+1}/{epochs}")
        epoch_loss = 0.0

        for x_clean, labels in pbar:
            x_clean = x_clean.to(device)
            labels = labels.to(device)
            batch_size = x_clean.size(0)

            # Dynamic ResNet embedding
            embed = embedder()
            embed = embed.repeat(batch_size, 1)

            # Random timesteps
            T = diffusion_model.num_timesteps
            t_sample = torch.randint(0, T, (batch_size,), device=device).long()

            with torch.no_grad():
                noise = torch.randn_like(x_clean)
                x_noisy = diffusion_model.q_sample(x_start=x_clean, t=t_sample, noise=noise)

                # Predict adversarial samples
                x_adv = diffusion_model.model(x_noisy, x_clean, t_sample, embed)

            # Train ResNet on adversarial samples
            logits = resnet(x_adv)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=epoch_loss / (pbar.n + 1))

        scheduler.step()

        # Save checkpoints
        os.makedirs('checkpoints_advtrain', exist_ok=True)
        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            torch.save(resnet.state_dict(), f'checkpoints_advtrain/resnet_adv_epoch{epoch+1}.pth')
            print(f"✅ Checkpoint saved at epoch {epoch+1}")

# ========== Main ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--diffusion-path', type=str, required=True, help='Path to trained DDPM checkpoint')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save-every', type=int, default=10)
    args = parser.parse_args()

    # Load data
    dataloader = get_cifar10(batch_size=args.batch_size)

    # Initialize ResNet20
    resnet = ResNet20().to(device)

    # Load Conditional DDPM
    unet = ConditionalUnet(image_channels=3, base_dim=64, embed_dim=256).to(device)
    diffusion = ConditionalGaussianDiffusion(model=unet, image_size=32, timesteps=1000).to(device)
    diffusion.load_state_dict(torch.load(args.diffusion_path, map_location=device))
    diffusion.eval()

    # Embedder
    embedder = ResNetEmbedder(resnet).to(device)

    # Train
    adv_train_resnet(resnet, embedder, diffusion, dataloader, epochs=args.epochs, lr=args.lr, save_every=args.save_every)