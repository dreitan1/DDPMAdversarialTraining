# evaluate_resnet.py

import os
import argparse
import torch
import torch.nn as nn
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
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    return testloader

# ========== Evaluation Function ==========
def evaluate_resnet(resnet, diffusion_model, embedder, dataloader):
    resnet.eval()
    diffusion_model.eval()

    total = 0
    correct_clean = 0
    correct_adv = 0

    with torch.no_grad():
        for x_clean, y_true in tqdm(dataloader, desc="Evaluating"):
            x_clean = x_clean.to(device)
            y_true = y_true.to(device)
            batch_size = x_clean.size(0)

            # Predict on clean
            logits_clean = resnet(x_clean)
            preds_clean = logits_clean.argmax(dim=1)
            correct_clean += (preds_clean == y_true).sum().item()

            # Generate adversarial samples
            embed = embedder()
            embed = embed.repeat(batch_size, 1)

            T = diffusion_model.num_timesteps
            t_sample = torch.randint(0, T, (batch_size,), device=device).long()

            noise = torch.randn_like(x_clean)
            x_noisy = diffusion_model.q_sample(x_start=x_clean, t=t_sample, noise=noise)

            x_adv = diffusion_model.model(x_noisy, x_clean, t_sample, embed)

            # Predict on adversarial
            logits_adv = resnet(x_adv)
            preds_adv = logits_adv.argmax(dim=1)
            correct_adv += (preds_adv == y_true).sum().item()

            total += batch_size

    clean_acc = 100 * correct_clean / total
    adv_acc = 100 * correct_adv / total
    attack_success_rate = 100 - adv_acc

    print(f"✅ Evaluation Results:")
    print(f" - Clean Accuracy: {clean_acc:.2f}%")
    print(f" - Adversarial Accuracy: {adv_acc:.2f}%")
    print(f" - Attack Success Rate: {attack_success_rate:.2f}%")

# ========== Main ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resnet-path', type=str, required=True, help='Path to adversarially trained ResNet checkpoint')
    parser.add_argument('--diffusion-path', type=str, required=True, help='Path to trained DDPM checkpoint')
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()

    # Load data
    dataloader = get_cifar10(batch_size=args.batch_size)

    # Load ResNet
    resnet = ResNet20().to(device)
    resnet.load_state_dict(torch.load(args.resnet_path, map_location=device))
    resnet.eval()

    # Load Conditional DDPM
    unet = ConditionalUnet(image_channels=3, base_dim=64, embed_dim=256).to(device)
    diffusion = ConditionalGaussianDiffusion(model=unet, image_size=32, timesteps=1000).to(device)
    diffusion.load_state_dict(torch.load(args.diffusion_path, map_location=device))
    diffusion.eval()

    # Embedder
    embedder = ResNetEmbedder(resnet).to(device)

    # Evaluate
    evaluate_resnet(resnet, diffusion, embedder, dataloader)