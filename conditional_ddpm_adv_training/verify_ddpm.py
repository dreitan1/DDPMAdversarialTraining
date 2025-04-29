# ================= verify_diffusion_denoise.py =================
import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.conditional_unet import ConditionalUNet
from models.conditional_diffusion import GaussianDiffusion
from models.resnet import resnet
from models.resnet_embedder import ResNetEmbedder

# ====== Argument parser ======
parser = argparse.ArgumentParser()
parser.add_argument('--diffusion-path', type=str, required=True, help='Path to trained DDPM model')
parser.add_argument('--resnet-path', type=str, default=None, help='Path to trained ResNet model')
parser.add_argument('--noise', type=float, default=0, help='Noise level for the input images')
parser.add_argument('--save-dir', type=str, default='./verify_samples', help='Directory to save generated images')
args = parser.parse_args()

# ====== Device setup ======
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ====== Load CIFAR-10 Dataset ======
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

num_workers = 0 if device.type == 'mps' else 4

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=5, shuffle=True, num_workers=num_workers)

# ====== Load trained diffusion model ======
cond_unet = ConditionalUNet(cond_dim=256).to(device)
diffusion_model = GaussianDiffusion(model=cond_unet, timesteps=1000).to(device)

state_dict = torch.load(args.diffusion_path, map_location=device)
diffusion_model.load_state_dict(state_dict)
diffusion_model.eval()

# ====== Get 5 sample images ======
x_clean, _ = next(iter(testloader))
x_clean = x_clean.to(device)

noise = args.noise * torch.randn_like(x_clean)
x_noisy = (x_clean + noise).clamp(-1, 1)

batch_size = x_clean.size(0)
cond_dim = 256

if args.resnet_path is None:
    cond_embed = torch.randn(batch_size, cond_dim, device=device)
else:
    resnet_model = resnet("resnet18", num_classes=10, device=device)
    resnet_model.load_state_dict(torch.load(args.resnet_path, map_location=device))
    resnet_model.to(device).eval()

    embedder = ResNetEmbedder(embed_dim=cond_dim).to(device)
    embedder.resnet = resnet_model  # assign the model before forward
    cond_embed_single = embedder().detach()
    cond_embed = cond_embed_single.unsqueeze(0).repeat(batch_size, 1)

# ====== Predict and denoise ======
with torch.no_grad():
    predicted_noise = diffusion_model.model(x_noisy, cond_embed)

x_denoised = (x_noisy - predicted_noise).clamp(-1, 1)

# ====== Visualize and Save ======
os.makedirs(args.save_dir, exist_ok=True)
fig, axes = plt.subplots(5, 3, figsize=(10, 15))

for i in range(5):
    img = (x_clean[i].cpu() + 1) / 2
    axes[i, 0].imshow(img.permute(1,2,0))
    axes[i, 0].axis('off')
    axes[i, 0].set_title('Clean')

    img = (x_noisy[i].cpu() + 1) / 2
    axes[i, 1].imshow(img.permute(1,2,0))
    axes[i, 1].axis('off')
    axes[i, 1].set_title('Noisy')

    img = (x_denoised[i].cpu() + 1) / 2
    axes[i, 2].imshow(img.permute(1,2,0))
    axes[i, 2].axis('off')
    axes[i, 2].set_title('Denoised')

plt.tight_layout()
save_path = os.path.join(args.save_dir, 'denoise_comparison.png')
plt.savefig(save_path)
print(f"Saved comparison plot to {save_path}")
plt.show()