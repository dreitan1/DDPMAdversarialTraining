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
parser.add_argument('--save-dir', type=str, default='./verify_ddpm', help='Directory to save generated images')
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
cond_unet = ConditionalUNet(param_dim=256).to(device)
diffusion_model = GaussianDiffusion(model=cond_unet, timesteps=1000).to(device)

state_dict = torch.load(args.diffusion_path, map_location=device)
diffusion_model.load_state_dict(state_dict)
diffusion_model.eval()

# ====== Get sample clean images ======
clean_img, _ = next(iter(testloader))  # shape: (B, 3, 32, 32)
clean_img = clean_img.to(device)
batch_size = clean_img.size(0)
cond_dim = 256

# ====== Generate conditioning vectors ======
if args.resnet_path is None:
    # Use random parameter vectors if no ResNet is provided
    param_vec = torch.randn(batch_size, cond_dim, device=device)
else:
    # Embed from ResNet feature space
    resnet_model = resnet("resnet18", num_classes=10, device=device)
    resnet_model.load_state_dict(torch.load(args.resnet_path, map_location=device))
    resnet_model.to(device).eval()

    embedder = ResNetEmbedder(embed_dim=cond_dim).to(device)
    embedder.resnet = resnet_model

    with torch.no_grad():
        param_vec = embedder(clean_img)

# ====== Predict adversarial image using diffusion ======
with torch.no_grad():
    generated_img = diffusion_model.predict_image(clean_img, param_vec)

# ====== Visualize and Save ======
os.makedirs(args.save_dir, exist_ok=True)
fig, axes = plt.subplots(batch_size, 2, figsize=(8, 2 * batch_size))

for i in range(batch_size):
    img_clean = clean_img[i].cpu() * 0.5 + 0.5
    img_gen = generated_img[i].cpu() * 0.5 + 0.5

    axes[i, 0].imshow(img_clean.permute(1, 2, 0).clamp(0, 1))
    axes[i, 0].axis('off')
    axes[i, 0].set_title('Clean')

    axes[i, 1].imshow(img_gen.permute(1, 2, 0).clamp(0, 1))
    axes[i, 1].axis('off')
    axes[i, 1].set_title('Generated')

plt.tight_layout()

# Save with filename based on model
ddpm_base = os.path.splitext(os.path.basename(args.diffusion_path))[0]
filename = f"{ddpm_base}_comparison.png"
save_path = os.path.join(args.save_dir, filename)

plt.savefig(save_path)
print(f"Saved comparison plot to {save_path}")
plt.show()