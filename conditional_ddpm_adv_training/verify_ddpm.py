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
import time

# ====== Argument parser ======
parser = argparse.ArgumentParser()
parser.add_argument('--diffusion-path1', type=str, required=False, help='Path to trained DDPM model')
parser.add_argument('--diffusion-path2', type=str, required=False, help='Path to trained DDPM model')
parser.add_argument('--diffusion-path3', type=str, required=False, help='Path to trained DDPM model')
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
diffusion_path3 = "checkout_ddpm_t1000_ep500/diffusion_epoch500.pth"
diffusion_path2 = "checkout_ddpm_t500_ep500/diffusion_epoch500.pth"
diffusion_path1 = "checkout_ddpm_t300_ep500/diffusion_epoch500.pth"
cond_unet1 = ConditionalUNet(param_dim=256).to(device)
diffusion_model1 = GaussianDiffusion(model=cond_unet1, timesteps=300).to(device)
diffusion_model1.load_state_dict(torch.load(diffusion_path1, map_location=device))
diffusion_model1.eval()

cond_unet2 = ConditionalUNet(param_dim=256).to(device)
diffusion_model2 = GaussianDiffusion(model=cond_unet2, timesteps=500).to(device)
diffusion_model2.load_state_dict(torch.load(diffusion_path1, map_location=device))
diffusion_model2.eval()

cond_unet3 = ConditionalUNet(param_dim=256).to(device)
diffusion_model3 = GaussianDiffusion(model=cond_unet3, timesteps=1000).to(device)
diffusion_model3.load_state_dict(torch.load(diffusion_path1, map_location=device))
diffusion_model3.eval()

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
    start_time = time.time()
    generated_img1 = diffusion_model1.predict_image(clean_img, param_vec)
    end_time = time.time()
    time_per_image = (end_time - start_time) / batch_size
    print(f"Estimated time to generate one image (T={diffusion_model1.timesteps}): {time_per_image:.4f} seconds")

   
    start_time = time.time()
    generated_img2 = diffusion_model2.predict_image(clean_img, param_vec)
    end_time = time.time()
    time_per_image = (end_time - start_time) / batch_size
    print(f"Estimated time to generate one image (T={diffusion_model2.timesteps}): {time_per_image:.4f} seconds")


    start_time = time.time()
    generated_img3 = diffusion_model3.predict_image(clean_img, param_vec)
    end_time = time.time()
    time_per_image = (end_time - start_time) / batch_size
    print(f"Estimated time to generate one image (T={diffusion_model3.timesteps}): {time_per_image:.4f} seconds")


# ====== Visualize and Save ======
os.makedirs(args.save_dir, exist_ok=True)
fig, axes = plt.subplots(batch_size, 4, figsize=(8, 4 * batch_size))

for i in range(batch_size):
    img_clean = clean_img[i].cpu() * 0.5 + 0.5
    img_gen1 = generated_img1[i].cpu() * 0.5 + 0.5
    img_gen2 = generated_img2[i].cpu() * 0.5 + 0.5
    img_gen3 = generated_img3[i].cpu() * 0.5 + 0.5

    axes[i, 0].imshow(img_clean.permute(1, 2, 0).clamp(0, 1))
    axes[i, 0].axis('off')
    axes[i, 0].set_title('Clean')

    axes[i, 1].imshow(img_gen1.permute(1, 2, 0).clamp(0, 1))
    axes[i, 1].axis('off')
    axes[i, 1].set_title(f'T={diffusion_model1.timesteps}')

    axes[i, 2].imshow(img_gen2.permute(1, 2, 0).clamp(0, 1))
    axes[i, 2].axis('off')
    axes[i, 2].set_title(f'T={diffusion_model2.timesteps}')

    axes[i, 3].imshow(img_gen3.permute(1, 2, 0).clamp(0, 1))
    axes[i, 3].axis('off')
    axes[i, 3].set_title(f'T={diffusion_model3.timesteps}')
plt.tight_layout()

# Save with filename based on model]
filename = f"ddpm_comparison.png"
save_path = os.path.join(args.save_dir, filename)

plt.savefig(save_path)
print(f"Saved comparison plot to {save_path}")
plt.show()