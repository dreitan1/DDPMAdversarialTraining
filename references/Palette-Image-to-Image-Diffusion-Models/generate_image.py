import torch
from models.model import Palette
from PIL import Image
import numpy as np
import os
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import random

# Load the configuration
config_path = "config/inpainting_cifar10.json"  # Change to inpainting_celebahq.json if needed
checkpoint_path = "datasets/celebhq/checkpoint/200_Network.pth"  # Update with your checkpoint path

# Load the model
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = Palette.load_from_checkpoint(checkpoint_path, map_location=device)
model.eval()
model.to(device)

# Load the input image from CIFAR-10 dataset and randomly choose one

# Download and load the CIFAR-10 dataset
cifar10_dataset = CIFAR10(root="datasets/cifar10", train=False, download=True, transform=ToTensor())

# Randomly select an image from the dataset
random_index = random.randint(0, len(cifar10_dataset) - 1)
input_tensor, _ = cifar10_dataset[random_index]

# Add batch dimension and move to the appropriate device
input_tensor = input_tensor.unsqueeze(0).to(device)

# Generate the output
with torch.no_grad():
    output_tensor, _ = model.netG.restoration(y_cond=input_tensor, sample_num=8)

# Convert the output tensor to an image
output_image = (output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
output_image = Image.fromarray(output_image)

# Save the output image
output_image_path = "path/to/your/output_image.png"  # Update with your desired output path
output_image.save(output_image_path)

print(f"Generated image saved to {output_image_path}")