import argparse
import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.conditional_unet import ConditionalUNet
from models.conditional_diffusion import GaussianDiffusion
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from load_data import Dataset


def train_diffusion(ddpm, dataloader, optimizer, epochs, device, save_path, dataset_size, save_every):
    ddpm.to(device)
    ddpm.train()

    total_train_start = time.time()
    epoch_times = []

    for epoch in range(epochs):
        epoch_start = time.time()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0.0

        for param_embed, x_clean, x_adv in pbar:
            x_clean = x_clean.to(device)
            x_adv = x_adv.to(device)
            param_embed = param_embed.to(device)

            # Truncate param vector to 256-dim
            # Truncate param vector to 256-dim
            param_embed = param_embed.view(param_embed.size(0), -1)
            param_proj = param_embed[:, :256].to(device)

            t = torch.randint(0, ddpm.timesteps, (x_clean.size(0),), device=device).long()
            loss = ddpm.p_losses(x_start=x_adv, t=t, clean_img=x_clean, param=param_proj)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # Always save latest
        os.makedirs(save_path, exist_ok=True)

        # Save every N epochs
        if (epoch + 1) % save_every == 0 or (epoch + 1) == epochs:
            epoch_path = os.path.join(save_path, f"diffusion_epoch{epoch+1}.pth")
            torch.save(ddpm.state_dict(), epoch_path)
            print(f"Saved model checkpoint: {epoch_path}")
        else:
            print(f"Epoch {epoch+1} completed. No checkpoint saved.")

    total_time = time.time() - total_train_start
    avg_epoch_time = sum(epoch_times) / len(epoch_times)


    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Average time per epoch: {avg_epoch_time:.2f} seconds")

    # Get GPU or CPU name
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        device_name = "Apple MPS (Metal)"
    else:
        device_name = "CPU"

    # Save training report
    report_path = os.path.join(save_path, "train_report.txt")
    with open(report_path, "w") as f:
        f.write("=== Training Report ===\n")
        f.write(f"Total epochs: {epochs}\n")
        f.write(f"Dataset size: {dataset_size} samples\n")
        f.write(f"Device used: {device} ({device_name})\n")
        f.write(f"Total training time: {total_time:.2f} seconds\n")
        f.write(f"Average time per epoch: {avg_epoch_time:.2f} seconds\n")

    print(f"\nTraining complete. Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help='Path to custom dataset')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save-dir', type=str, default='checkpoints_ddpm')
    parser.add_argument('--save-every', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--timesteps', type=int, default=1000, help='')
    
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else
                          "cpu")
    print(f"Using device: {device}")

    dataset = Dataset(PATH=args.data_dir)
    dataset_size = len(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

    model = ConditionalUNet(param_dim=256)
    diffusion = GaussianDiffusion(model=model, timesteps=args.timesteps)
    optimizer = optim.Adam(diffusion.parameters(), lr=args.lr)

    train_diffusion(diffusion, dataloader, optimizer, args.epochs, device, args.save_dir, dataset_size, args.save_every)


if __name__ == '__main__':
    main()