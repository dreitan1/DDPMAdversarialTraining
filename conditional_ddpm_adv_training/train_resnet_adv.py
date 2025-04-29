import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from models.resnet import resnet
from models.resnet_embedder import ResNetEmbedder
from models.conditional_unet import ConditionalUNet
from models.conditional_diffusion import GaussianDiffusion

def fgsm_attack(model, criterion, x, y, epsilon):
    x_adv = x.clone().detach().requires_grad_(True)
    model.eval()
    outputs = model(x_adv)
    loss = criterion(outputs, y)
    model.zero_grad()
    loss.backward()
    perturbation = epsilon * x_adv.grad.sign()
    x_adv = (x_adv + perturbation).clamp(-1, 1).detach()
    return x_adv

def main():
    # ====== Argument Parser ======
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save-dir', type=str, default='checkpoints_resnet_adv')
    parser.add_argument('--diffusion-path', type=str, required=True)
    parser.add_argument('--clean-ratio', type=float, default=0.0, help='Ratio of clean images to mix (0 = all adversarial, 1 = all clean)')
    parser.add_argument('--save-every', type=int, default=10, help='Save model every N epochs')
    parser.add_argument('--fgsm-eps', type=float, default=0.1, help='Epsilon for FGSM adversarial evaluation')
    args = parser.parse_args()

    # ====== Device Setup ======
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # ====== CIFAR-10 Dataset ======
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # ====== Load Diffusion Model ======
    cond_unet = ConditionalUNet(cond_dim=256).to(device)
    diffusion = GaussianDiffusion(model=cond_unet, timesteps=1000).to(device)
    diffusion.load_state_dict(torch.load(args.diffusion_path, map_location=device))
    diffusion.eval()

    # ====== ResNet18 & Embedder ======
    resnet_model = resnet("resnet18", num_classes=10, device=device).to(device)
    embedder = ResNetEmbedder(resnet_model, embed_dim=256).to(device)

    # ====== Optimizer & Loss ======
    optimizer = optim.Adam(resnet_model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # ====== Training Loop ======
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        resnet_model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(trainloader, desc=f"Epoch {epoch}/{args.epochs}")
        for x_clean, labels in pbar:
            x_clean, labels = x_clean.to(device), labels.to(device)

            noise = 0 * torch.randn_like(x_clean)
            x_noisy = (x_clean + noise).clamp(-1, 1)

            batch_size = x_clean.size(0)
            cond_dim = 256

            embedder = ResNetEmbedder(embed_dim=cond_dim).to(device)
            embedder.resnet = resnet_model  # assign the model before forward
            cond_embed_single = embedder().detach()
            cond_embed = cond_embed_single.unsqueeze(0).repeat(batch_size, 1)
            
            with torch.no_grad():
                predicted_noise = diffusion.model(x_noisy, cond_embed)

            x_adv = (x_noisy - predicted_noise).clamp(-1, 1)

            if args.clean_ratio == 0:
                x_train = x_adv
            elif args.clean_ratio == 1:
                x_train = x_clean
            else:
                x_train = args.clean_ratio * x_clean + (1 - args.clean_ratio) * x_adv

            outputs = resnet_model(x_train)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_clean.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=total_loss / total, acc=100. * correct / total)

        if epoch % 50 == 0:
            # ====== Clean Test Accuracy ======
            resnet_model.eval()
            correct_test = 0
            total_test = 0
            with torch.no_grad():
                for x_test, y_test in testloader:
                    x_test, y_test = x_test.to(device), y_test.to(device)
                    outputs = resnet_model(x_test)
                    _, predicted = outputs.max(1)
                    correct_test += predicted.eq(y_test).sum().item()
                    total_test += y_test.size(0)
            test_acc = 100. * correct_test / total_test
            print(f"[Epoch {epoch}] Clean Test Accuracy: {test_acc:.2f}%")

            # ====== FGSM Adversarial Accuracy ======
            correct_fgsm = 0
            total_fgsm = 0
            for x_test, y_test in testloader:
                x_test, y_test = x_test.to(device), y_test.to(device)
                x_fgsm = fgsm_attack(resnet_model, criterion, x_test, y_test, epsilon=args.fgsm_eps)
                outputs = resnet_model(x_fgsm)
                _, predicted = outputs.max(1)
                correct_fgsm += predicted.eq(y_test).sum().item()
                total_fgsm += y_test.size(0)
            fgsm_acc = 100. * correct_fgsm / total_fgsm
            print(f"[Epoch {epoch}] FGSM Adversarial Accuracy (eps={args.fgsm_eps}): {fgsm_acc:.2f}%")

        if (epoch % args.save_every == 0) or (epoch == args.epochs):
            ckpt_path = os.path.join(args.save_dir, f'resnet_epoch{epoch}.pth')
            torch.save(resnet_model.state_dict(), ckpt_path)
            print(f"[Epoch {epoch}] Saved checkpoint to {ckpt_path}")

    print("Adversarial training complete.")

if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn')
    main()
