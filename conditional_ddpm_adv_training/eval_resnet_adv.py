import os
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.resnet import resnet

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

def evaluate(model, dataloader, device, attack=None, epsilon=0.1):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        if attack == 'fgsm':
            x = fgsm_attack(model, criterion, x, y, epsilon)
        with torch.no_grad():
            outputs = model(x)
            _, predicted = outputs.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)
    return 100. * correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resnet-path', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--fgsm-eps', type=float, default=0.1)
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = resnet("resnet18", num_classes=10, device=device)
    model.load_state_dict(torch.load(args.resnet_path, map_location=device))
    model.to(device)

    clean_acc = evaluate(model, testloader, device)
    adv_acc = evaluate(model, testloader, device, attack='fgsm', epsilon=args.fgsm_eps)

    print(f"Clean Accuracy: {clean_acc:.2f}%")
    print(f"FGSM Accuracy (eps={args.fgsm_eps}): {adv_acc:.2f}%")

if __name__ == '__main__':
    main()
