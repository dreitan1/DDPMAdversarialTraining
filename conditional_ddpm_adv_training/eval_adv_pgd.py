import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.resnet import resnet
import sys
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pgd import PGDAttack

def imshow(img, title):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resnet-path', type=str, required=True, help='Path to trained ResNet model')
    parser.add_argument('--pgd-eps', type=float, default=0.3)
    parser.add_argument('--pgd-iters', type=int, default=5)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    model = resnet("resnet18", num_classes=10, device=device)
    model.load_state_dict(torch.load(args.resnet_path, map_location=device))
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    attack = PGDAttack(model, loss_fn=criterion, nb_iter=args.pgd_iters, eps=args.pgd_eps)
    correct_clean = 0
    correct_pgd = 0
    total = 0
    pbar = tqdm(testloader, desc="Evaluating")

    fig, axes = plt.subplots(10, 2, figsize=(10, 15))
    i=0
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        pgd_inputs = attack.perturb(inputs.clone(), labels)[0]

        with torch.no_grad():
            outputs_clean = model(inputs)
            _, predicted_clean = outputs_clean.max(1)

            outputs_pgd = model(pgd_inputs)
            _, predicted_pgd = outputs_pgd.max(1)

            if i < 10:
                img = (inputs[0].cpu() + 1) / 2
                axes[i, 0].imshow(img.permute(1,2,0))
                axes[i, 0].axis('off')
                axes[i, 0].set_title('Clean')

                img = (pgd_inputs[0].cpu() + 1) / 2
                axes[i, 1].imshow(img.permute(1,2,0))
                axes[i, 1].axis('off')
                axes[i, 1].set_title('diffusion genereated')
                i += 1
            #print(f"Clean: {predicted_clean.item()}, PGD: {predicted_pgd.item()}, Label: {labels.item()}")
            correct_clean += predicted_clean.eq(labels).sum().item()
            correct_pgd += predicted_pgd.eq(labels).sum().item()
            total += labels.size(0)

        #imshow(inputs[0], title=f"Original Image (True: {labels.item()}, Pred: {predicted_clean.item()})")
        #imshow(pgd_inputs[0], title=f"PGD Image (True: {labels.item()}, Pred: {predicted_pgd.item()})")

        pbar.set_postfix(clean_acc=100. * correct_clean / total, pgd_acc=100. * correct_pgd / total)

    clean_acc = 100. * correct_clean / total
    pgd_acc = 100. * correct_pgd / total

    print(f"Clean Accuracy: {clean_acc:.2f}%")
    print(f"PGD Accuracy (eps={args.pgd_eps}, iters={args.pgd_iters}): {pgd_acc:.2f}%")

if __name__ == '__main__':
    main()