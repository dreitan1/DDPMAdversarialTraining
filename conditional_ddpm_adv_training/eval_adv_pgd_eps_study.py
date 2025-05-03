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

# Add PGD path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pgd import PGDAttack

def imshow(img, title):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

def main():
    # === Config ===
    resnet_path = "/Users/kinchintong/project/DDPMAdversarialTraining/conditional_ddpm_adv_training/checkpoints_resnet_adv/resnet18_ddpm_adv_epoch20.pth"
    epsilons = [0.001, 0.005,0.010, 0.01, 0.02, 0.025, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    iters = 5

    # === Device ===
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Data ===
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # === Load Model ===
    model = resnet("resnet18", num_classes=10, device=device)
    model.load_state_dict(torch.load(resnet_path, map_location=device))
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    # === Tracking Accuracy ===
    clean_acc_list = []
    adv_acc_list = []

    for eps in epsilons:
        print(f"\nEvaluating with epsilon: {eps}")
        attack = PGDAttack(model, loss_fn=criterion, nb_iter=iters, eps=eps)
        correct_clean = 0
        correct_pgd = 0
        total = 0

        pbar = tqdm(testloader, desc=f"Epsilon {eps}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            pgd_inputs = attack.perturb(inputs.clone(), labels)[0]

            with torch.no_grad():
                outputs_clean = model(inputs)
                _, predicted_clean = outputs_clean.max(1)

                outputs_pgd = model(pgd_inputs)
                _, predicted_pgd = outputs_pgd.max(1)

                correct_clean += predicted_clean.eq(labels).sum().item()
                correct_pgd += predicted_pgd.eq(labels).sum().item()
                total += labels.size(0)

            pbar.set_postfix(
                clean_acc=100. * correct_clean / total,
                pgd_acc=100. * correct_pgd / total
            )

        clean_acc = correct_clean / total
        pgd_acc = correct_pgd / total

        clean_acc_list.append(clean_acc)
        adv_acc_list.append(pgd_acc)

    # === Plot and Save ===
    plt.figure(figsize=(6, 5))
    plt.plot(epsilons, clean_acc_list, 'o-', label="Clean Accuracy")
    plt.plot(epsilons, adv_acc_list, '^--', label="PGD Accuracy")
    plt.xticks(np.arange(0, 0.6, step=0.1))
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.title("Accuracy vs Epsilon (PGD Attack)")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_vs_epsilon.png")
    plt.show()

if __name__ == '__main__':
    main()