import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.resnet import resnet
from models.conditional_unet import ConditionalUNet
from models.conditional_diffusion import GaussianDiffusion
from models.resnet_embedder import ResNetEmbedder
import time


def extract_resnet_params(model):
    return torch.cat([p.view(-1) for p in model.parameters() if p.requires_grad])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save-dir', type=str, default='checkpoints_resnet_adv')
    parser.add_argument('--diffusion-path', type=str, required=True)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--timesteps', type=int, default=1000)
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

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    cond_unet = ConditionalUNet(param_dim=256).to(device)
    diffusion_model = GaussianDiffusion(model=cond_unet, timesteps=args.timesteps).to(device)
    diffusion_model.load_state_dict(torch.load(args.diffusion_path, map_location=device))
    diffusion_model.eval()

    resnet_name = "resnet18"
    resnet_model = resnet(resnet_name, num_classes=10, device=device).to(device)

    optimizer = optim.Adam(resnet_model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(args.save_dir, exist_ok=True)

    total_train_start = time.time()
    epoch_times = []
    cond_dim = 256

    # Determine the output dimension of the ResNet model
    dummy_input = torch.randn(1, 3, 32, 32).to(device)  # Assuming CIFAR-10 input size (3x32x32)
    with torch.no_grad():
        resnet_output = resnet_model(dummy_input)
    input_dim = resnet_output.view(resnet_output.size(0), -1).shape[1]  # Flatten and get feature size

    for epoch in range(1, args.epochs + 1):
        ep_start_time = time.time()
        epoch_start = time.time()

        resnet_model.train()
        total_loss = 0
        correct = 0
        total = 0

        show_images = (epoch % 50 == 0 or epoch == args.epochs)
        if show_images:
            fig, axes = plt.subplots(10, 2, figsize=(10, 15))
            i = 0

        pbar = tqdm(trainloader, desc=f"Epoch {epoch}/{args.epochs}")
        for x_clean, labels in pbar:
            x_clean, labels = x_clean.to(device), labels.to(device)

            output_clean = resnet_model(x_clean)
            input_dim = output_clean.view(output_clean.size(0), -1).shape[1]  # Flatten and get feature size

            # Initialize the embedder with the correct input_dim
            embedder = ResNetEmbedder(input_dim=input_dim, out_dim=cond_dim).to(device)
            embedder.resnet = resnet_model

            # Pass the flattened ResNet output to the embedder
            with torch.no_grad():
                flattened_output = output_clean.view(output_clean.size(0), -1)  # Flatten the ResNet output
                param_vec = embedder(flattened_output)
                start_time = time.time()
                generated_img = diffusion_model.predict_image(x_clean, param_vec)
                elapsed_time = time.time() - start_time
                print(f"Time taken for diffusion_model.predict_image: {elapsed_time:.4f} seconds")

                if (epoch % args.save_every == 0) or (epoch == args.epochs):
                    # ====== Visualize and Save ======
                    os.makedirs(args.save_dir, exist_ok=True)
                    fig, axes = plt.subplots(5, 2, figsize=(8, 2 * 5))
                    for i in range(5):
                        img_clean = x_clean[i].cpu() * 0.5 + 0.5
                        img_gen = generated_img[i].cpu() * 0.5 + 0.5

                        axes[i, 0].imshow(img_clean.permute(1, 2, 0).clamp(0, 1))
                        axes[i, 0].axis('off')
                        axes[i, 0].set_title('Clean')

                        axes[i, 1].imshow(img_gen.permute(1, 2, 0).clamp(0, 1))
                        axes[i, 1].axis('off')
                        axes[i, 1].set_title('Generated')

                    # Adversarial training            
                    plt.tight_layout()
                    save_path = os.path.join(args.save_dir, f'train_{resnet_name}_adv-{epoch}.png')
                    plt.savefig(save_path)
                    print(f"Saved comparison plot to {save_path}")
                    plt.close()

            # Forward pass on both clean and adversarial data
            
            output_adv = resnet_model(generated_img)

            # Combine losses
            loss_clean = nn.CrossEntropyLoss()(output_clean, labels)
            loss_adv = nn.CrossEntropyLoss()(output_adv, labels)
            loss = 0.5 * loss_clean + 0.5 * loss_adv

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_clean.size(0)
            _, predicted = output_clean.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=total_loss / total, acc=100. * correct / total)

        ep_elapsed_time = time.time() - ep_start_time
        print(f"Time taken for epoch {epoch} : {ep_elapsed_time:.4f} seconds")


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

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        if (epoch % args.save_every == 0) or (epoch == args.epochs):
            ckpt_path = os.path.join(args.save_dir, f'{resnet_name}_ddpm_adv_epoch{epoch}.pth')
            torch.save(resnet_model.state_dict(), ckpt_path)
            print(f"[Epoch {epoch}] Saved checkpoint to {ckpt_path}")

    total_time = time.time() - total_train_start
    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    report_path = os.path.join(args.save_dir, "train_report.txt")
    with open(report_path, "w") as f:
        f.write("=== Training Report ===\n")
        f.write(f"Total epochs: {args.epochs}\n")
        f.write(f"Dataset size: {len(trainset)} samples\n")
        f.write(f"Device used: {device}\n")
        f.write(f"Total training time: {total_time:.2f} seconds\n")
        f.write(f"Average time per epoch: {avg_epoch_time:.2f} seconds\n")

    print("Training complete. Report saved to:", report_path)


if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn')
    main()
