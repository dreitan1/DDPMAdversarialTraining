import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import numpy as np

from resnet import resnet
from wideresnet import wideresnet
from pgd import PGDAttack

import uuid
from tqdm import tqdm
import os

from autoattack import AutoAttack

import time


# Transform images in dataset
class TransformModel(nn.Module):
    def __init__(self, model, transform):
        super(TransformModel, self).__init__()
        self.model = model
        self.t = transform
    

    def forward(self, x):
        return self.model(self.t(x))


device = torch.device('cuda:1')

transform = transforms.Compose(
    [
     transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

model = resnet('resnet18')
# model = wideresnet('wrn-28-10-swish')
model.train()
model = model.to(device)
model = TransformModel(model, transform)

criterion = torch.nn.CrossEntropyLoss()
criterion = criterion.to(device)

eps = 0.3
iters = 5
attack = PGDAttack(model, loss_fn=criterion, nb_iter=iters, eps=eps)

epochs = 50
# epochs = 0

batch_size = 256

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

train = True

optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

PATH = './train_data'

if train:
    save_data = False

    if save_data:
        # Make sure proper folders exist
        os.makedirs("train_data", exist_ok=True)
        os.makedirs("train_data/adv_imgs", exist_ok=True)
        os.makedirs("train_data/clean_imgs", exist_ok=True)
        os.makedirs("train_data/params", exist_ok=True)

    times = []

    start_time = time.perf_counter()

    # Perform Adversarial Training with PGD and save images and model parameters
    for ep in range(epochs):
        c = 0
        pbar = tqdm(trainloader, desc=f"Training epoch {ep}")
        for step, data in enumerate(pbar):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if ep < 10:
                loss = criterion(model(inputs), labels)
            else:
                # Perform attack and measure time taken
                adv_inputs = attack.perturb(inputs.clone(), labels)[0]

                if save_data and (c < 1) and (ep % 4 == 0):
                    s = time.perf_counter()
                    param = f"{c}-{ep}"
                    os.mkdir(f"{PATH}/adv_imgs/{param}")
                    os.mkdir(f"{PATH}/clean_imgs/{param}")
                    # Save perturbed images
                    for adv_im, clean_im, l in zip(adv_inputs, inputs, labels):
                        fname = f"{l}_{str(uuid.uuid4())}"
                        torchvision.utils.save_image(adv_im, f"{PATH}/adv_imgs/{param}/{fname}.png")
                        torchvision.utils.save_image(clean_im, f"{PATH}/clean_imgs/{param}/{fname}.png")
                    # Save model parameters
                    params = torch.cat([v.flatten() for k, v in model.state_dict().items() if 'weight' in k or 'bias' in k]).cpu().numpy()
                    np.savetxt(f"{PATH}/params/{param}.txt", params, fmt="%f")
                    c += 1
                    e = time.perf_counter()

                    # Ignore time taken by saving files
                    times.append(e - s)

                loss = criterion(model(inputs), labels) + criterion(model(adv_inputs), labels)

            pbar.set_description(f"Training epoch {ep}, loss = {loss:2f}")

            loss.backward()
            optimizer.step()
        lr_scheduler.step()

    end_time = time.perf_counter()

    print(f"Approx. time taken by AT = {(end_time - start_time) - sum(times)}")

    # Save AT model
    torch.save(model.state_dict(), f"{PATH}/at_model1.pt")
else:
    model.load_state_dict(torch.load(f"{PATH}/at_model.pt", weights_only=True))


accuracy = []
# Evaluate model roboustness accuracy
for data in tqdm(trainloader, desc=f"Testing Robust Train Accuracy"):
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    adv_input = attack.perturb(inputs, labels)[0]

    outputs = torch.argmax(model(adv_input), dim=-1)

    acc = (labels == outputs).int()

    accuracy.append(acc.sum() / acc.shape[0])


print(f"Train Accuracy: {sum(acc) / len(acc)}")




accuracy = []
# Evaluate model roboustness accuracy
for data in tqdm(testloader, desc=f"Testing Robust Test Accuracy"):
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    adv_input = attack.perturb(inputs, labels)[0]

    outputs = torch.argmax(model(adv_input), dim=-1)

    acc = (labels == outputs).int()

    accuracy.append(acc.sum() / acc.shape[0])


print(f"Test Accuracy: {sum(acc) / len(acc)}")


