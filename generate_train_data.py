import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np

from resnet import resnet
from pgd import PGDAttack

import uuid
from tqdm import tqdm
import os

device = torch.device('cuda:1')

model = resnet('resnet18')
model.train()
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
criterion = criterion.to(device)

attack = PGDAttack(model, loss_fn=criterion, nb_iter=5)

epochs = 100

transform = transforms.Compose(
    [
     transforms.ToTensor(),
     transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

batch_size = 128

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,75])

PATH = './train_data'

save_data = True

# Make sure proper folders exist
os.makedirs("train_data", exist_ok=True)
os.makedirs("train_data/imgs", exist_ok=True)
os.makedirs("train_data/params", exist_ok=True)

# Perform Adversarial Training with PGD and save images and model parameters
for ep in range(epochs):
    c = 0
    for data in tqdm(trainloader, desc=f"Training epoch {ep}"):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        adv_inputs = attack.perturb(inputs, labels)[0]
        if save_data and c < 1:
            param = f"{c}-{ep}"
            os.mkdir(f"{PATH}/imgs/{param}")
            # Save perturbed images
            for i, l in zip(adv_inputs, labels):
                fname = f"{l}_{str(uuid.uuid4())}"
                torchvision.utils.save_image(i, f"{PATH}/imgs/{param}/{fname}.png")
            # Save model parameters
            params = torch.cat([v.flatten() for k, v in model.state_dict().items() if 'weight' in k or 'bias' in k]).cpu().numpy()
            np.savetxt(f"{PATH}/params/{param}.txt", params, fmt="%f")
            c += 1
        
        optimizer.zero_grad()

        loss = criterion(model(inputs), labels)
        loss += criterion(model(adv_inputs), labels)

        loss.backward()
        optimizer.step()
    lr_scheduler.step()

model.eval()


