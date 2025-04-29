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

model.load_state_dict(torch.load('./train_data/at_model.pt', weights_only=True))


criterion = torch.nn.CrossEntropyLoss()
criterion = criterion.to(device)

eps = 0.3
iters = [5, 10, 20, 40, 80]

im = torchvision.io.read_image('./train_data/clean_imgs/0-94/1_f7a0db69-7049-4b37-a323-f5049871fbdc.png').to(torch.float32) / 255

torchvision.utils.save_image(im, f"./pgd_tests/0pgd_im.png")

for iter in iters:
    attack = PGDAttack(model, loss_fn=criterion, nb_iter=iter, eps=eps)

    adv_im = attack.perturb(im.to(device).unsqueeze(0), torch.Tensor([1]).to(torch.int64).to(device))[0]

    torchvision.utils.save_image(adv_im, f"./pgd_tests/{iter}pgd_im.png")


