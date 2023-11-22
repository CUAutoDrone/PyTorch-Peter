import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


train_dataset = torchvision.datasets.MNIST(
    "files/",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=torchvision.transforms.Compose(
        [lambda x: torch.LongTensor([x]), lambda x: F.one_hot(x, 10)]
    ),
)

training_set_size = 64
train_loader = DataLoader(train_dataset, batch_size=training_set_size, shuffle=True)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer1 = nn.Linear(784, 500)
        self.linear_layer2 = nn.Linear(500, 10)
        self.linear_layer3 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.linear_layer1(x)
        x = F.relu(x)
        x = self.linear_layer2(x)
        x = F.relu(x)
        x = self.linear_layer3(x)
        x = F.relu(x)
        return x


net = Model()
input = torch.rand(1, 1, 28, 28)
output = net(input)
