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

test_dataset = torchvision.datasets.MNIST(
    "files/",
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=torchvision.transforms.Compose(
        [lambda x: torch.LongTensor([x]), lambda x: F.one_hot(x, 10)]
    ),
)

test_set_size = 64
test_loader = DataLoader(test_dataset, batch_size=test_set_size, shuffle=True)


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

optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

criterion = nn.CrossEntropyLoss()

train_losses = []
test_losses = []


def train_epoch(model, optimizer, dataloader):
    model.train()

    for input, labels in dataloader:
        optimizer.zero_grad()
        output = model(input)
        labels = torch.squeeze(labels, dim=1).float()
        loss = criterion(output, labels)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return train_losses


def test_epoch(model, dataloader):
    model.eval()

    with torch.no_grad():
        for input, labels in dataloader:
            output = model(input)
            labels = torch.squeeze(labels, dim=1).float()
            loss = criterion(output, labels)
            test_losses.append(loss.item())
    return test_losses


for i in range(10):
    a = train_epoch(net, optimizer, train_loader)
    print(np.mean(a))
    b = test_epoch(net, test_loader)
    print(np.mean(b))
torch.save(net, "model.pth")

epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, label="Training loss")
plt.title("Training Loss over Epochs")
plt.legend()
plt.grid(True)

plt.savefig("trainingloss.png")

plt.figure(figsize=(8, 5))
plt.plot(epochs, test_losses, label="Test loss")
plt.title("Test Loss over Epochs")
plt.legend()
plt.grid(True)

plt.savefig("testloss.png")
