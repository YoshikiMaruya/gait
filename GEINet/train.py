import torch
from torch import nn, optim
from torch.optim import optimizer
from torch.utils.data import (
  DataLoader
)
import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from model import geinet as gei

train_images = ImageFolder(
  "../CASIA-B_train",
  transform=transforms.Compose([
    transforms.RandomCrop(32),
    transforms.ToTensor()
  ])
)

test_images = ImageFolder(
  "../CASIA-B_test",
  transform=transforms.Compose([
    transforms.RandomCrop(32),
    transforms.ToTensor()
  ])
)

train_loader = DataLoader(
  train_images, batch_size=32, shuffle=True
)

test_loader = DataLoader(
  test_images, batch_size=32, shuffle=True
)

gei = gei.GEINet()
print(gei)

def eval_net(GEINet, data_loader, device="cpu"):
  GEINet.eval()
  ys = []
  ypreds = []
  for x, y in data_loader:
    x = x.to(device)
    y = y.to(device)
    with torch.no_grad():
      _, y_pred = GEINet(x).max(1)
    ys.append(y)
    ypreds.append(ys)

    acc = (ys == ypreds).float().sum() / len(ys)
    return acc.item()

def train_net(GEINet, train_loader, test_loader, loss_fn=nn.CrossEntropyLoss(), n_iter=10, device="cpu"):
  train_losses = []
  train_acc = []
  val_acc = []
  optimizer = torch.optim.Adam(GEINet)
  for epoch in range(n_iter):
    running_loss = 0.0
    GEINet.train()
    n = 0
    n_acc = 0
    for i, (xx, yy) in tqdm.tqdm(enumerate(train_loader), total = len(train_loader)):
      xx = xx.to(device)
      yy = yy.to(device)
      h = GEINet(xx)
      loss = loss_fn(h, yy)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
      n += len(xx)
      _, y_pred = h.max(1)
      n_acc += (yy == y_pred).float().sum().item()

    train_losses.append(running_loss / i)
    train_acc.append(n_acc / n)
    val_acc.append(eval_net(GEINet, test_loader, device))

    print(epoch, train_losses[-1], train_acc[-1], val_acc[-1], flush=True)

# GEINet.to("cuda:0")
train_net(GEINet, train_loader, test_loader, n_iter=20, device="cpu")
