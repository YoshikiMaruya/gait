from cv2 import resize
import torch
from torch import nn, optim, linalg
from torch.optim import optimizer
from torch.utils.data import (
  DataLoader
)
from torchvision.transforms.transforms import Resize
import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt    #グラフ出力用module

BATCH_SIZE = 31
WEIGHT_DECAY = 0.005
LEARNING_RATE = 0.0001
EPOCH = 100

train_images = ImageFolder(
  "../CASIA-B_train",
  transform=transforms.Compose([
    transforms.Resize((240, 320)),
    transforms.ToTensor()
  ])
)

test_images = ImageFolder(
  "../CASIA-B_test",
  transform=transforms.Compose([
    transforms.Resize((240, 320)),
    transforms.ToTensor()
  ])
)

train_loader = DataLoader(
  train_images, batch_size=BATCH_SIZE, shuffle=True
)

test_loader = DataLoader(
  test_images, batch_size=BATCH_SIZE, shuffle=False
)


class GEINet(nn.Module):
  def __init__(self):
    super(GEINet, self).__init__()
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax()
    self.pool1 = nn.MaxPool2d(2, stride=2)
    self.conv1 = nn.Conv2d(3, 18, 7)
    self.norm1 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=1.0)
    self.pool2 = nn.MaxPool2d(2, stride=2)
    self.conv2 = nn.Conv2d(18, 45, 5)
    self.norm2 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=1.0)

    self.fc1 = nn.Linear(45*56*76, 1024)
    self.fc2 = nn.Linear(1024, 74)

    self.dropout = nn.Dropout2d(0.5)

  def forward(self, l):
    l = self.conv1(l)
    l = self.relu(l)
    l = self.pool1(l)
    l = self.norm1(l)
    l = self.conv2(l)
    l = self.relu(l)
    l = self.pool2(l)
    l = self.norm2(l)
    l = l.view(l.size()[0], -1)
    l = self.fc1(l)
    l = self.relu(l)
    l = self.dropout(l)
    l = self.fc2(l)
    fc2_outputs = l
    l = self.softmax(l)
    return l, fc2_outputs

# gei_l, gei_fc2_outputs = GEINet()
# print(gei_l.forward())

gei = GEINet()


# for i in range(3):
#   for (inputs, labels) in test_loader:
#     print(inputs.size())
#     gei_l, gei_fc2_outputs = gei.forward(inputs)
#     print(gei_fc2_outputs.size())
#     print(labels)

#     dist = [[0] * 50 for i in range(50)]

#     for gallery in range(75, 125):
#       feature_gallery = gei_fc2_outputs
#       for probe in range(75, 125):
#         feature_probe = gei_fc2_outputs
#         norm_inputs = feature_gallery - feature_probe
#         print(norm_inputs.size())
#         dist[probe][gallery] = linalg.norm(norm_inputs, 1)
