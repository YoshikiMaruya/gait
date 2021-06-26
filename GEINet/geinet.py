import torch
from torch import nn, optim
from torch.utils.data import (
  DataLoader
)
import tqdm
from torchvision import transforms
import numpy as np

class GEINet(nn.Module):
  def __init__(self):
      super(GEINet, self).__init__()
      self.relu = nn.ReLU()
      self.softmax = nn.Softmax()
      self.pool1 = nn.MaxPool2d(2, stride=2)
      self.conv1 = nn.Conv2d(1, 18, 7)
      self.norm1 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=1.0)
      self.pool2 = nn.MaxPool2d(3, stride=2)
      self.conv2 = nn.Conv2d(18, 45, 5)
      self.norm2 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=1.0)

      self.fc1 = nn.Linear(1024, 74)
      self.fc2 = nn.Linear(74, )

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
    l = self.fc1(l)
    l = self.relu(l)
    l = self.dropout(l)
    l = self.fc2(l)
    l = self.softmax(l)
    return l
