from torch import nn, optim, linalg
from torch.optim import optimizer
from torch.utils.data import (
  DataLoader
)
import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import model.geinet
import utils.data_loader as train_data

BATCH_SIZE = 31
WEIGHT_DECAY = 0.005
LEARNING_RATE = 0.0001
EPOCH = 100

train_images = ImageFolder(
  train_data.load_data,
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

gei = model.geinet.GEINet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(gei.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.005)

print(train_images)

# for (inputs, labels) in train_loader:
#   optimizer.zero_grad()
#   gei_l, gei_feature = gei.forward(inputs)
#   print(gei_feature.size())
