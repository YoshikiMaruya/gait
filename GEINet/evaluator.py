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

def evaluation(data, config):
  dataset = config['dataset']
  # feature, view, labal
