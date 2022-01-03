from PIL import Image
import torch
import numpy as np

def calc_feature(image, transform, model):
  image = Image.open(image).convert("RGB")
  image = transform(image)
  image = image.unsqueeze(0)
  # if torch.cuda.is_available():
  #   image = image.to(device)
  _, feature = model.forward(image)

  return feature
