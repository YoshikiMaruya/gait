import glob
import numpy as np
from PIL import Image
import torch

def label_load():
  train_label = list()
  test_label = list()

  files = glob.glob('/home/yoshimaru/gait/GEI/*.png')

  for f in sorted(files):
    img = Image.open(f)
    if f[25:28] < "075":
      train_label.append(f[25:28])
    if f[25:28] >= "075":
      test_label.append(f[25:28])

  train_label = np.array(train_label, dtype=np.float64)
  test_label = np.array(test_label, dtype=np.float64)

  train_label = torch.from_numpy(train_label).long()
  test_label = torch.from_numpy(test_label).long()

  return train_label, test_label
