from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import os.path as osp

class GEIData(Dataset):
  IMG_EXTENSIONS = [".png", ".jpg"]

  def __init__(self, img_dir, label, transform=None):
    self.img_paths = self._get_img_paths(img_dir)
    self.label = label
    self.transform = transform

  def __getitem__(self, index):
    path = self.img_paths[index]
    label = self.label[index]

    img = Image.open(path)
    img = img.convert("RGB")

    if self.transform is not None:
      img = self.transform(img)
      # label = self.transform(label)

    return img, label

  def _get_img_paths(self, img_dir):

    img_dir = Path(img_dir)
    img_paths = [p for p in img_dir.iterdir() if p.suffix in GEIData.IMG_EXTENSIONS]

    return img_paths

  def __len__(self):
    return len(self.img_paths)
