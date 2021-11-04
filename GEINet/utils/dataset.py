from pathlib import Path
from torch.utils.data import DataLoader, Dataset, dataloader
from torchvision import transforms
from PIL import Image
from torchvision.datasets.folder import IMG_EXTENSIONS, ImageFolder

class GEIData(Dataset):
  IMG_EXTENSIONS = [".png", ".jpg"]

  def __init__(self, img_dir, transform=None):
    self.img_paths = self._get_img_paths(img_dir)
    self.transform = transform

  def __getitem__(self, index):
    path = self.img_paths[index]

    img = Image.open(path)

    if self.transform is not None:
      img = self.transform(img)

    return img

  def _get_img_paths(self, img_dir):

    img_dir = Path(img_dir)
    img_paths = [p for p in img_dir.iterdir() if p.suffix in GEIData.IMG_EXTENSIONS]

    return img_paths

  def __len__(self):
    return len(self.img_paths)


# transform = transforms.Compose([transforms.Resize((240, 320)), transforms.ToTensor()])

# dataset = GEIData("/home/yoshimaru/gait/GEI", transform)

# dataloader = DataLoader(dataset, batch_size=20)

# for batch in dataloader:
#   print(batch.shape)
