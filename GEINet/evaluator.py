# Yoshiki Maruya
import glob
from PIL import Image
from numpy.random import f
from model.geinet import GEINet
from torch import nn
import numpy as np
from torchvision import transforms
import torch

DIR = '/home/yoshimaru/gait/GEINet/gei_image/test_gei/'

def evaluation(model):
  probe_images = list()
  gallery_images = list()

  # prove_list = [["nm-05", "nm-06"], ["bg-01", "bg-02"], ["cl-01","cl-02"]]
  # gallery_list = ["nm-01", "nm-02", "nm-03", "nm-04"]

  test_files = glob.glob(DIR + '*.png')
  transform = transforms.Compose([transforms.Resize((240, 320)), transforms.ToTensor()])

  for file in sorted(test_files):
    if file[51:56] == "nm-05" or file[51:56] == "nm-06":
      probe_images.append(file)
    if file[51:56] == "nm-01" or file[51:56] == "nm-02" or file[51:56] == "nm-03" or file[51:56] == "nm-04":
      gallery_images.append(file)

# view [57:60]
  # search the missing data
  # for i in range(75,125):
  #   count = 0
  #   for g in gallery_images:
  #     if "nm-04" == g[51:56] and i == int(g[47:50]):
  #       count += 1
  #   if count != 11:
  #     print(i)


  dist = []
  view_list = ["000", "018", "036", "054", "072", "090", "108", "126", "144", "162", "180"]

  # recognition part
  for i, probe_image in enumerate(probe_images, 75):
    print(f'{i}th start')
    probe_image = Image.open(probe_image).convert('RGB')
    probe_image = transform(probe_image)
    probe_image = probe_image.unsqueeze(0)
    _, probe_feature = model.forward(probe_image)
    for j, gallery_image in enumerate(gallery_images, 75):
      id = gallery_image[47:50]
      if not int(id) == 75:
        break
      gallery_image = Image.open(gallery_image).convert('RGB')
      gallery_image = transform(gallery_image)
      gallery_image = gallery_image.unsqueeze(0)
      _, gallery_feature = model.forward(gallery_image)
      squ_diff_list = np.array(torch.square(probe_feature - gallery_feature).tolist())
      dist.append(np.sqrt(np.sum(np.sum(squ_diff_list))))
    break
  return np.array(dist)
def main():
  geinet = GEINet()
  eval = evaluation(geinet)
  print(eval)
if __name__ == "__main__":
  main()
