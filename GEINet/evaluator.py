# Yoshiki Maruya
import glob
from PIL import Image
from model.geinet import GEINet
from torch import nn
import numpy as np
from torchvision import transforms

DIR = '/home/yoshimaru/gait/GEINet/gei_image/test_gei/'

def evaluation(model):
  prove_images = list()
  gallery_images = list()

  # prove_list = [["nm-05", "nm-06"], ["bg-01", "bg-02"], ["cl-01","cl-02"]]
  # gallery_list = ["nm-01", "nm-02", "nm-03", "nm-04"]

  test_files = glob.glob(DIR + '*.png')
  transform = transforms.Compose([transforms.Resize((240, 320)), transforms.ToTensor()])

  for file in sorted(test_files):
    if file[51:56] == "nm-05" or file[51:56] == "nm-06":
      prove_images.append(file)
    if file[51:56] == "nm-01" or file[51:56] == "nm-02" or file[51:56] == "nm-03" or file[51:56] == "nm-04":
      gallery_images.append(file)

  print(len(prove_images) * len(gallery_images))
  dist = []
  # recognition part
  for i, prove_image in enumerate(prove_images):
    prove_image = Image.open(prove_image).convert('RGB')
    prove_image = transform(prove_image)
    prove_image = prove_image.unsqueeze(0)
    _, prove_feature = model.forward(prove_image)
    for j, gallery_image in enumerate(gallery_images):
      gallery_image = Image.open(gallery_image).convert('RGB')
      gallery_image = transform(gallery_image)
      gallery_image = gallery_image.unsqueeze(0)
      _, gallery_feature = model.forward(gallery_image)
      # dist.append(abs((sum(sum(np.array((prove_feature.tolist())))) - sum(sum(np.array((gallery_feature.tolist())))))/74.0))
      break
    break

  return dist
def main():
  geinet = GEINet()
  eval = evaluation(geinet)
  print(eval)
if __name__ == "__main__":
  main()
