# Yoshiki Maruya
import glob
from model.geinet import GEINet
import torch
import numpy as np

def evaluation(model):
  prove_images = list()
  gallery_images = list()

  # prove_list = ["nm-05", "nm-06"]
  # gallery_list = ["nm-01", "nm-02", "nm-03", "nm-04"]

  test_files = glob.glob('/home/yoshimaru/gait/GEINet/gei_image/test_gei/*.png')

  for file in sorted(test_files):
    file = np.array(file)
    if file[51:56] == "nm-05" or file[51:56] == "nm-06":
      file = torch.LongTensor(file)
      prove_images.append(file)
    if file[51:56] == "nm-01" or file[51:56] == "nm-02" or file[51:56] == "nm-03" or file[51:56] == "nm-04":
      file = torch.LongTensor(file)
      gallery_images.append(file)

  # prove_images = torch.LongTensor(prove_images)
  # gallery_images = torch.LongTensor(gallery_images)

  for prove_image in prove_images:
    _, prove_feature = model.forward(prove_image)
    print(prove_feature)
    # for gallery_image in sorted(gallery_images):
    #   _, gallery_feature = model.forward(gallery_image)


def main():
  geinet = GEINet()
  print(evaluation(geinet))
if __name__ == "__main__":
  main()
