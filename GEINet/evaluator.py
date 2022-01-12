# Yoshiki Maruya
import glob
from types import DynamicClassAttribute
from PIL import Image
from numpy.random import f
from model.geinet import GEINet
from torch import nn
import numpy as np
from torchvision import transforms
import torch
from tool import calc_feature

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


  # view_list = ["000", "018", "036", "054", "072", "090", "108", "126", "144", "162", "180"]
  ss = []
  so = []
  os = []
  oo = []

  # recognition part
  for i, probe_image in enumerate(probe_images, 75):
    print(f'{i}th start')
    same_id_same_view = []
    same_id_other_view = []
    other_id_same_view = []
    other_id_other_view = []
    p_id = probe_image[47:50]
    p_view = probe_image[57:60]
    if i == 77:
      break
    probe_feature = calc_feature(probe_image, transform, model)
    for j, gallery_image in enumerate(gallery_images, 75):
      g_id = gallery_image[47:50]
      g_view = gallery_image[57:60]
      if int(g_id) == 80:
        break
      gallery_feature = calc_feature(gallery_image, transform, model)
      squ_diff_list = np.array(torch.square(probe_feature - gallery_feature).tolist())
      dist = np.sqrt(np.sum(np.sum(squ_diff_list)))
      if p_id == g_id and p_view == g_view:
        print(p_id, g_id, p_view, g_view)
        same_id_same_view.append(dist)
      if p_id == g_id and p_view != g_view:
        print(p_id, g_id, p_view, g_view)
        same_id_other_view.append(dist)
      if p_id != g_id and p_view == g_view:
        print(p_id, g_id, p_view, g_view)
        other_id_same_view.append(dist)
      if p_id != g_id and p_view != g_view:
        print(p_id, g_id, p_view, g_view)
        other_id_other_view.append(dist)
    ss.append(same_id_same_view)
    so.append(same_id_other_view)
    os.append(other_id_same_view)
    oo.append(other_id_other_view)
  return np.array(ss), np.array(so), np.array(os), np.array(oo)
def main():
  geinet = GEINet()
  same_id_same_view, same_id_other_view, other_id_same_view, other_id_other_view = evaluation(geinet)
  np.save(
    "same_id_same_view",
    same_id_same_view,
    fix_imports = True
  )
  np.save(
    "same_id_other_view",
    same_id_other_view,
    fix_imports = True
  )
  np.save(
    "other_id_same_view",
    other_id_same_view,
    fix_imports = True
  )
  np.save(
    "other_id_other_view",
    other_id_other_view,
    fix_imports = True
  )

  print("----------------samesame----------------")
  print(same_id_same_view)
  print("----------------sameother----------------")
  print(same_id_other_view)
  print("----------------othersame----------------")
  print(other_id_same_view)
  print("----------------otherother----------------")
  print(other_id_other_view)
  print(len(same_id_same_view), len(same_id_other_view))
  print(len(other_id_same_view), len(other_id_other_view))
if __name__ == "__main__":
  main()
