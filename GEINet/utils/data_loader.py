import os
import glob
from PIL import Image
import cv2
import numpy as np

train_dir = '/home/yoshimaru/gait/GEINet/gei_image/train_gei'
test_dir = '/home/yoshimaru/gait/GEINet/gei_image/test_gei'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

files = glob.glob('/home/yoshimaru/gait/GEI/*.png')

for f in sorted(files):
  img = Image.open(f)
  root, ext = os.path.splitext(f)
  basename = os.path.basename(root)
  if f[25:28] < "075":
    img.save(os.path.join(train_dir, basename + ext))
  if f[25:28] >= "075":
    img.save(os.path.join(test_dir, basename + ext))
