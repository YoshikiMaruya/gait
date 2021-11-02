import os
import glob
from posixpath import basename
from PIL import Image

dir = '/home/yoshimaru/gait/GEINet/gei_image'
os.makedirs(dir, exist_ok=True)

files = glob.glob('/home/yoshimaru/gait/GEI/*.png')

for file in files:
  image = Image.open(file)
  image_resize = image.resize((240, 320))
  root, ext = os.path.splitext(file)
  basename = os.path.basename(root)
  image_resize.save(os.path.join(dir, basename + ext))
