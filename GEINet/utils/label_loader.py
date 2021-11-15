import glob
from PIL import Image

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

  return train_label, test_label
