import os

def label_load(img_dir):
  labels = list()
  for label in sorted(list(img_dir)):
    labels.append(label[25:28])

  return labels
def main():
  print(label_load("/home/yoshimaru/gait/GEINet/gei_image/train_gei"))
if __name__ == "__main__":
  main()
