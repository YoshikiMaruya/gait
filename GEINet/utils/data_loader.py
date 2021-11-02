import os
import os.path as osp
import torchvision.transforms.functional as TF
import numpy as np

def load_data(dataset_path):
  train_data = list()
  test_data = list()

  prove_dict = {'CASIA': [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']]}
  gallery_dict = {'CASIA': ['nm-03', 'nm-04', 'nm-05', 'nm-06']}

  # for value in prove_dict.values():
  #   print(value[0][0])

  for _data in sorted(list(os.listdir(dataset_path))):
    if _data[:3] >= "075":
      test_data.append(_data)
    train_data.append(_data)

  

# def main():
#   # print(sorted(list(os.listdir("/home/yoshimaru/gait/CASIA-B_train"
#   load_data("/home/yoshimaru/gait/GEI")
# if __name__ == "__main__":
#   main()
