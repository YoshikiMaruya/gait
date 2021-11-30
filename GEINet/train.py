from torch import nn, optim, linalg
from torch.optim import optimizer
from torch.utils.data import (
  DataLoader
)
import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import model.geinet
import utils.dataset
import utils.label_loader

BATCH_SIZE = 1
WEIGHT_DECAY = 0.005
LEARNING_RATE = 0.1
MOMENTUM = 0.9
EPOCH = 100

transform = transforms.Compose([transforms.Resize((240, 320)), transforms.ToTensor()])
train_label, test_label = utils.label_loader.label_load()
train_label = train_label - 1
test_label = test_label - 1

train_dataset = utils.dataset.GEIData("/home/yoshimaru/gait/GEINet/gei_image/train_gei", train_label, transform)
test_dataset = utils.dataset.GEIData("/home/yoshimaru/gait/GEINet/gei_image/test_gei", test_label, transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

gei = model.geinet.GEINet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(gei.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

train_loss_value=[]      #trainingのlossを保持するlist
train_acc_value=[]       #trainingのaccuracyを保持するlist
test_loss_value=[]       #testのlossを保持するlist
test_acc_value=[]        #testのaccuracyを保持するlist

for (inputs, labels) in train_loader:
  optimizer.zero_grad()
  gei_l, gei_feature = gei.forward(inputs)
  # print(inputs)
  _, predicted = gei_l.max(1)
  print(gei_l.max(1))
  print(gei_l)


# for epoch in range(EPOCH):
#   for (input, label) in train_loader:
#     optimizer.zero_grad()
#     gei_l, gei_feature = gei.forward(input)
#     loss = criterion(gei_l, label)
#     optimizer.step()

#   sum_loss = 0.0          #lossの合計
#   sum_correct = 0         #正解率の合計
#   sum_total = 0           #dataの数の合計

#   for (input, label) in train_loader:
#     optimizer.zero_grad()
#     gei_l, gei_fc2_outputs = gei.forward(input)
#     loss = criterion(gei_l, label)
#     sum_loss += loss.item()                            #lossを足していく
#     _, predicted = gei_l.max(1)                      #出力の最大値の添字(予想位置)を取得
#     sum_total += label.size(0)                        #labelの数を足していくことでデータの総和を取る
#     sum_correct += (predicted == label).sum().item()  #予想位置と実際の正解を比べ,正解している数だけ足す
#   print("train mean loss={}, accuracy={}".format(sum_loss*BATCH_SIZE/len(train_loader.dataset), float(sum_correct/sum_total)))  #lossとaccuracy出力
#   train_loss_value.append(sum_loss*BATCH_SIZE/len(train_loader.dataset))  #traindataのlossをグラフ描画のためにlistに保持
#   train_acc_value.append(float(sum_correct/sum_total))   #traindataのaccuracyをグラフ描画のためにlistに保持

  # sum_loss = 0.0
  # sum_correct = 0
  # sum_total = 0
