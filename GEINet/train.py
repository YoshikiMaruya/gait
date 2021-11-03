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
import utils.data_loader as train_data

BATCH_SIZE = 20
WEIGHT_DECAY = 0.005
LEARNING_RATE = 0.0001
EPOCH = 100

train_images = ImageFolder(
  "../GEI",
  transform=transforms.Compose([
    transforms.Resize((240, 320)),
    transforms.ToTensor()
  ])
)

test_images = ImageFolder(
  "../GEI",
  transform=transforms.Compose([
    transforms.Resize((240, 320)),
    transforms.ToTensor()
  ])
)

train_loader = DataLoader(
  train_images, batch_size=BATCH_SIZE, shuffle=True
)

test_loader = DataLoader(
  test_images, batch_size=BATCH_SIZE, shuffle=False
)

gei = model.geinet.GEINet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(gei.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.005)

train_loss_value=[]      #trainingのlossを保持するlist
train_acc_value=[]       #trainingのaccuracyを保持するlist
test_loss_value=[]       #testのlossを保持するlist
test_acc_value=[]        #testのaccuracyを保持するlist

for epoch in range(EPOCH):
  for (input, label) in train_loader:
    optimizer.zero_grad()
    gei_l, gei_feature = gei.forward()
    outputs = gei_l(input)
    loss = criterion(outputs, label)
    optimizer.step()

  sum_loss = 0.0          #lossの合計
  sum_correct = 0         #正解率の合計
  sum_total = 0           #dataの数の合計

  for (inputs, labels) in train_loader:
    optimizer.zero_grad()
    gei_l, gei_fc2_outputs = gei.forward(inputs)
    print(labels)
    loss = criterion(gei_l, labels)
    sum_loss += loss.item()                            #lossを足していく
    _, predicted = gei_l.max(1)                      #出力の最大値の添字(予想位置)を取得
    sum_total += labels.size(0)                        #labelの数を足していくことでデータの総和を取る
    sum_correct += (predicted == labels).sum().item()  #予想位置と実際の正解を比べ,正解している数だけ足す
  print("train mean loss={}, accuracy={}".format(sum_loss*BATCH_SIZE/len(train_loader.dataset), float(sum_correct/sum_total)))  #lossとaccuracy出力
  train_loss_value.append(sum_loss*BATCH_SIZE/len(train_loader.dataset))  #traindataのlossをグラフ描画のためにlistに保持
  train_acc_value.append(float(sum_correct/sum_total))   #traindataのaccuracyをグラフ描画のためにlistに保持

  # sum_loss = 0.0
  # sum_correct = 0
  # sum_total = 0
