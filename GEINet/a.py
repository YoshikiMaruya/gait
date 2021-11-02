
class GEINet(nn.Module):
  def __init__(self):
    super(GEINet, self).__init__()
    self.relu = nn.ReLU()
    self.pool1 = nn.MaxPool2d(2, stride=2)
    self.conv1 = nn.Conv2d(3, 18, 7)
    self.norm1 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=1.0)
    self.pool2 = nn.MaxPool2d(2, stride=2)
    self.conv2 = nn.Conv2d(18, 45, 5)
    self.norm2 = nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75, k=1.0)

    self.fc1 = nn.Linear(45*56*76, 1024)
    self.softmax = nn.Softmax()
    self.fc2 = nn.Linear(1024, 74)

    self.dropout = nn.Dropout2d(0.5)

  def forward(self, l):
    l = self.conv1(l)
    l = self.relu(l)
    l = self.pool1(l)
    l = self.norm1(l)
    l = self.conv2(l)
    l = self.relu(l)
    l = self.pool2(l)
    l = self.norm2(l)
    l = l.view(l.size()[0], -1)
    l = self.fc1(l)
    l = self.relu(l)
    l = self.dropout(l)
    l = self.fc2(l)
    fc2_outputs = l
    l = self.softmax(l)
    return l, fc2_outputs

gei = GEINet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(gei.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.005)

train_loss_value=[]      #trainingのlossを保持するlist
train_acc_value=[]       #trainingのaccuracyを保持するlist
test_loss_value=[]       #testのlossを保持するlist
test_acc_value=[]        #testのaccuracyを保持するlist

for epoch in range(50):
  for (inputs, labels) in train_loader:
    optimizer.zero_grad()
    gei_l, gei_fc2_outputs = gei.forward(inputs)
    # outputs = gei(inputs)
    loss = criterion(gei_l, labels)
    loss.backward()
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

  sum_loss = 0.0
  sum_correct = 0
  sum_total = 0


  for (inputs, labels) in test_loader:
    optimizer.zero_grad()
    gei_l, gei_fc2_outputs = gei.forward(inputs)
    dist = [[0] * 50 for i in range(50)]

    for gallery in range(75, 125):
      feature_gallery = gei_fc2_outputs
      for probe in range(75, 125):
        feature_probe = gei_fc2_outputs
        norm_inputs = feature_gallery - feature_probe
        dist[probe][gallery] = linalg.norm(norm_inputs, )

    loss = criterion(gei_l, labels)
    sum_loss += loss.item()
    _, predicted = gei_l.max(1)
    sum_total += labels.size(0)
    sum_correct += (predicted == labels).sum().item()
  print("test  mean loss={}, accuracy={}".format(sum_loss*BATCH_SIZE/len(test_loader.dataset), float(sum_correct/sum_total)))
  test_loss_value.append(sum_loss*BATCH_SIZE/len(test_loader.dataset))
  test_acc_value.append(float(sum_correct/sum_total))

plt.figure(figsize=(6,6))      #グラフ描画用

#以下グラフ描画
plt.plot(range(EPOCH), train_loss_value)
plt.plot(range(EPOCH), test_loss_value, c='#00ff00')
plt.xlim(0, EPOCH)
plt.ylim(0, 2.5)
plt.xlabel('EPOCH')
plt.ylabel('LOSS')
plt.legend(['train loss', 'test loss'])
plt.title('loss')
plt.savefig("loss_image.png")
plt.clf()

plt.plot(range(EPOCH), train_acc_value)
plt.plot(range(EPOCH), test_acc_value, c='#00ff00')
plt.xlim(0, EPOCH)
plt.ylim(0, 1)
plt.xlabel('EPOCH')
plt.ylabel('ACCURACY')
plt.legend(['train acc', 'test acc'])
plt.title('accuracy')
plt.savefig("accuracy_image.png")
