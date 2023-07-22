#!/usr/bin/env python
# coding: utf-8

# # **1.** **各種インポート**

# In[ ]:


import numpy as np

import matplotlib.pyplot as plt #グラフ表示など用ライブラリ
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator #軸の表示設定用
# ノートブックでmatplotを使う時にはinline指定をした方が良い
get_ipython().run_line_magic('matplotlib', 'inline')

from pathlib import Path # pathの扱いが便利になるライブラリ

from tqdm.notebook import tqdm # プログレスバー表示用
from tqdm import tqdm

import torch #PyTorch
import torch.nn as nn #PyTorchのニューラルネットワークに関するもの
import torch.optim as optim #最適化関数利用のため
import torch.nn.functional as F

import torchvision # PyTorchの画像に関するものをまとめたもの
import torchvision.models
from torchvision import transforms #データセットの画像加工用
from torchvision.datasets import ImageFolder #ImageFolderクラス利用のため

import cv2
import albumentations as A


# # **2.** **各種設定**

# In[ ]:


#@title #Hyper Parameters
n_epochs =  20#@param {type:"integer"}
batchsize =  8#@param {type:"integer"}
learning_rate = 0.001 #@param {type:"raw"}


# In[ ]:


#@title #Input/Output
#@markdown ### trainデータセットのパス:
#TRAIN_DATA_PATH = "/content/ManWoman-dataset/dataset/Train"#@param{type:"string"}
TRAIN_DATA_PATH = "C:/Users/yutam/Desktop/Data_analysis/FaceDetection/ManWoman-dataset/Train"

#@markdown ### testデータセットのパス:
#TEST_DATA_PATH = "/content/ManWoman-dataset/dataset/Test"#@param{type:"string"}
TEST_DATA_PATH ="C:/Users/yutam/Desktop/Data_analysis/FaceDetection/Face_dateset/final_files"

#@markdown ---

#@markdown ### 1エポックごとに何回ログをprintするか:
PRINT_COUNT_PER_EPOCH =   10#@param{type:"integer"}

#@markdown ### 出力結果を保存する基本的なパス:
#BASE_OUT_PATH = '/content/drive/MyDrive/grad-research/' #@param{type:"string"}
BASE_OUT_PATH ='C:/Users/yutam/Desktop/Data_analysis/FaceDetection/Output_Models'

#BASE_OUT_PATH = '/content/drive/Shareddrives/HOSONOLAB_2021/20220126_データセット/'
#@markdown ### 学習したモデルを保存する名前:
MODEL_OUT_NAME = 'model.pth' #@param{type:"string"}


# # **3.** **前処理**

# In[ ]:


# #データセットの画像加工用 Transform を作成
# transform = {
#     'Train':transforms.Compose(
#     [transforms.Resize((224,224)), # 画像サイズを一定にする
#      transforms.ToTensor()  # NNで計算しやすい様に画像を変換
#      ]),
#      'Test':transforms.Compose(
#     [transforms.Resize((224,224)), # 画像サイズを一定にする
#      transforms.ToTensor()  # NNで計算しやすい様に画像を変換
#      ])
# }
#データセットの画像加工用 Transform を作成
transform = {
    'Train':transforms.Compose(
    [transforms.Resize(226), # 画像サイズを一定にする
     transforms.CenterCrop(224),
     transforms.ToTensor()  # NNで計算しやすい様に画像を変換
     ]),
     'Test':transforms.Compose(
    [transforms.Resize(226), # 画像サイズを一定にする
     transforms.CenterCrop(224),
     transforms.ToTensor()  # NNで計算しやすい様に画像を変換
     ])
}


# Dataset を作成
train_dataset = ImageFolder(TRAIN_DATA_PATH, transform['Train'])
print("train dataset:/n",train_dataset,"/n")

datapoints = len(train_dataset)

print("train dateset length:/n",datapoints,"/n")
print("train dataset class to idx:/n",train_dataset.class_to_idx,"/n")

# Dataloader を作成
# 学習データはエポックごとに各バッチの傾向が変わる（学習の傾向が変わる）ようにshuffleする
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = batchsize,
                                          shuffle=True, num_workers = 0)
print("train loader:/n",trainloader,"/n")

# classラベルをデータセットから読み取る
classes = [key for key in  train_dataset.class_to_idx]
print("classes:/n",classes,"/n") 

#gpuが使える場合はgpu、そうでない場合はcpuをデバイスに指定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
print("use device:",device)


# # **4.** **学習**

# In[ ]:


#インスタンス生成
#model = MyModel()
#model = VGG(n_classes = 2)
#model = torchvision.models.densenet121()
#model = torchvision.models.efficientnet_b7(pretrained = True, num_classes = 2)

model = torchvision.models.efficientnet_b7(pretrained = True) # efficientnet_b7を使用
#print("model:/n",model,"/n")
model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(2560, 2),
        )

#model.classifier = nn.Linear(2208,2)

print("model:/n",model,"/n")
# 学習モードに切り替える
model.train()
# モデルをdeviceに送る。CPUで動かしたい時はやらなくても良い。
model.to(device) 
#optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9) #最適化関数
optimizer = optim.Adam(model.parameters(), lr = learning_rate,betas=(0.9,0.999)) #最適化関数にはAdamを使用
print("optimizer:/n",optimizer,"/n")

criterion = nn.CrossEntropyLoss() #損失関数-CrossEntropyLoss
print("criterion:/n",criterion,"/n")

# 各エポックで最後にprintした値をグラフ表示用に格納しておくためのもの
results_train = {'loss': [],'accuracy': []}

# 1エポックごとの反復回数（iteration）
iteration = datapoints / batchsize
# 何iterationごとにprintするか
print_iteration = iteration // PRINT_COUNT_PER_EPOCH

print('[epoch, iteration]')

# training loop
for epoch in range(n_epochs):
  #list for loss and accuracy

  running_loss = 0.0
  running_accuracy = 0.0
  
  # 新規エポックごとにリストに要素追加
  results_train['loss'].append(running_loss)
  results_train['accuracy'].append(running_accuracy)

  for i, data in enumerate(trainloader, 0):
    images, labels = data
    images = images.to(device)
    labels = labels.to(device)
    # forwardを実行
    # 特殊なメソッドなのでmodel.forward()のように書かなくてもmodel()で実行されるようになっている
    outputs = model(images)

    optimizer.zero_grad()
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # calculate print loss and accuracy
    # lossの加算
    running_loss += loss.item()

    # outputsの各バッチで何番目のクラスの確率が最大かをpredictedに格納
    _, predicted = torch.max(outputs.data, 1)
    # predictedとlabelsが一致する個数が予測に正解している数correct
    correct = (predicted == labels).sum()
    # batchsizeで割ることで精度にし、%にする
    accuracy = 100 * correct / batchsize
    # accuracyの加算
    running_accuracy += accuracy.item()
    # print_iterationごとのprint
    if i % print_iteration == print_iteration - 1:
        # 加算したlossとaccuracyを反復数で割る
        # 一番最後のprintした値で常に上書きすれば、各epohで最後にprintした値が取得できる
        results_train['loss'][epoch] = running_loss / print_iteration
        results_train['accuracy'][epoch] = running_accuracy / print_iteration
        
        print('[%5d, %9d]  loss: %.3f,  accuracy: %.3f' %
              (epoch + 1, i + 1, results_train['loss'][epoch],results_train['accuracy'][epoch] ))
        
        # print iteration毎に変数初期化
        running_loss = 0.0
        running_accuracy = 0.0

#モデルの保存

#torch.save(model.state_dict(),'model.pth')
torch.save(model.state_dict(), BASE_OUT_PATH + MODEL_OUT_NAME)


# # **5.** **結果の表示**

# In[ ]:


# 台紙を作成
# facecolorはグラフ全体の背景色を設定
# dpiで解像度が変わる
fig = plt.figure(figsize=(6.4, 4.8), dpi=200, facecolor='w')

# 上下に2つのグラフを用意
# ylim=(0, 100)のように引数を指定すれば表示範囲が0～100になる
# 見やすい範囲については各自で考える
axT = fig.add_subplot(211,xlabel='epoch',ylabel='loss')#2行1列の1番目
axB = fig.add_subplot(212,xlabel='epoch',ylabel='accuracy')#2行1列の2番目

# x軸の要素は、今回epochなので指定しなくても良いが念のため
epochs = range(n_epochs)

# epochは整数なので整数表示のためのオプション
# この行を実行しないとx軸が実数表示になるはず
axT.xaxis.set_major_locator(MaxNLocator(integer=True))
axB.xaxis.set_major_locator(MaxNLocator(integer=True))

# プロット
axT.plot(epochs, results_train['loss'])
axB.plot(epochs, results_train['accuracy'])

# 軸ラベルと図が被ることを防止
fig.tight_layout()
# 画像として保存
fig.savefig(BASE_OUT_PATH + 'loss_acc.png', facecolor=fig.get_facecolor())


# # **6.** **テスト**

# In[ ]:


#@markdown ### テストに使用する学習済みモデルを読み込むパス:
#MODEL_LOAD_PATH =  '/content/drive/My Drive/grad-research/model.pth' #@param{type:"string"}
MODEL_LOAD_PATH = ''


# In[ ]:


test_dataset = ImageFolder(TEST_DATA_PATH, transform['Test'])
print("test dataset class to idx:/n",test_dataset.class_to_idx,"/n")
test_samples = len(test_dataset)
print("test samples:/n",test_samples,"/n")

# classラベルをデータセットから読み取る
classes = [key for key in  test_dataset.class_to_idx]
print("classes:/n",classes,"/n") 

test_batchsize = 5
# テストデータは全部のデータに同じことをするだけなので普通はsuffleしない
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batchsize,
                                         shuffle=False, num_workers = 2)
print("test loader:/n",testloader,"/n")

