# ====================== 各種Import ======================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from pathlib import Path
import os, datetime

from tqdm.notebook import tqdm
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.models
from torchvision import transforms
from torchvision.datasets import ImageFolder

import cv2
import albumentations as A

# ======================
class face_detection():
# ====================== コンストラクタ ======================
	def __init__(self):
	## 各種設定
		self.now_str = datetime.datetime.strftime('%Y%m%d-%H%M')
		self.n_epochs = 20
		self.batchsize = 8
		self.learning_rate = 0.001

		self.TRAIN_DATA_PATH = "./Input/train"
		self.TEST_DATA_PATH = "./Input/test"

		self.PRINT_COUNT_PER_EPOCH = 10

		self.BASE_OUT_PATH = "./Output"
		self.MODEL_OUT_NAME = f"model_"+ self.now_str + ".pth"

# ====================== 前処理 ======================
	## データセットの画像加工用 Transform を作成
	def crete_transform(self):
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
		return transform

	## Dataset を作成
	def create_dataset(self, TRAIN_DATA_PATH, transform):
		train_dataset = ImageFolder(TRAIN_DATA_PATH, transform['Train'])
		print("train dataset:/n",train_dataset,"/n")

		datapoints = len(train_dataset)

		print("train dateset length:/n",datapoints,"/n")
		print("train dataset class to idx:/n",train_dataset.class_to_idx,"/n")

		return train_dataset

# ====================== Dataloader を作成 ======================
	## 学習データはエポックごとに各バッチの傾向が変わる（学習の傾向が変わる）ようにshuffleする
	def create_dataloader(self, train_dataset, batch_size):
		trainloader = torch.utils.data.DataLoader(train_dataset,
												batch_size = batch_size,
												shuffle=True,
												num_workers = 0)
		print("train loader:/n",trainloader,"/n")

	## classラベルをデータセットから読み取る
	def read_label_from_dataset(self, train_dataset):
		classes = [key for key in  train_dataset.class_to_idx]
		print("classes:/n",classes,"/n")

	## gpuが使える場合はgpu、そうでない場合はcpuをデバイスに指定
	def specifying_the_device_to_use(self):
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		print("use device:",device)
		return device

# ====================== 学習 ======================
	def train(self, ):
		model = torchvision.models.efficientnet_b7(pretrained = True) # efficientnet_b7を使用
		model.classifier = nn.Sequential(
					nn.Dropout(p=0.5, inplace=True),
					nn.Linear(2560, 2),
				)

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

# ====================== モデルの保存 ======================
#torch.save(model.state_dict(),'model.pth')
torch.save(model.state_dict(), BASE_OUT_PATH + MODEL_OUT_NAME)

# ======================結果の表示 ======================
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

# ====================== テスト ======================
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
