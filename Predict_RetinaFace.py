import os, datetime
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from retinaface import RetinaFace
import torch
import torch.nn as nn
from torchvision import transforms
import albumentations as A
import cv2
import numpy as np

from Create_LearnedModel import face_detection, torchvision

class predict_retinaface():
	def __init__(self) -> None:
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.retinaface_model = RetinaFace
		self.files = os.listdir("./Sample_Image")
		self.sample_imgs = [os.path.join("./Sample_Image", file) for file in self.files if file.lower().endswith(('.jpg', '.png'))]
		self.now_str = datetime.datetime.now()

	def check_output_dir(self):
		instance = face_detection()
		self.learned_model_path = instance.main()
		print(f"Output model path : {self.learned_model_path}")

	def detect_faces(self, img):
		faces = self.retinaface_model.extract_faces(img)
		annotations = self.retinaface_model.detect_faces(img)
		return faces, annotations

	def load_learned_model(self):
		learned_model = torch.load(self.learned_model_path)
		model = torchvision.models.efficientnet_b7(pretrained = False)
		model.classifier = nn.Sequential(
			nn.Dropout(p=0.5, inplace=True),
			nn.Linear(2560, 2)      # 2クラス分類の最終層
		)
		model.load_state_dict(learned_model)
		model.eval()
		model.to(self.device)
		return model

	def annotate_faces(self, faces, annotations, img, model):	# -> ndarray
		image = cv2.imread(img)
		vis_image = image.copy()
		totensor = transforms.ToTensor()
		transform2 = A.Compose([A.SmallestMaxSize(max_size=226, p=1),
                        A.CenterCrop(height=224, width=224, p=1)
                        ])
		predictions = []
		for face in faces:
			# 顔の座標を取得
			face_img = transform2(image=face)['image']  # モデルの入力サイズに合わせる
			model_input = totensor(face_img)
			model_input = model_input.to(self.device)

			outputs = model(model_input.unsqueeze(0))
			predictions += [torch.argmax(outputs, 1)]

		prediction_id = 0
		for _, annotation in annotations.items():
			is_man = predictions[prediction_id]
			if is_man:
				color = (0, 0, 255) # blue
				text = "woman"
			else:
				color = (255, 0, 0) # red
				text = "man"

			facial_area = annotation["facial_area"]
			x_min = facial_area[0]
			y_min = facial_area[1]
			x_max = facial_area[2]
			y_max = facial_area[3]

			x_min = np.clip(x_min, 0, x_max - 1)
			y_min = np.clip(y_min, 0, y_max - 1)

			vis_image = cv2.rectangle(vis_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=color, thickness=2)
			vis_image = cv2.putText(vis_image, text, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX,  1, color, 2, cv2.LINE_AA)
			prediction_id += 1
		return vis_image

	def main(self):
		imgs = self.sample_imgs
		print("INFO: START prediction by RetinaFace. ")
		for i, img in enumerate(imgs,start=1):
			self.check_output_dir()
			faces, annotations = self.detect_faces(img)
			model = self.load_learned_model()
			result_image = self.annotate_faces(faces, annotations, img, model)
			cv2.imwrite(f'Sample_result_'+ str(i) +'.jpg', result_image)

if __name__ == "__main__":
	obj = predict_retinaface()
	obj.main()
	print("All processing has finished !")
