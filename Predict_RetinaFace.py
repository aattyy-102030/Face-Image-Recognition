from retinaface import RetinaFace
import torch
# import albumentations as A
import cv2

from Create_LearnedModel import face_detection
import os, glob, datetime
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class predict_retinaface():
	def __init__(self) -> None:
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.retinaface_model = RetinaFace
		# self.sample_img = "Sample_Image.jpg"
		self.files = os.listdir("./Sample_Image")
		self.sample_imgs = [os.path.join("./Sample_Image", file) for file in self.files if file.lower().endswith(('.jpg', '.png'))]
		self.now_str = datetime.datetime.now()

	def check_output_dir(self):
		# instance = face_detection()
		# self.learned_model_path = instance.main()
		self.learned_model_path = "model_20240613-2154.pth"
		print(f"Output model path : {self.learned_model_path}")

	def detect_faces(self, img) -> dict:
		faces = self.retinaface_model.detect_faces(img)
		return faces

	def load_learned_model(self):
		learned_model = torch.load(self.learned_model_path)
		learned_model.eval()
		learned_model.to(self.device)
		return learned_model

	def annotate_faces(self, faces, img, learned_model):	# -> ndarray
		image = cv2.imread(img)
		for face in faces.values():
			# 顔の座標を取得
			x1, y1, x2, y2 = face['facial_area']
			face_img = image[y1:y2, x1:x2]
			face_img = cv2.resize(face_img, (224, 224))  # モデルの入力サイズに合わせる
			tensor = torch.from_numpy(face_img).permute(2, 0, 1).unsqueeze(0).float()

			# 性別判定
			with torch.no_grad():
				gender = learned_model(tensor)

			# 矩形の色を設定
			color = (255, 0, 0) if gender > 0.5 else (0, 0, 255)  # 男性なら青、女性なら赤

			# 画像に矩形を描画
			cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

		return image

	def main(self):
		imgs = self.sample_imgs
		print("INFO: START prediction by RetinaFace. ")
		for i, img in enumerate(imgs,start=1):
			self.check_output_dir()
			faces = self.detect_faces(img)
			learned_model = self.load_learned_model()
			result_image = self.annotate_faces(faces, img, learned_model)
			cv2.imwrite(f'Sample_result_'+ i +'.jpg', result_image)

if __name__ == "__main__":
	obj = predict_retinaface()
	obj.main()
	print("All processing has finished !")
