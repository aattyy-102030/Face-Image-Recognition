#from retinaface import RetinaFace
from Create_LearnedModel import face_detection
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class predict_retinaface():
	def __init__(self) -> None:
		pass

	def check_output_dir():
		instance = face_detection()
		received_str = instance.main()
		print(f"Output model path : {received_str}")



if __name__ == "__main__":
	obj = predict_retinaface()
	obj.check_output_dir()
	print("All processing has finished !")
