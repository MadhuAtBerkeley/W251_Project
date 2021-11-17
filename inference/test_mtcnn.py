
from PIL import Image
from models.mtcnn import MTCNNDetector



image = Image.open('./0001_01.jpg')
mtcnn_detect = MTCNNDetector(path='./models')
img = mtcnn_detect.detect_face(image)
img.show()
while(1):
  i=0
