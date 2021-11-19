
from PIL import Image
from models.mtcnn_rtt import MTCNNDetector



image1= Image.open('./0001_01.jpg')
mtcnn_detect = MTCNNDetector(path='./models')
img1 = mtcnn_detect.detect_face(image1)
img_emb1 = mtcnn_detect.face_vector(img1)
image2= Image.open('./0017_01.jpg')
img2 = mtcnn_detect.detect_face(image2)
#img_emb2 = mtcnn_detect.face_vector(img2)
print(img2.size)
img2.show()

while(1):
  i=0
