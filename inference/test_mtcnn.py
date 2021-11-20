
from PIL import Image
from models.mtcnn_rtt import MTCNNDetector
import glob
import os
import pycuda.driver as cuda
import numpy as np
anchor_emb_dict = {}


def recognize_target(target_emb):
    min_dist = 100000
    anchor = None
    for key, value in anchor_emb_dict.items():
       
        dist = np.linalg.norm(target_emb-value)
        #print(np.sum(target_emb), np.sum(value), dist)
        if dist < min_dist:
           min_dist = dist
           anchor = key
    return anchor

def generate_anchors(mtcnn_rtt, path=None):
    for filename in glob.glob(path, recursive = True):
        with open(filename, 'r') as f:
           anchor = str(filename.split('/')[-2])   
           #print(anchor)
           image= Image.open(filename)
           image = mtcnn_rtt.detect_face(image)
           emb_vect = mtcnn_rtt.face_vector(image)[0]
           anchor_emb_dict[anchor] = emb_vect 
    return

def test_siamese_network(mtcnn_rtt, path=None):
    count = 0
    good_count = 0
    for filename in glob.glob(path, recursive = True):
        count = count + 1
        with open(filename, 'r') as f:
           anchor = str(filename.split('/')[-2]) 
           image= Image.open(filename)
           image = mtcnn_rtt.detect_face(image)
           if image == None:
              continue
           target_emb = mtcnn_rtt.face_vector(image)[0]
           predict_target = recognize_target(target_emb)
           #print(predict_target, anchor)
           if predict_target == anchor:
              good_count = good_count + 1 
           
    print("Accuracy :{}".format(count*1.0/good_count))
    return
   

def main():
  
   mtcnn_detect = MTCNNDetector(path='./models')

   #Generate reference
   generate_anchors(mtcnn_detect, path='../data/test_anchors/**/*.jpg')
  
  
   #check performance/accuracy
   test_siamese_network(mtcnn_detect, path='../data/test_no_mask/n000001/*.jpg')
   
   

   return

if __name__ == "__main__":
   main()






