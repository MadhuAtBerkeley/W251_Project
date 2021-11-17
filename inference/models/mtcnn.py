"""trt_mtcnn.py

This script demonstrates how to do real-time face detection with
TensorRT optimized MTCNN engine.
"""

import numpy as np
from PIL import Image


from utils.box_utils import nms, calibrate_box, get_image_boxes, convert_to_square
from models.pnet_detector import  PNetDetector
from models.rnet_detector import RNetDetector
from models.onet_detector import ONetDetector 
from utils.box_utils import show_bboxes

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class TensorRTEngine(object):
    
    def __init__(self, in_buf_size=1000, out_buf_size=1000):
        
        # allocate device memory
        self.d_input = cuda.mem_alloc(in_buf_size)
        self.d_output = cuda.mem_alloc(out_buf_size)
        self.bindings = [int(self.d_input), int(self.d_output)]
        self.stream = cuda.Stream()
        
    def run(self, batch, output, context):
        # transfer input data to device
        cuda.memcpy_htod_async(self.d_input, batch, self.stream)
        
        # execute model
        context.execute_async_v2(self.bindings, self.stream.handle, None)
        
        ## transfer predictions back
        cuda.memcpy_dtoh_async(output, self.d_output, self.stream)
        
        # syncronize threads
        self.stream.synchronize()
        
        return output
    

class MTCNNDetector(object):

    def __init__(self, max_input_size = 700*384*3*4, max_output_size = 6*345*187*4, min_face_size=40,path=None):    
    
        trt_engine = TensorRTEngine(in_buf_size=max_input_size, out_buf_size = max_output_size)    
        if path == None:
           path = '.'
        f = open(path+"/trt_engines/rnet_engine.trt", "rb")
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 
        engine_rnet = runtime.deserialize_cuda_engine(f.read())
        context_rnet = engine_rnet.create_execution_context()

        f = open(path+"/trt_engines/onet_engine.trt", "rb")
        engine_onet = runtime.deserialize_cuda_engine(f.read())
        context_onet = engine_onet.create_execution_context()

        f = open(path+"/trt_engines/pnet_engine.trt", "rb")
        engine_pnet = runtime.deserialize_cuda_engine(f.read())
        context_pnet = engine_pnet.create_execution_context()

        self.pnet_detector = PNetDetector(engine=trt_engine, context=context_pnet, min_face_size=min_face_size)

        self.rnet_detector = RNetDetector(engine=trt_engine, context=context_rnet)
        self.onet_detector = ONetDetector(engine=trt_engine, context=context_onet)

         # for probabilities
        self.thresholds = [0.6, 0.7, 0.8]
        self.img_height = 720
        self.img_width = 1280
    
    def detect_face(self, image): # = Image.open('./0008_01.jpg')

        img_w, img_h = image.size
        scale = min((self.img_height*1.0)/img_h, (self.img_width*1.0)/img_w)
        if scale < 1.0:
           image = image.resize((int(self.img_width*scale),int(self.img_height*scale)))
        
        
        output = self.pnet_detector.run_pnet(image)
        bounding_boxes = self.pnet_detector.propose_bboxes(image, output, self.thresholds[0])

        #show_bboxes(image, bounding_boxes)    
    
        img_boxes = get_image_boxes(bounding_boxes, image, size=24)
        img =  np.asarray(img_boxes, np.float16)
        img =  img[0:64]
        bounding_boxes = self.rnet_detector.run_rnet(img, self.thresholds[1], bounding_boxes)
        #show_bboxes(image, bounding_boxes)
    
        img_boxes = get_image_boxes(bounding_boxes, image, size=48)
        img =  np.asarray(img_boxes, np.float16)
        img =  img[0:64]
        bounding_boxes, landmarks = self.onet_detector.run_onet(img, self.thresholds[2], bounding_boxes)
        img = show_bboxes(image, bounding_boxes, landmarks)
        return img
        #img.show()


