"""trt_mtcnn.py

This script demonstrates how to do real-time face detection with
TensorRT optimized MTCNN engine.
"""

import numpy as np
from PIL import Image
import os


from utils.box_utils import nms, calibrate_box, get_image_boxes, convert_to_square
from models.pnet_detector import  PNetDetector
from models.rnet_detector import RNetDetector
from models.onet_detector import ONetDetector 
from models.face_recognizer import FaceRecognizer 
from utils.box_utils import show_bboxes

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


class TensorRTEngine(object):
    
    def __init__(self, in_buf_size=1000, out_buf_size=1000):
        
        #self.cuda_ctx = cuda.Device(0).make_context()
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
    #def release(self):
        #self.cuda_ctx.pop()
        #del self.cuda_ctx
    

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

        f = open(path+"/trt_engines/resnet_engine.trt", "rb")
        engine_resnet = runtime.deserialize_cuda_engine(f.read())
        context_resnet = engine_resnet.create_execution_context()

        self.pnet_detector = PNetDetector(engine=trt_engine, context=context_pnet, min_face_size=min_face_size)

        self.rnet_detector = RNetDetector(engine=trt_engine, context=context_rnet)
        self.onet_detector = ONetDetector(engine=trt_engine, context=context_onet)

        self.face_recognizer = FaceRecognizer(engine=trt_engine, context=context_resnet, num_classes=42)

         # for probabilities
        self.thresholds = [0.6, 0.7, 0.8]
        self.img_height = 720
        self.img_width = 1280
    
    def detect_face(self, image): # = Image.open('./0008_01.jpg')

        img_w, img_h = image.size
        scale = min((self.img_height*1.0)/img_h, (self.img_width*1.0)/img_w)
        if scale < 1.0:
           image = image.resize((int(img_w*scale),int(img_h*scale)))
        
        
        output = self.pnet_detector.run_pnet(image)
        bounding_boxes = self.pnet_detector.propose_bboxes(image, output, self.thresholds[0])

        show_bboxes(image, bounding_boxes)    
    
        img_boxes = get_image_boxes(bounding_boxes, image, size=24)
        img =  np.asarray(img_boxes, np.float16)
        img =  img[0:64]
        bounding_boxes = self.rnet_detector.run_rnet(img, self.thresholds[1], bounding_boxes)
        show_bboxes(image, bounding_boxes)
    
        img_boxes = get_image_boxes(bounding_boxes, image, size=48)
        img =  np.asarray(img_boxes, np.float16)
        img =  img[0:64]
        bounding_boxes, landmarks = self.onet_detector.run_onet(img, self.thresholds[2], bounding_boxes)
        #img = show_bboxes(image, bounding_boxes, landmarks)
        #print (bounding_boxes.shape)
        
        
        if (bounding_boxes.shape[0] > 0):
            bounding_boxes[0][0] = max(0,bounding_boxes[0][0]-20)
            bounding_boxes[0][1] = max(0, bounding_boxes[0][1]-20)
            bounding_boxes[0][2] = bounding_boxes[0][2]+20
            bounding_boxes[0][3] = bounding_boxes[0][3]+20
            img = extract_face(image, bounding_boxes[0])
        else:
            img=None
        return img
       
    def face_vector(self, image):
        output = self.face_recognizer.run_resnet(image)
        return np.array(output)

    
        
               


def crop_resize(img, box, image_size):
    out = img.crop(box).copy().resize((image_size, image_size), Image.BILINEAR)
    return out

def extract_face(img, box, image_size=160, margin=0, save_path=None):
    """Extract face + margin from PIL Image given bounding box.
    
    Arguments:
        img {PIL.Image} -- A PIL Image.
        box {numpy.ndarray} -- Four-element bounding box.
        image_size {int} -- Output image size in pixels. The image will be square.
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image. 
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size.
        save_path {str} -- Save path for extracted face image. (default: {None})
    
    Returns:
        torch.tensor -- tensor representing the extracted face.
    """
    margin = [
        margin * (box[2] - box[0]) / (image_size - margin),
        margin * (box[3] - box[1]) / (image_size - margin),
    ]
    raw_image_size = img.size
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, raw_image_size[0])),
        int(min(box[3] + margin[1] / 2, raw_image_size[1])),
    ]

    face = crop_resize(img, box, image_size)

    
    return face


