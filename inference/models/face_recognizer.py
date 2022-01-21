import math
from PIL import Image
import numpy as np
from utils.box_utils  import _preprocess


class FaceRecognizer(object):
   
    def __init__(self, engine=None, context=None, num_classes=42):
        self.resnet_engine = engine
        self.context_resnet = context
        self.output = np.empty([1, 512], dtype = np.float16)
        self.num_classes = num_classes

    def run_resnet(self, image):
        img = np.asarray(image, 'float16')
        img = _preprocess(img)
        img = np.ascontiguousarray(img, 'float16')
        output = self.resnet_engine.run(img, self.output, self.context_resnet)
        return output
        
