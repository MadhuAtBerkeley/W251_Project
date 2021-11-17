
import math
from PIL import Image
import numpy as np
from utils.box_utils import nms, _preprocess, calibrate_box, get_image_boxes, convert_to_square


class ONetDetector(object):
   
    def __init__(self, engine=None, context=None, batch_size=64):
        self.onet_engine = engine
        self.context_onet = context
        self.output = np.empty([batch_size, 16], dtype = np.float16)
        self.batch_size = batch_size

    def run_onet(self, images, threshold, bounding_boxes):
        output = self.onet_engine.run(images, self.output, self.context_onet)
 
        #output = np.vstack(output)
        offsets = output[:,0:4]
        landmarks = output[:,4:14]
        probs = output[:,14:16]
    
   
        keep = np.where(probs[:, 1] > threshold)[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]
        landmarks = landmarks[keep]
   
        # compute landmark points
        width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
        xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
        landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1)*landmarks[:, 0:5]
        landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1)*landmarks[:, 5:10]
   
        bounding_boxes = calibrate_box(bounding_boxes, offsets)
        keep = nms(bounding_boxes, 0.7, mode='min')
        bounding_boxes = bounding_boxes[keep]
        landmarks = landmarks[keep]
        print('number of bounding boxes:', len(bounding_boxes))
        return bounding_boxes, landmarks



