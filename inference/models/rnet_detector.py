
import math
from PIL import Image
import numpy as np
from utils.box_utils import nms, _preprocess, calibrate_box, get_image_boxes, convert_to_square


class RNetDetector(object):
   
    def __init__(self, engine=None, context=None, batch_size=64):
        self.rnet_engine = engine
        self.context_rnet = context
        self.output = np.empty([batch_size, 6], dtype = np.float16)
        self.batch_size = batch_size

    def run_rnet(self, images, threshold, bounding_boxes):
        output = []
        for count in range(int(images.shape[0]/self.batch_size)):
            output.append(self.rnet_engine.run(images[count*self.batch_size:(count+1)*self.batch_size], self.output, self.context_rnet))
 
        output = np.vstack(output)
        offsets = output[:,0:4]
        probs = output[:,4:]
    
   
        keep = np.where(probs[:, 1] > threshold)[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]
   
        keep = nms(bounding_boxes, 0.7)
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])
        print('number of bounding boxes:', len(bounding_boxes))
        return bounding_boxes
        



   
   
   
