import torch
from torch.autograd import Variable
import math
from PIL import Image
import numpy as np
from .box_utils import nms, _preprocess, calibrate_box, get_image_boxes, convert_to_square




def run_rnet(rnet, img_boxes, thresholds, bounding_boxes):
    img_boxes = torch.tensor(img_boxes)
    output = rnet(img_boxes)
    #offsets = output[0].data.numpy()  # shape [n_boxes, 4]
    #probs = output[1].data.numpy() 
    #print(offsets.shape, probs.shape)
    output = output.data.numpy()
    offsets = output[:,0:4]
    probs = output[:,4:]
    
   
    keep = np.where(probs[:, 1] > thresholds[1])[0]
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
   
   
   