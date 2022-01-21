import torch
from torch.autograd import Variable
import math
from PIL import Image
import numpy as np
from .box_utils import nms, _preprocess, calibrate_box, get_image_boxes, convert_to_square




def run_onet(onet, img_boxes, thresholds, bounding_boxes):
    img_boxes = torch.tensor(img_boxes)
    output = onet(img_boxes)
    #landmarks = output[1].data.numpy()  # shape [n_boxes, 10]
    #offsets = output[0].data.numpy()  # shape [n_boxes, 4]
    #probs = output[2].data.numpy()  # shape [n_boxes, 2]
    output = output.data.numpy()
    offsets = output[:,0:4]
    landmarks = output[:,4:14]
    probs = output[:,14:16]
   
    keep = np.where(probs[:, 1] > thresholds[2])[0]
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
   