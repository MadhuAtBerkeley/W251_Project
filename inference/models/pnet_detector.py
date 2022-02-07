
import math
from PIL import Image
import numpy as np
from utils.box_utils import nms, _preprocess
from utils.box_utils import calibrate_box, get_image_boxes, convert_to_square
from time import time

def preprocess_images(img):
    """Preprocessing step before feeding the network.

    Arguments:
        img: a float numpy array of shape [h, w, c].

    Returns:
        a float numpy array of shape [1, c, h, w].
    """
    img = img.transpose((0, 3, 1, 2))
    #img = np.expand_dims(img, 0)
    #img = (img - 127) #.5)*0.0078125
    return img//2

class PNetDetector(object):
    
    def __init__(self, engine=None, context=None, ref_image_size=(1280, 720), min_face_size=40, factor=0.707, verbose=False):
        width, height = ref_image_size[0], ref_image_size[1]
        min_length = height
        self.image_h_offsets = [[0], [0], [0]]
        # scales for scaling the image
        #self.scales = [[0.30, 0.026], [0.212, 0.053, 0.037, 0.019], [0.15, 0.106, 0.075]]
        #self.scales = [[0.12], [0.085, 0.042], [0.06, 0.03, 0.02]]
        self.scales = [[0.075], [0.053, 0.019], [0.0375, 0.0265]]
        self.min_detection_size = 12
        self.min_face_size = min_face_size
        
        # scales the image so that
        # minimum size that we can detect equals to
        # minimum face size that we want to detect
        m = self.min_detection_size/self.min_face_size
        min_length *= m
        self.image_w_offset = int(width*m)
        
        
        self.pnet_engine = engine
        self.context_pnet = context

        for i in range(len(self.scales)):
            for j in range(len(self.scales[i])):
                self.image_h_offsets[i].append(int(self.scales[i][j]*height))
    

        
        
        if(verbose):
            print('scales:'.format(self.scales))
            print('image pyramid offsets:', self.image_h_offsets)
        
    def run_pnet(self, images):
        
        t1 = time()
        width, height = images[0].size #hape[1], image.shape[0]
        
        max_h_offset  = max(sum(offset) for offset in self.image_h_offsets)
        im_data = np.zeros((3, max_h_offset, self.image_w_offset, 3), dtype=np.uint8)
        
        
        for k, image in enumerate(images):
            for i in range(len(self.scales)):
                for j, scale in enumerate(self.scales[i]):
                    h_offset = sum(self.image_h_offsets[i][0:j+1])
                    h = int(height * scale)
                    w = int(width * scale)
                    if h == 0 or w == 0:
                        continue
                    #print(i, h_offset, h, h+h_offset)
                    img = image.resize((w, h), Image.BILINEAR)
                    img = np.asarray(img, 'uint8')
                    im_data[i+k*len(self.scales), h_offset:(h_offset+h), :w,:] = img
               
        
            
        img = preprocess_images(im_data)
        img = np.ascontiguousarray(img, dtype=np.int8) #float16)
        print(img.shape)
        assert img.flags['C_CONTIGUOUS'] == True
           
  
        output = np.empty([3,6,22,43], dtype = np.float16)
        assert output.flags['C_CONTIGUOUS'] == True
        output = self.pnet_engine.run(img, output, self.context_pnet)
        t2 = time()    
        print("Pnet Time:{}".format(t2-t1))
        return output
    
    
    
    def propose_bboxes(self, images, probs_offsets, threshold=0.7, verbose=False):
        t1 = time()
        width, height = images[0].size
        output = probs_offsets
        probs = output[:,5,:,:]
        offsets =output[:,0:4,:,:]
   
        bounding_boxes = []
        for k in range(len(images)):
            bounding_boxes.append([])
     
            bboxes_per_image = []
            for i in range(len(self.scales)):
                for j, scale in enumerate(self.scales[i]):
                    h_offset = sum(self.image_h_offsets[i][0:j+1])//2
                    h = (int(height * scale) - 12) // 2 + 1
                    w = (int(width * scale) - 12) // 2 + 1
                    boxes = _generate_bboxes(probs[i, h_offset:h_offset+h,:w], offsets[i,:,h_offset:h_offset+h,:w], scale, threshold)
           
                    if len(boxes) == 0:
                        continue
                    keep = nms(boxes[:, 0:5], overlap_threshold=0.5)    
                    bboxes_per_image.append(boxes[keep])
               
            bboxes_per_image = np.vstack(bboxes_per_image)
        #print('number of bounding boxes:', len(bounding_boxes))
   
            keep = nms(bboxes_per_image[:, 0:5], threshold)
            bboxes_per_image = bboxes_per_image[keep]    
       
    # use offsets predicted by pnet to transform bounding boxes
            bboxes_per_image = calibrate_box(bboxes_per_image[:, 0:5],  bboxes_per_image[:, 5:])
    # shape [n_boxes, 5]

            bboxes_per_image = convert_to_square(bboxes_per_image)
            bboxes_per_image[:, 0:4] = np.round(bboxes_per_image[:, 0:4])
            if verbose:
                print('number of bounding boxes:', len(bboxes_per_image))
            bounding_boxes[k] = bboxes_per_image

     
        t2 = time()    
        print("Propose BBoxes CPU Time:{}".format(t2-t1))
        return bounding_boxes

        



def _generate_bboxes(probs, offsets, scale, threshold):
    """Generate bounding boxes at places
    where there is probably a face.

    Arguments:
        probs: a float numpy array of shape [n, m].
        offsets: a float numpy array of shape [1, 4, n, m].
        scale: a float number,
            width and height of the image were scaled by this number.
        threshold: a float number.

    Returns:
        a float numpy array of shape [n_boxes, 9]
    """

    # applying P-Net is equivalent, in some sense, to
    # moving 12x12 window with stride 2
    stride = 2
    cell_size = 12

    # indices of boxes where there is probably a face
    inds = np.where(probs > threshold)

    if inds[0].size == 0:
        return np.array([])

    # transformations of bounding boxes
    tx1, ty1, tx2, ty2 = [offsets[i, inds[0], inds[1]] for i in range(4)]
    # they are defined as:
    # w = x2 - x1 + 1
    # h = y2 - y1 + 1
    # x1_true = x1 + tx1*w
    # x2_true = x2 + tx2*w
    # y1_true = y1 + ty1*h
    # y2_true = y2 + ty2*h

    offsets = np.array([tx1, ty1, tx2, ty2])
    score = probs[inds[0], inds[1]]

    # P-Net is applied to scaled images
    # so we need to rescale bounding boxes back
    bounding_boxes = np.vstack([
        np.round((stride*inds[1] + 1.0)/scale),
        np.round((stride*inds[0] + 1.0)/scale),
        np.round((stride*inds[1] + 1.0 + cell_size)/scale),
        np.round((stride*inds[0] + 1.0 + cell_size)/scale),
        score, offsets
    ])
    # why one is added?

    return bounding_boxes.T
