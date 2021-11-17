
import math
from PIL import Image
import numpy as np
from utils.box_utils import nms, _preprocess
from utils.box_utils import calibrate_box, get_image_boxes, convert_to_square

class PNetDetector(object):
    
    def __init__(self, engine=None, context=None, ref_image_size=(1280, 720), min_face_size=40, factor=0.707):
        width, height = ref_image_size[0], ref_image_size[1]
        min_length = height
        self.image_h_offsets = [0]
        # scales for scaling the image
        self.scales = []
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

        factor_count = 0
        while min_length > self.min_detection_size:
            scale = m*factor**factor_count
            self.scales.append(scale)
            self.image_h_offsets.append(int(scale*height))
            min_length *= factor
            factor_count += 1

        print('scales:', ['{:.2f}'.format(s) for s in self.scales])
        print('image pyramid offsets:', self.image_h_offsets)
        
    def run_pnet(self, image):
        width, height = image.size
        im_data = np.zeros((sum(self.image_h_offsets), self.image_w_offset, 3), dtype=np.float16)
        
               
        for i, scale in enumerate(self.scales):
            h_offset = sum(self.image_h_offsets[0:i+1])
            h = int(height * scale)
            w = int(width * scale)
            img = image.resize((w, h), Image.BILINEAR)
            img = np.asarray(img, 'float16')
            im_data[h_offset:(h_offset+h), :w,:] = img
            
            
        #img = torch.tensor(_preprocess(im_data))
        img = _preprocess(im_data)
      
        img = np.ascontiguousarray(img, dtype=np.float16)
        assert img.flags['C_CONTIGUOUS'] == True
        #output = net(img)     
  
        output = np.empty([1,6,345,187], dtype = np.float16)
        assert output.flags['C_CONTIGUOUS'] == True
        output = self.pnet_engine.run(img, output, self.context_pnet)
        
        return output
    
    
    def propose_bboxes(self, image, probs_offsets, threshold=0.7):
        width, height = image.size
        output = probs_offsets
        probs = output[0,5,:,:]
        offsets =output[:,0:4,:,:]
        
        # probs: probability of a face at each sliding window
        # offsets: transformations to true bounding boxes
        bounding_boxes = []
        for i, scale in enumerate(self.scales):
            h_offset = sum(self.image_h_offsets[0:i+1])//2
            h = (int(height * scale) - 12) // 2 + 1
            w = (int(width * scale) - 12) // 2 + 1
            boxes = _generate_bboxes(probs[h_offset:h_offset+h,:w], offsets[:,:,h_offset:h_offset+h,:w], scale, threshold)
            
            if len(boxes) == 0:
                continue

            keep = nms(boxes[:, 0:5], overlap_threshold=0.7)
            bounding_boxes.append(boxes[keep])
    
   
    
        bounding_boxes = np.vstack(bounding_boxes)
        bounding_boxes = bounding_boxes[bounding_boxes[:,4].argsort()[::-1]]
        bounding_boxes = bounding_boxes[0:64]
        #print('number of bounding boxes:', len(bounding_boxes))
    
        keep = nms(bounding_boxes[:, 0:5], 0.7)
        bounding_boxes = bounding_boxes[keep]

        # use offsets predicted by pnet to transform bounding boxes
        bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
        # shape [n_boxes, 5]

        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])
        print('number of bounding boxes:', len(bounding_boxes))
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
    tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
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
