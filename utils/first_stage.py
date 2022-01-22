import torch
from torch.autograd import Variable
import math
from PIL import Image
import numpy as np
from .box_utils import nms, calibrate_box, get_image_boxes, convert_to_square


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

def preprocess_images(img):
    """Preprocessing step before feeding the network.

    Arguments:
        img: a float numpy array of shape [h, w, c].

    Returns:
        a float numpy array of shape [1, c, h, w].
    """
    img = img.transpose((0, 3, 1, 2))
    #img = np.expand_dims(img, 0)
    img = (img - 127.5)*0.0078125
    return img

def run_pnet(images, net, threshold, min_face_size=20):
    """Run P-Net, generate bounding boxes, and do NMS.

    Arguments:
        image: an instance of PIL.Image.
        net: an instance of pytorch's nn.Module, P-Net.
        scale: a float number,
            scale width and height of the image by this number.
        threshold: a float number,
            threshold on the probability of a face when generating
            bounding boxes from predictions of the net.

    Returns:
        a float numpy array of shape [n_boxes, 9],
            bounding boxes with scores and offsets (4 + 1 + 4).
    """

    # scale the image and convert it to a float array
    width, height = images[0].size
    min_length = min(height, width)
    
    
    min_detection_size = 12
    image_h_offsets = [[0], [0], [0]]
    factor = 0.707  # sqrt(0.5)

# scales for scaling the image
    scales = [[0.30, 0.026], [0.212, 0.053, 0.037, 0.019], [0.15, 0.106, 0.075]]

    # scales the image so that
    # minimum size that we can detect equals to
    # minimum face size that we want to detect
    m = min_detection_size/min_face_size
    min_length *= m

    for i in range(len(scales)):
        for j in range(len(scales[i])):
            image_h_offsets[i].append(int(scales[i][j]*height))
    

    print('scales:'.format(scales))
    print('image pyramid offsets:', image_h_offsets)
 
    im_data = np.zeros((len(images)*len(scales), sum(image_h_offsets[2]), int(width*m), 3), dtype=np.float32)
    for k, image in enumerate(images):
        for i in range(len(scales)):
            for j, scale in enumerate(scales[i]):
                h_offset = sum(image_h_offsets[i][0:j+1])
                h = int(height * scale)
                w = int(width * scale)
                print(i, h_offset, h, h+h_offset)
                img = image.resize((w, h), Image.BILINEAR)
                img = np.asarray(img, 'float32')
                im_data[i+k*len(scales), h_offset:(h_offset+h), :w,:] = img
            
    
    img = torch.tensor(preprocess_images(im_data))
 
  
    output = net(img)        
    probs = output.data.numpy()[:,5,:,:]
    offsets =output.data.numpy()[:,0:4,:,:]
    
    bounding_boxes = []
    for k in range(len(images)):
        bounding_boxes.append([])
      
        bboxes_per_image = []
        for i in range(len(scales)):
            for j, scale in enumerate(scales[i]):
                h_offset = sum(image_h_offsets[i][0:j+1])//2
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
        print('number of bounding boxes:', len(bboxes_per_image))
        bounding_boxes[k] = bboxes_per_image
    

    return bounding_boxes