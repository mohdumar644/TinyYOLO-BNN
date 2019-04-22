import sys
import os
import scipy.io
import numpy as np
import time
import cv2
import warnings
warnings.filterwarnings('ignore')


def sigmoid(x):
  return 1. / (1. + np.exp(-x))



def softmax1(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out


def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


def iou(boxA,boxB):
  # boxA = boxB = [x1,y1,x2,y2]

  # Determine the coordinates of the intersection rectangle
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])
 
  # Compute the area of intersection
  intersection_area = (xB - xA + 1) * (yB - yA + 1)
 
  # Compute the area of both rectangles
  boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
  # Compute the IOU
  iou = intersection_area / float(boxA_area + boxB_area - intersection_area)

  return iou



def non_maximal_suppression(thresholded_predictions,iou_threshold):

  nms_predictions = []

  # Add the best B-Box because it will never be deleted
  nms_predictions.append(thresholded_predictions[0])

  # For each B-Box (starting from the 2nd) check its iou with the higher score B-Boxes
  # thresholded_predictions[i][0] = [x1,y1,x2,y2]
  i = 1
  while i < len(thresholded_predictions):
    n_boxes_to_check = len(nms_predictions)
    #print('N boxes to check = {}'.format(n_boxes_to_check))
    to_delete = False

    j = 0
    while j < n_boxes_to_check:
        curr_iou = iou(thresholded_predictions[i][0],nms_predictions[j][0])
        if(curr_iou > iou_threshold ):
            to_delete = True
        #print('Checking box {} vs {}: IOU = {} , To delete = {}'.format(thresholded_predictions[i][0],nms_predictions[j][0],curr_iou,to_delete))
        j = j+1

    if to_delete == False:
        nms_predictions.append(thresholded_predictions[i])
    i = i+1

  return nms_predictions





def postprocessing(predictions,input_image,score_threshold,iou_threshold,input_height,input_width,anchors,colors,classes):

  # input_image = cv2.resize(input_image,(input_height, input_width), interpolation = cv2.INTER_CUBIC)
 
  n_grid_cells = 13
  n_b_boxes = 5
  n_b_box_coord = 4

  # Names and colors for each class
#   classes = ["runway"]
  num_classes = len(classes)
#   colors = [(254.0, 254.0, 254)]

  # Pre-computed YOLOv2 shapes of the k=5 B-Boxes
#   anchors = [(1.65,8.92), (7.26,6.39), (8.21,10.46), (8.75,3.16),( 9.00,0.97)]
  anchors=np.asarray(anchors)
  num_anchors = anchors.shape[0]

  network_output=predictions


  h = network_output.shape[2]
  w = network_output.shape[3]

  lin_x = np.tile(np.linspace(0, w-1, w),h).reshape(h*w)
  lin_y = np.linspace(0, h-1, h).repeat(w)
  anchor_w = anchors[:, 0].reshape(1, num_anchors, 1)
  anchor_h = anchors[:, 1].reshape(1, num_anchors, 1)   

  network_output = network_output.reshape(1, num_anchors, -1, h*w) # (1, 5, 6, 169)

  network_output[:, :, 0, :] = (sigmoid( network_output[:, :, 0, :]  ) +  lin_x)* 32.
  network_output[:, :, 1, :] = (sigmoid( network_output[:, :, 1, :]  ) +  lin_y)* 32.

  network_output[:, :, 2, :] = (np.exp(network_output[:, :, 2, :] ) * anchor_w)* 32.
  network_output[:, :, 3, :] = (np.exp(network_output[:, :, 3, :] ) * anchor_h)* 32.
  network_output[:, :, 4, :] = sigmoid(network_output[:, :, 4, :])

  if num_classes > 1:  #TODO
      cls_scores = softmax(network_output[:, :, 5:, :], axis=2)
      cls_max_idx = np.argmax(cls_scores,2)
      cls_max = np.amax(cls_scores,2)
      cls_max *= network_output[:, :, 4, :]
  else:
      cls_max = network_output[:, :, 4, :]
      cls_max_idx = np.zeros_like(cls_max,dtype=np.int32)

  score_thresh = cls_max > score_threshold

  score_thresh_flat = score_thresh.reshape(-1)
  coords = network_output.transpose(0,1,3,2)[..., 0:4]
  coords = coords[score_thresh[..., None].repeat(4,3)].reshape(-1, 4).round()
    
#   print(coords.shape)
   
  scores = cls_max[score_thresh]
  idx = cls_max_idx[score_thresh]

  left   = np.round(coords[:,0] - (coords[:,2]/2.))
  right   = np.round(coords[:,0] + (coords[:,2]/2.))
  top   = np.round(coords[:,1] - (coords[:,3]/2.))
  bottom   = np.round(coords[:,1] + (coords[:,3]/2.))

  thresholded_predictions = [[[int(left[i]),int(top[i]),int(right[i]),int(bottom[i])],scores[i],classes[idx[i]]] for i in range(coords.shape[0])]

  thresholded_predictions.sort(key=lambda tup: tup[1],reverse=True)
  if len(thresholded_predictions) == 0:
    return input_image
  
  # print('Printing {} B-boxes survived after score thresholding:'.format(len(thresholded_predictions)))
  # for i in range(len(thresholded_predictions)):
    # print('B-Box {} : {}'.format(i+1,thresholded_predictions[i]))

  # Non maximal suppression
  # print('Non maximal suppression with iou threshold = {}'.format(iou_threshold))
  nms_predictions = non_maximal_suppression(thresholded_predictions,iou_threshold)


  # Print survived b-boxes
  # print('Printing the {} B-Boxes survived after non maximal suppression:'.format(len(nms_predictions)))
  # for i in range(len(nms_predictions)):
    # print('B-Box {} : {}'.format(i+1,nms_predictions[i]))

  # Draw final B-Boxes and label on input image
  for i in range(len(nms_predictions)):

      color = colors[classes.index(nms_predictions[i][2])]
      best_class_name = nms_predictions[i][2]

      # Put a class rectangle with B-Box coordinates and a class label on the image
      input_image = cv2.rectangle(input_image,(nms_predictions[i][0][0],nms_predictions[i][0][1]),(nms_predictions[i][0][2],nms_predictions[i][0][3]),color,thickness=3)
#       cv2.putText(input_image,best_class_name,(int((nms_predictions[i][0][0]+nms_predictions[i][0][2])/2),int((nms_predictions[i][0][1]+nms_predictions[i][0][3])/2)),cv2.FONT_HERSHEY_SIMPLEX,1,color,1)
      sz, _ = cv2.getTextSize(best_class_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)  
      input_image = cv2.rectangle(input_image,(nms_predictions[i][0][0],nms_predictions[i][0][1]-sz[1]),(nms_predictions[i][0][0]+sz[0],nms_predictions[i][0][1]),color,thickness=-3)
      cv2.putText(input_image,best_class_name,(int((nms_predictions[i][0][0])),int((nms_predictions[i][0][1]))),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    
    
  return input_image


