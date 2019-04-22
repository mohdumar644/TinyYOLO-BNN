#!/usr/bin/env python
#
#   Copyright EAVISE
#   Example: Perform a single image detection with the Lightnet tiny yolo network
#

import os
import argparse
import logging
import cv2
import torch
from torchvision import transforms as tf
import brambox.boxes as bbb
import lightnet as ln

log = logging.getLogger('lightnet.detect')
logging.basicConfig(level=logging.DEBUG)

# Parameters
CLASSES = 20
NETWORK_SIZE = (416, 416)
LABELS = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

CONF_THRESH = .4
NMS_THRESH = .4


# Functions
def create_network():
    """ Create the lightnet network """
    net = ln.models.TinyYolo(CLASSES,  CONF_THRESH, NMS_THRESH)

    net.load(args.weight)
    net.eval()
    net.postprocess.append(ln.data.transform.TensorToBrambox(NETWORK_SIZE, LABELS))
    net = net.to(device)
    return net


def detect(net, img_path):
    """ Perform a detection """
    # Load image
    img = cv2.imread(img_path)
    im_h, im_w = img.shape[:2]

    img_tf = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tf = ln.data.transform.Letterbox.apply(img_tf, dimension=NETWORK_SIZE)
    img_tf = tf.ToTensor()(img_tf)
    img_tf.unsqueeze_(0)
    img_tf = img_tf.to(device)

    # Run detector
    with torch.no_grad():
        out = net(img_tf, target=None)
    
    out = ln.data.transform.ReverseLetterbox.apply(out, network_size=NETWORK_SIZE, image_size =(im_w,im_h)) # Resize bb to true image dimensions

    print(out)

    return img, out


# Main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run an image through the lightnet yolo network')
    parser.add_argument('weight', help='Path to weight file')
    parser.add_argument('image', help='Path to image file(s)', nargs='*')
    parser.add_argument('-c', '--cuda', action='store_true', help='Use cuda')
    parser.add_argument('-s', '--save', action='store_true', help='Save image in stead of displaying it')
    parser.add_argument('-l', '--label', action='store_true', help='Print labels and scores on the image')
    args = parser.parse_args()

    # Parse Arguments
    device = torch.device('cpu')
    if args.cuda:
        if torch.cuda.is_available():
            log.debug('CUDA enabled')
            device = torch.device('cuda')
        else:
            log.error('CUDA not available')

    # Network
    network = create_network()
    print(network)
    print()

    # Detection
    if len(args.image) > 0:
        for img_name in args.image:
            log.info(img_name)
            image, output = detect(network, img_name)

            image = bbb.draw_boxes(image, output[0], show_labels=args.label)
            if args.save:
                cv2.imwrite('detections.png', image)
            else:
                cv2.imshow('image', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                pass
    else:
        while True:
            try:
                img_path = input('Enter image path: ')    
            except (KeyboardInterrupt, EOFError):
                print('')
                break
        
            if not os.path.isfile(img_path):
                log.error(f'\'{img_path}\' is not a valid path')
                break

            image, output = detect(network, img_path)
            image = bbb.draw_boxes(image, output[0], show_labels=args.label)
            if args.save:
                cv2.imwrite('detections.png', image)
            else:
                cv2.imshow('image', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
