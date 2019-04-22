#
#   Darknet Tiny YOLOv2 model
#   Copyright EAVISE
#

import os
from collections import OrderedDict, Iterable
import torch
import torch.nn as nn

import lightnet.network as lnn
import lightnet.data as lnd

__all__ = ['TinyYolo']

from ..network.layer.dorefa import *



class TinyYolo(lnn.module.Darknet):
    """ `Tiny Yolo v2`_ implementation with pytorch.
    This network uses :class:`~lightnet.network.RegionLoss` as its loss function
    and :class:`~lightnet.data.GetBoundingBoxes` as its default postprocessing function.

    Args:
        num_classes (Number, optional): Number of classes; Default **20**
        conf_thresh (Number, optional): Confidence threshold for postprocessing of the boxes; Default **0.25**
        nms_thresh (Number, optional): Non-maxima suppression threshold for postprocessing; Default **0.4**
        input_channels (Number, optional): Number of input channels; Default **3**
        anchors (list, optional): 2D list with anchor values; Default **Tiny yolo v2 anchors**

    Attributes:
        self.loss (fn): loss function. Usually this is :class:`~lightnet.network.RegionLoss`
        self.postprocess (fn): Postprocessing function. By default this is :class:`~lightnet.data.GetBoundingBoxes` + :class:`~lightnet.data.NonMaxSupression`

    .. _Tiny Yolo v2: https://github.com/pjreddie/darknet/blob/777b0982322142991e1861161e68e1a01063d76f/cfg/tiny-yolo-voc.cfg
    """
    def __init__(self, num_classes=20, conf_thresh=.25, nms_thresh=.5, input_channels=3,
                 anchors=[(1.08, 1.19), (3.42, 4.41), (6.63, 11.38), (9.42, 5.11), (16.62, 10.52)]):
        """ Network initialisation """
        super().__init__()
        if not isinstance(anchors, Iterable) and not isinstance(anchors[0], Iterable):
            raise TypeError('Anchors need to be a 2D list of numbers')

        # Parameters
        self.num_classes = num_classes
        self.anchors = anchors
        self.reduction = 32     # input_dim/output_dim

        # Network

        layer_list = OrderedDict([
            ('1_convbatch',     lnn.layer.Conv2dBatchReLU(input_channels, 16, 3, 1, 1)),
            ('2_max',           nn.MaxPool2d(2, 2)),
            ('3_convbatch',     lnn.layer.Conv2dBatchReLU(16, 64, 3, 1, 1)), #==================64
            # ('3_convbatch',     lnn.layer.Conv2dBatchReLU(16, 32, 3, 1, 1)),
            ('4_max',           nn.MaxPool2d(2, 2)),
            ('5_convbatch',     lnn.layer.Conv2dBatchReLU(64, 64, 3, 1, 1)),   #==================64
            # ('5_convbatch',     lnn.layer.Conv2dBatchReLU(32, 64, 3, 1, 1)),
            ('6_max',           nn.MaxPool2d(2, 2)),
            ('7_convbatch',     lnn.layer.Conv2dBatchReLU(64, 128, 3, 1, 1)),
            ('8_max',           nn.MaxPool2d(2, 2)),
            ('9_convbatch',     lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
            ('10_max',          nn.MaxPool2d(2, 2)),
            ('11_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
            #('12_max',          lnn.layer.PaddedMaxPool2d(2, 1, (0, 1, 0, 1))),
            ('13_convbatch',    lnn.layer.Conv2dBatchReLU(512, 512, 3, 1, 1)),
            ('14_convbatch',    lnn.layer.Conv2dBatchReLU(512, 512, 3, 1, 1)),
            ('15_conv',         BinarizeConv2dLast(512, len(self.anchors)*(5+self.num_classes), 1, 1, 0)),
        ])  



        # layer_list = OrderedDict([
        #     ('1_convbatch',     lnn.layer.Conv2dBatchReLU(input_channels, 16, 3, 1, 1)),
        #     ('2_max',           nn.MaxPool2d(2, 2)),
        #     ('3_convbatch',     lnn.layer.Conv2dBatchReLU(16, 32, 3, 1, 1)),
        #     ('4_max',           nn.MaxPool2d(2, 2)),
        #     ('5_convbatch',     lnn.layer.Conv2dBatchReLU(32, 64, 3, 1, 1)),
        #     ('6_max',           nn.MaxPool2d(2, 2)),
        #     ('7_convbatch',     lnn.layer.Conv2dBatchReLU(64, 128, 3, 1, 1)),
        #     ('8_max',           nn.MaxPool2d(2, 2)),
        #     ('9_convbatch',     lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
        #     ('10_max',          nn.MaxPool2d(2, 2)),
        #     ('11_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
        #     ('12_max',          lnn.layer.PaddedMaxPool2d(2, 1, (0, 1, 0, 1))),
        #     ('13_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1)),
        #     ('14_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 1024, 3, 1, 1)),
        #     ('15_conv',         nn.Conv2d(1024, len(self.anchors)*(5+self.num_classes), 1, 1, 0)),
        # ])












        self.layers = nn.Sequential(layer_list)

        # Post
        self.loss = lnn.loss.RegionLoss(self.num_classes, self.anchors, self.reduction, 0)
        self.postprocess = lnd.transform.Compose([
            lnd.transform.GetBoundingBoxes(self.num_classes, self.anchors, conf_thresh),
            lnd.transform.NonMaxSupression(nms_thresh)
        ])
