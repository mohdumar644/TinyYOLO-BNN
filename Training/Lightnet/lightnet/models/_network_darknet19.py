#
#   Darknet Darknet19 model
#   Copyright EAVISE
#

import os
from collections import OrderedDict
import torch
import torch.nn as nn

import lightnet.network as lnn
import lightnet.data as lnd

__all__ = ['Darknet19']


class Darknet19(lnn.module.Darknet):
    """ `Darknet19`_ implementation with pytorch.

    Args:
        num_classes (Number, optional): Number of classes; Default **1000**
        input_channels (Number, optional): Number of input channels; Default **3**

    Attributes:
        self.loss (fn): loss function; Default :class:`torch.nn.Crossentropyloss`
        self.postprocess (fn): Postprocessing function; Default :class:`torch.nn.Softmax`
        self.remap_yolo (list): Remapping sequences for :func:`~lightnet.network.module.Lightnet.save` that allow to save the first 23 layers for using them with :class:`~lightnet.models.Yolo`.

    .. _Darknet19: https://github.com/pjreddie/darknet/blob/master/cfg/darknet19.cfg
    """
    remap_yolo = [
        (r'^layers.([1-9]_)', r'layers.0.\1'),
        (r'^layers.(1[0-7]_)', r'layers.0.\1'),
        (r'^layers.([12][890-3]_)', r'layers.1.\1'),
    ]

    def __init__(self, num_classes=1000, input_channels=3):
        """ Network initialisation """
        super().__init__()

        # Parameters
        self.num_classes = num_classes

        # Network
        self.layers = nn.Sequential(
            OrderedDict([
                ('1_convbatch',     lnn.layer.Conv2dBatchReLU(input_channels, 32, 3, 1, 1)),
                ('2_max',           nn.MaxPool2d(2, 2)),
                ('3_convbatch',     lnn.layer.Conv2dBatchReLU(32, 64, 3, 1, 1)),
                ('4_max',           nn.MaxPool2d(2, 2)),
                ('5_convbatch',     lnn.layer.Conv2dBatchReLU(64, 128, 3, 1, 1)),
                ('6_convbatch',     lnn.layer.Conv2dBatchReLU(128, 64, 1, 1, 0)),
                ('7_convbatch',     lnn.layer.Conv2dBatchReLU(64, 128, 3, 1, 1)),
                ('8_max',           nn.MaxPool2d(2, 2)),
                ('9_convbatch',     lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ('10_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0)),
                ('11_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ('12_max',          nn.MaxPool2d(2, 2)),
                ('13_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ('14_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0)),
                ('15_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ('16_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0)),
                ('17_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ('18_max',          nn.MaxPool2d(2, 2)),
                ('19_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                ('20_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                ('21_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                ('22_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                ('23_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                ('24_conv',         nn.Conv2d(1024, num_classes, 1, 1, 0)),
                ('25_avgpool',      lnn.layer.GlobalAvgPool2d())
            ])
        )

        # Post
        self.loss = nn.CrossEntropyLoss(size_average=False)
        self.postprocess = lnd.transform.Compose([
            nn.Softmax(1)
        ])

    def _forward(self, x):
        return self.layers(x)
