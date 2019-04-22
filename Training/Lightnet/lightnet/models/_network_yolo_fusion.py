#
#   Darknet YOLOv2 with sensor fusion
#   Copyright EAVISE
#

from collections import OrderedDict, Iterable
import torch
import torch.nn as nn
import lightnet.network as lnn
import lightnet.data as lnd

__all__ = ['YoloLateFusion', 'YoloMidFusion']


class YoloLateFusion(lnn.module.Darknet):
    """ Yolo v2 that processes multi-channel input images with late fusion.
    This network uses :class:`~lightnet.network.RegionLoss` as its loss function
    and :class:`~lightnet.data.GetBoundingBoxes` as its default postprocessing function.

    Args:
        num_classes (Number, optional): Number of classes; Default **20**
        conf_thresh (Number, optional): Confidence threshold for postprocessing of the boxes; Default **0.25**
        nms_thresh (Number, optional): Non-maxima suppression threshold for postprocessing; Default **0.4**
        input_channels (Number, optional): Number of input channels for the main subnetwork; Default **3**
        fusion_channels (Number, optional): Number of input channels for the fusion subnetwork; Default **1**
        anchors (list, optional): 2D list with anchor values; Default **Yolo v2 anchors**

    Attributes:
        self.loss (fn): loss function. Usually this is :class:`~lightnet.network.RegionLoss`
        self.postprocess (fn): Postprocessing function. By default this is :class:`~lightnet.data.GetBoundingBoxes` + :class:`~lightnet.data.NonMaxSupression`
    """
    def __init__(self, num_classes=20, conf_thresh=.25, nms_thresh=.4, input_channels=3, fusion_channels=1,
                 anchors=[(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)]):
        """ Network initialisation """
        super().__init__()
        if not isinstance(anchors, Iterable) and not isinstance(anchors[0], Iterable):
            raise TypeError('Anchors need to be a 2D list of numbers')
        if input_channels < 1 or fusion_channels < 1:
            raise ValueError('input_channels and fusion_channels need to be at least 1 [{input_channels}, {fusion_channels}]')

        # Parameters
        self.num_classes = num_classes
        self.anchors = anchors
        self.reduction = 32     # input_dim/output_dim
        self.input_channels = input_channels
        self.fusion_channels = fusion_channels

        # Network
        layer_list = [
            # Sequence 0: input = fusion channels
            OrderedDict([
                ('F1_convbatch',     lnn.layer.Conv2dBatchReLU(fusion_channels, 32, 3, 1, 1)),
                ('F2_max',           nn.MaxPool2d(2, 2)),
                ('F3_convbatch',     lnn.layer.Conv2dBatchReLU(32, 64, 3, 1, 1)),
                ('F4_max',           nn.MaxPool2d(2, 2)),
                ('F5_convbatch',     lnn.layer.Conv2dBatchReLU(64, 128, 3, 1, 1)),
                ('F6_convbatch',     lnn.layer.Conv2dBatchReLU(128, 64, 1, 1, 0)),
                ('F7_convbatch',     lnn.layer.Conv2dBatchReLU(64, 128, 3, 1, 1)),
                ('F8_max',           nn.MaxPool2d(2, 2)),
                ('F9_convbatch',     lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ('F10_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0)),
                ('F11_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ('F12_max',          nn.MaxPool2d(2, 2)),
                ('F13_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ('F14_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0)),
                ('F15_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ('F16_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0)),
                ('F17_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ('F18_convbatch',    lnn.layer.Conv2dBatchReLU(512, 64, 1, 1, 0)),
                ('F19_reorg',        lnn.layer.Reorg(2)),
            ]),

            # Sequence 1 : input = main channels
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
            ]),

            # Sequence 2 : input = sequence1
            OrderedDict([
                ('18_max',          nn.MaxPool2d(2, 2)),
                ('19_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                ('20_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                ('21_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                ('22_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                ('23_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                ('24_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 1024, 3, 1, 1)),
                ('25_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 1024, 3, 1, 1)),
            ]),

            # Sequence 3 : input = sequence1
            OrderedDict([
                ('26_convbatch',    lnn.layer.Conv2dBatchReLU(512, 64, 1, 1, 0)),
                ('27_reorg',        lnn.layer.Reorg(2)),
            ]),

            # Sequence 4 : input = sequence3 + sequence2 + sequence0
            OrderedDict([
                ('28_convbatch',    lnn.layer.Conv2dBatchLeaky((4*64)+1024+(4*64), 1024, 3, 1, 1)),
                ('29_conv',         nn.Conv2d(1024, len(self.anchors)*(5+self.num_classes), 1, 1, 0)),
            ])
        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

        # Post
        self.loss = lnn.loss.RegionLoss(self.num_classes, self.anchors, self.reduction, self.seen)
        self.postprocess = lnd.transform.Compose([
            lnd.transform.GetBoundingBoxes(self.num_classes, self.anchors, conf_thresh),
            lnd.transform.NonMaxSupression(nms_thresh, False)
        ])

    def _forward(self, x):
        if x.size(1) != self.input_channels + self.fusion_channels:
            raise TypeError('This network requires {self.input_channels+self.fusion_channels} channel input images')
        main = x[:, :self.input_channels]
        fusion = x[:, self.input_channels:self.input_channels+self.fusion_channels]

        # Fusion
        out_fusion = self.layers[0](fusion)

        # Main
        out1 = self.layers[1](main)
        out2 = self.layers[2](out1)
        out3 = self.layers[3](out1)

        # Combination
        out = self.layers[4](torch.cat((out3, out2, out_fusion), 1))

        return out


class YoloMidFusion(lnn.module.Darknet):
    """ Yolo v2 that processes multi-channel input images with midway fusion.
    This network uses :class:`~lightnet.network.RegionLoss` as its loss function
    and :class:`~lightnet.data.GetBoundingBoxes` as its default postprocessing function.

    Args:
        num_classes (Number, optional): Number of classes; Default **20**
        conf_thresh (Number, optional): Confidence threshold for postprocessing of the boxes; Default **0.25**
        nms_thresh (Number, optional): Non-maxima suppression threshold for postprocessing; Default **0.4**
        input_channels (Number, optional): Number of input channels for the main subnetwork; Default **3**
        fusion_channels (Number, optional): Number of input channels for the fusion subnetwork; Default **1**
        anchors (list, optional): 2D list with anchor values; Default **Yolo v2 anchors**

    Attributes:
        self.loss (fn): loss function. Usually this is :class:`~lightnet.network.RegionLoss`
        self.postprocess (fn): Postprocessing function. By default this is :class:`~lightnet.data.GetBoundingBoxes` + :class:`~lightnet.data.NonMaxSupression`
    """
    def __init__(self, num_classes=20, conf_thresh=.25, nms_thresh=.4, input_channels=3, fusion_channels=1,
                 anchors=[(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)]):
        """ Network initialisation """
        super().__init__()
        if not isinstance(anchors, Iterable) and not isinstance(anchors[0], Iterable):
            raise TypeError('Anchors need to be a 2D list of numbers')
        if input_channels < 1 or fusion_channels < 1:
            raise ValueError('input_channels and fusion_channels need to be at least 1 [{input_channels}, {fusion_channels}]')

        # Parameters
        self.num_classes = num_classes
        self.anchors = anchors
        self.reduction = 32     # input_dim/output_dim
        self.input_channels = input_channels
        self.fusion_channels = fusion_channels

        # Network
        layer_list = [
            # Sequence 0: input = fusion channels
            OrderedDict([
                ('F1_convbatch',     lnn.layer.Conv2dBatchReLU(fusion_channels, 32, 3, 1, 1)),
                ('F2_max',           nn.MaxPool2d(2, 2)),
                ('F3_convbatch',     lnn.layer.Conv2dBatchReLU(32, 64, 3, 1, 1)),
                ('F4_max',           nn.MaxPool2d(2, 2)),
                ('F5_convbatch',     lnn.layer.Conv2dBatchReLU(64, 128, 3, 1, 1)),
                ('F6_convbatch',     lnn.layer.Conv2dBatchReLU(128, 64, 1, 1, 0)),
                ('F7_convbatch',     lnn.layer.Conv2dBatchReLU(64, 128, 3, 1, 1)),
                ('F8_max',           nn.MaxPool2d(2, 2)),
                ('F9_convbatch',     lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ('F10_convbatch',    lnn.layer.Conv2dBatchReLU(256, 128, 1, 1, 0)),
                ('F11_convbatch',    lnn.layer.Conv2dBatchReLU(128, 256, 3, 1, 1)),
                ('F12_max',          nn.MaxPool2d(2, 2)),
                ('F13_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ('F14_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0)),
                ('F15_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
                ('F16_convbatch',    lnn.layer.Conv2dBatchReLU(512, 256, 1, 1, 0)),
                ('F17_convbatch',    lnn.layer.Conv2dBatchReLU(256, 512, 3, 1, 1)),
            ]),

            # Sequence 1 : input = main channels
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
            ]),

            # Sequence 2 : input = sequence1 + sequence0
            OrderedDict([
                ('18_max',          nn.MaxPool2d(2, 2)),
                ('19_convbatch',    lnn.layer.Conv2dBatchReLU(512+512, 1024, 3, 1, 1)),
                ('20_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                ('21_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                ('22_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 512, 1, 1, 0)),
                ('23_convbatch',    lnn.layer.Conv2dBatchReLU(512, 1024, 3, 1, 1)),
                ('24_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 1024, 3, 1, 1)),
                ('25_convbatch',    lnn.layer.Conv2dBatchReLU(1024, 1024, 3, 1, 1)),
            ]),

            # Sequence 3 : input = sequence1 + sequence0
            OrderedDict([
                ('26_convbatch',    lnn.layer.Conv2dBatchReLU(512+512, 64, 1, 1, 0)),
                ('27_reorg',        lnn.layer.Reorg(2)),
            ]),

            # Sequence 4 : input = sequence3 + sequence2
            OrderedDict([
                ('28_convbatch',    lnn.layer.Conv2dBatchReLU((4*64)+1024, 1024, 3, 1, 1)),
                ('29_conv',         nn.Conv2d(1024, len(self.anchors)*(5+self.num_classes), 1, 1, 0)),
            ])
        ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

        # Post
        self.loss = lnn.loss.RegionLoss(self.num_classes, self.anchors, self.reduction)
        self.postprocess = lnd.transform.Compose([
            lnd.transform.GetBoundingBoxes(self.num_classes, self.anchors, conf_thresh),
            lnd.transform.NonMaxSupression(nms_thresh)
        ])

    def _forward(self, x):
        if x.size(1) != self.input_channels + self.fusion_channels:
            raise TypeError('This network requires {self.input_channels+self.fusion_channels} channel input images')
        main = x[:, :self.input_channels]
        fusion = x[:, self.input_channels:self.input_channels+self.fusion_channels]

        # Fusion
        out_fusion = self.layers[0](fusion)

        # Main
        out_main = self.layers[1](main)

        # Combination
        out_combo = torch.cat((out_main, out_fusion), 1)
        out2 = self.layers[2](out_combo)
        out3 = self.layers[3](out_combo)
        out = self.layers[4](torch.cat((out3, out2), 1))

        return out
