#
#   Lightnet related postprocessing
#   These are functions to transform the output of the network to brambox detection objects
#   Copyright EAVISE
#

import logging
import torch
from torch.autograd import Variable
from brambox.boxes.detections.detection import *
from .util import BaseTransform

__all__ = ['GetBoundingBoxes', 'NonMaxSupression', 'TensorToBrambox', 'ReverseLetterbox']
log = logging.getLogger(__name__)


class GetBoundingBoxes(BaseTransform):
    """ Convert output from darknet networks to bounding box tensor.

    Args:
        num_classes (int): number of categories
        anchors (list): 2D list representing anchor boxes (see :class:`lightnet.network.Darknet`)
        conf_thresh (Number [0-1]): Confidence threshold to filter detections

    Returns:
        (list [Batch x Tensor [Boxes x 6]]): **[x_center, y_center, width, height, confidence, class_id]** for every bounding box

    Note:
        The output tensor uses relative values for its coordinates.
    """
    def __init__(self, num_classes, anchors, conf_thresh):
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.anchors = torch.Tensor(anchors)
        self.num_anchors = self.anchors.shape[0]
        self.anchors_step = self.anchors.shape[1]

    def __call__(self, network_output):
        # Check dimensions
        if network_output.dim() == 3:
            network_output.unsqueeze_(0)

        # Variables
        device = network_output.device
        batch = network_output.size(0)
        h = network_output.size(2)
        w = network_output.size(3)

        # Compute xc,yc, w,h, box_score on Tensor
        lin_x = torch.linspace(0, w-1, w).repeat(h, 1).view(h*w).to(device)
        lin_y = torch.linspace(0, h-1, h).view(h, 1).repeat(1, w).view(h*w).to(device)
        anchor_w = self.anchors[:, 0].contiguous().view(1, self.num_anchors, 1).to(device)
        anchor_h = self.anchors[:, 1].contiguous().view(1, self.num_anchors, 1).to(device)

        network_output = network_output.view(batch, self.num_anchors, -1, h*w)  # -1 == 5+num_classes (we can drop feature maps if 1 class)
        network_output[:, :, 0, :].sigmoid_().add_(lin_x).div_(w)               # X center
        network_output[:, :, 1, :].sigmoid_().add_(lin_y).div_(h)               # Y center
        network_output[:, :, 2, :].exp_().mul_(anchor_w).div_(w)                # Width
        network_output[:, :, 3, :].exp_().mul_(anchor_h).div_(h)                # Height
        network_output[:, :, 4, :].sigmoid_()                                   # Box score

        # Compute class_score
        if self.num_classes > 1:
            with torch.no_grad():
                cls_scores = torch.nn.functional.softmax(network_output[:, :, 5:, :], 2)
            cls_max, cls_max_idx = torch.max(cls_scores, 2)
            cls_max_idx = cls_max_idx.float()
            cls_max.mul_(network_output[:, :, 4, :])
        else:
            cls_max = network_output[:, :, 4, :]
            cls_max_idx = torch.zeros_like(cls_max)

        score_thresh = cls_max > self.conf_thresh
        score_thresh_flat = score_thresh.view(-1)

        if score_thresh.sum() == 0:
            boxes = []
            for i in range(batch):
                boxes.append(torch.tensor([]))
            return boxes

        # Mask select boxes > conf_thresh
        coords = network_output.transpose(2, 3)[..., 0:4]
        coords = coords[score_thresh[..., None].expand_as(coords)].view(-1, 4)
        scores = cls_max[score_thresh]
        idx = cls_max_idx[score_thresh]
        detections = torch.cat([coords, scores[:, None], idx[:, None]], dim=1)

        # Get indices of splits between images of batch
        max_det_per_batch = self.num_anchors * h * w
        slices = [slice(max_det_per_batch * i, max_det_per_batch * (i+1)) for i in range(batch)]
        det_per_batch = torch.IntTensor([score_thresh_flat[s].int().sum() for s in slices])
        split_idx = torch.cumsum(det_per_batch, dim=0)

        # Group detections per image of batch
        boxes = []
        start = 0
        for end in split_idx:
            boxes.append(detections[start: end])
            start = end

        return boxes


class NonMaxSupression(BaseTransform):
    """ Performs nms on the bounding boxes, filtering boxes with a high overlap.

    Args:
        nms_thresh (Number [0-1]): Overlapping threshold to filter detections with non-maxima suppresion
        class_nms (Boolean, optional): Whether to perform nms per class; Default **True**

    Returns:
        (list [Batch x Tensor [Boxes x 6]]): **[x_center, y_center, width, height, confidence, class_id]** for every bounding box

    Note:
        This post-processing function expects the input to be bounding boxes,
        like the ones created by :class:`lightnet.data.GetBoundingBoxes` and outputs exactly the same format.
    """
    def __init__(self, nms_thresh, class_nms=True):
        self.nms_thresh = nms_thresh
        self.class_nms = class_nms

    def __call__(self, boxes):
        return [self._nms(box) for box in boxes]

    def _nms(self, boxes):
        """ Non maximum suppression.

        Args:
          boxes (tensor): Bounding boxes of one image

        Return:
          (tensor): Pruned boxes
        """
        if boxes.numel() == 0:
            return boxes

        a = boxes[:, :2]
        b = boxes[:, 2:4]
        bboxes = torch.cat([a-b/2, a+b/2], 1)
        scores = boxes[:, 4]
        classes = boxes[:, 5]

        # Sort coordinates by descending score
        scores, order = scores.sort(0, descending=True)
        x1, y1, x2, y2 = bboxes[order].split(1, 1)

        # Compute dx and dy between each pair of boxes (these mat contain every pair twice...)
        dx = (x2.min(x2.t()) - x1.max(x1.t())).clamp(min=0)
        dy = (y2.min(y2.t()) - y1.max(y1.t())).clamp(min=0)

        # Compute iou
        intersections = dx * dy
        areas = (x2 - x1) * (y2 - y1)
        unions = (areas + areas.t()) - intersections
        ious = intersections / unions

        # Filter based on iou (and class)
        conflicting = (ious > self.nms_thresh).triu(1)

        if self.class_nms:
            classes = classes[order]
            same_class = (classes.unsqueeze(0) == classes.unsqueeze(1))
            conflicting = (conflicting & same_class)

        conflicting = conflicting.cpu()
        keep = torch.zeros(len(conflicting), dtype=torch.uint8)
        supress = torch.zeros(len(conflicting), dtype=torch.float)
        for i, row in enumerate(conflicting):
            if not supress[i]:
                keep[i] = 1
                supress[row] = 1

        return boxes[order][keep[:, None].expand_as(boxes)].view(-1, 6).contiguous()


class TensorToBrambox(BaseTransform):
    """ Converts a tensor to a list of brambox objects.

    Args:
        network_size (tuple): Tuple containing the width and height of the images going in the network
        class_label_map (list, optional): List of class labels to transform the class id's in actual names; Default **None**

    Returns:
        (list [list [brambox.boxes.Detection]]): list of brambox detections per image

    Note:
        If no `class_label_map` is given, this transform will simply convert the class id's in a string.

    Note:
        Just like everything in PyTorch, this transform only works on batches of images.
        This means you need to wrap your tensor of detections in a list if you want to run this transform on a single image.
    """
    def __init__(self, network_size, class_label_map=None):
        self.width, self.height = network_size
        self.class_label_map = class_label_map
        if self.class_label_map is None:
            log.warn('No class_label_map given. The indexes will be used as class_labels.')

    def __call__(self, boxes):
        converted_boxes = []
        for box in boxes:
            if box.numel() == 0:
                converted_boxes.append([])
            else:
                converted_boxes.append(self._convert(box))
        return converted_boxes

    def _convert(self, boxes):
        boxes[:, 0:3:2].mul_(self.width)
        boxes[:, 0] -= boxes[:, 2] / 2
        boxes[:, 1:4:2].mul_(self.height)
        boxes[:, 1] -= boxes[:, 3] / 2

        brambox = []
        for box in boxes:
            det = Detection()
            det.x_top_left = box[0].item()
            det.y_top_left = box[1].item()
            det.width = box[2].item()
            det.height = box[3].item()
            det.confidence = box[4].item()
            if self.class_label_map is not None:
                det.class_label = self.class_label_map[int(box[5].item())]
            else:
                det.class_label = str(int(box[5].item()))

            brambox.append(det)

        return brambox


class ReverseLetterbox(BaseTransform):
    """ Performs a reverse letterbox operation on the bounding boxes, so they can be visualised on the original image.

    Args:
        network_size (tuple): Tuple containing the width and height of the images going in the network
        image_size (tuple): Tuple containing the width and height of the original images

    Returns:
        (list [list [brambox.boxes.Detection]]): list of brambox detections per image

    Note:
        This transform works on :class:`brambox.boxes.Detection` objects,
        so you need to apply the :class:`~lightnet.data.TensorToBrambox` transform first.

    Note:
        Just like everything in PyTorch, this transform only works on batches of images.
        This means you need to wrap your tensor of detections in a list if you want to run this transform on a single image.
    """
    def __init__(self, network_size, image_size):
        self.network_size = network_size
        self.image_size = image_size

    def __call__(self, boxes):
        im_w, im_h = image_size[:2]
        net_w, net_h = self.network_size[:2]

        if im_w == net_w and im_h == net_h:
            scale = 1
        elif im_w / net_w >= im_h / net_h:
            scale = im_w/net_w
        else:
            scale = im_h/net_h
        pad = int((net_w - im_w/scale) / 2), int((net_h - im_h/scale) / 2)

        converted_boxes = []
        for b in boxes:
            converted_boxes.append(self._transform(b, scale, pad))
        return converted_boxes

    @staticmethod
    def _transform(boxes, scale, pad):
        for box in boxes:
            box.x_top_left -= pad[0]
            box.y_top_left -= pad[1]

            box.x_top_left *= scale
            box.y_top_left *= scale
            box.width *= scale
            box.height *= scale
        return boxes
