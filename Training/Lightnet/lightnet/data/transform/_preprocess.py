#
#   Image and annotations preprocessing for lightnet networks
#   The image transformations work with both Pillow and OpenCV images
#   The annotation transformations work with brambox.annotations.Annotation objects
#   Copyright EAVISE
#

import random
import collections
import logging
import torch
import math
import numpy as np
from PIL import Image, ImageOps
import brambox.boxes as bbb
from .util import BaseTransform, BaseMultiTransform

log = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    log.warn('OpenCV is not installed and cannot be used')
    cv2 = None

__all__ = ['Crop', 'Letterbox', 'RandomFlip', 'RandomHSV', 'RandomJitter', 'RandomRotate', 'BramboxToTensor']


#
#   Transform to fit
#
class Crop(BaseMultiTransform):
    """ Rescale and crop images/annotations to the right network dimensions.
    This transform will first rescale to the closest (bigger) dimension possible and then take a crop to the exact dimensions.

    Args:
        dimension (tuple, optional): Default size for the letterboxing, expressed as a (width, height) tuple; Default **None**
        dataset (lightnet.data.Dataset, optional): Dataset that uses this transform; Default **None**
        center (Boolean, optional): Whether to take the crop from the center or randomly.

    Note:
        Create 1 Crop object and use it for both image and annotation transforms.
        This object will save data from the image transform and use that on the annotation transform.
    """
    def __init__(self, dimension=None, dataset=None, center=True):
        self.dimension = dimension
        self.dataset = dataset
        self.center = center
        if self.dimension is None and self.dataset is None:
            raise ValueError('This transform either requires a dimension or a dataset to infer the dimension')

        self.scale = None
        self.dx = 0
        self.dy = 0

    def _get_crop(self, im_w, im_h, net_w, net_h):
        if im_w / net_w >= im_h / net_h:
            self.scale = net_h / im_h
            self.dy = 0
            dw = int(im_w * self.scale - net_w)
            self.dx = dw // 2 if self.center else random.randint(0, dw)
        else:
            self.scale = net_w / im_w
            self.dx = 0
            dh = int(im_h * self.scale - net_h)
            self.dy = dh // 2 if self.center else random.randint(0, dh)

    def _tf_pil(self, img):
        if self.dataset is not None:
            net_w, net_h = self.dataset.input_dim
        else:
            net_w, net_h = self.dimension
        im_w, im_h = img.size
        self._get_crop(im_w, im_h, net_w, net_h)

        # Rescale
        if self.scale != 1:
            bands = img.split()
            bands = [b.resize((int(self.scale*im_w), int(self.scale*im_h))) for b in bands]
            img = Image.merge(img.mode, bands)
            im_w, im_h = img.size

        # Crop
        img = img.crop((self.dx, self.dy, self.dx+net_w, self.dy+net_h))
        return img

    def _tf_cv(self, img):
        if self.dataset is not None:
            net_w, net_h = self.dataset.input_dim
        else:
            net_w, net_h = self.dimension
        im_h, im_w = img.shape[:2]
        self._get_crop(im_w, im_h, net_w, net_h)

        # Rescale
        if self.scale != 1:
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)

        # Crop
        img = img[dy:dy+net_h, dx:dx+net_w]
        return img

    def _tf_anno(self, annos):
        raise NotImplementedError('This transformation does not work with annotations yet. Sorry!')


class Letterbox(BaseMultiTransform):
    """ Rescale images/annotations and add top/bottom borders to get to the right network dimensions.

    Args:
        dimension (tuple, optional): Default size for the letterboxing, expressed as a (width, height) tuple; Default **None**
        dataset (lightnet.data.Dataset, optional): Dataset that uses this transform; Default **None**

    Note:
        Create 1 Letterbox object and use it for both image and annotation transforms.
        This object will save data from the image transform and use that on the annotation transform.
    """
    def __init__(self, dimension=None, dataset=None, fill_color=127):
        self.dimension = dimension
        self.dataset = dataset
        self.fill_color = fill_color
        if self.dimension is None and self.dataset is None:
            raise ValueError('This transform either requires a dimension or a dataset to infer the dimension')

        self.pad = None
        self.scale = None

    def _tf_pil(self, img):
        """ Letterbox an image to fit in the network """
        if self.dataset is not None:
            net_w, net_h = self.dataset.input_dim
        else:
            net_w, net_h = self.dimension
        im_w, im_h = img.size

        if im_w == net_w and im_h == net_h:
            self.scale = None
            self.pad = None
            return img

        # Rescaling
        if im_w / net_w >= im_h / net_h:
            self.scale = net_w / im_w
        else:
            self.scale = net_h / im_h
        if self.scale != 1:
            bands = img.split()
            bands = [b.resize((int(self.scale*im_w), int(self.scale*im_h))) for b in bands]
            img = Image.merge(img.mode, bands)
            im_w, im_h = img.size

        if im_w == net_w and im_h == net_h:
            self.pad = None
            return img

        # Padding
        img_np = np.array(img)
        channels = img_np.shape[2] if len(img_np.shape) > 2 else 1
        pad_w = (net_w - im_w) / 2
        pad_h = (net_h - im_h) / 2
        self.pad = (int(pad_w), int(pad_h), int(pad_w+.5), int(pad_h+.5))
        img = ImageOps.expand(img, border=self.pad, fill=(self.fill_color,)*channels)
        return img

    def _tf_cv(self, img):
        """ Letterbox and image to fit in the network """
        if self.dataset is not None:
            net_w, net_h = self.dataset.input_dim
        else:
            net_w, net_h = self.dimension
        im_h, im_w = img.shape[:2]

        if im_w == net_w and im_h == net_h:
            self.scale = None
            self.pad = None
            return img

        # Rescaling
        if im_w / net_w >= im_h / net_h:
            self.scale = net_w / im_w
        else:
            self.scale = net_h / im_h
        if self.scale != 1:
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
            im_h, im_w = img.shape[:2]

        if im_w == net_w and im_h == net_h:
            self.pad = None
            return img

        # Padding
        channels = img.shape[2] if len(img.shape) > 2 else 1
        pad_w = (net_w - im_w) / 2
        pad_h = (net_h - im_h) / 2
        self.pad = (int(pad_w), int(pad_h), int(pad_w+.5), int(pad_h+.5))
        img = cv2.copyMakeBorder(img, self.pad[1], self.pad[3], self.pad[0], self.pad[2], cv2.BORDER_CONSTANT, value=(self.fill_color,)*channels)
        return img

    def _tf_anno(self, annos):
        """ Change coordinates of an annotation, according to the previous letterboxing """
        for anno in annos:
            if self.scale is not None:
                anno.x_top_left *= self.scale
                anno.y_top_left *= self.scale
                anno.width *= self.scale
                anno.height *= self.scale
            if self.pad is not None:
                anno.x_top_left += self.pad[0]
                anno.y_top_left += self.pad[1]
        return annos


#
#   Data augmentation
#
class RandomFlip(BaseMultiTransform):
    """ Randomly flip image.

    Args:
        threshold (Number [0-1]): Chance of flipping the image

    Note:
        Create 1 RandomFlip object and use it for both image and annotation transforms.
        This object will save data from the image transform and use that on the annotation transform.
    """
    def __init__(self, threshold):
        self.threshold = threshold
        self.flip = False
        self.im_w = None

    def _get_flip(self):
        self.flip = random.random() < self.threshold

    def _tf_pil(self, img):
        """ Randomly flip image """
        self._get_flip()
        self.im_w = img.size[0]
        if self.flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def _tf_cv(self, img):
        """ Randomly flip image """
        self._get_flip()
        self.im_w = img.shape[1]
        if self.flip:
            img = cv2.flip(img, 1)
        return img

    def _tf_anno(self, annos):
        """ Change coordinates of an annotation, according to the previous flip """
        if self.flip and self.im_w is not None:
            for anno in annos:
                anno.x_top_left = self.im_w - anno.x_top_left - anno.width

        return annos


class RandomHSV(BaseTransform):
    """ Perform random HSV shift on the RGB data.

    Args:
        hue (Number): Random number between -hue,hue is used to shift the hue
        saturation (Number): Random number between 1,saturation is used to shift the saturation; 50% chance to get 1/dSaturation in stead of dSaturation
        value (Number): Random number between 1,value is used to shift the value; 50% chance to get 1/dValue in stead of dValue

    Warning:
        If you use OpenCV as your image processing library, make sure the image is RGB before using this transform.
        By default OpenCV uses BGR, so you must use `cvtColor`_ function to transform it to RGB.

    .. _cvtColor: https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#ga397ae87e1288a81d2363b61574eb8cab
    """
    def __init__(self, hue, saturation, value):
        self.hue = hue
        self.saturation = saturation
        self.value = value

    def __call__(self, data):
        dh = random.uniform(-self.hue, self.hue)
        ds = random.uniform(1, self.saturation)
        if random.random() < 0.5:
            ds = 1/ds
        dv = random.uniform(1, self.value)
        if random.random() < 0.5:
            dv = 1/dv

        if data is None:
            return None
        elif isinstance(data, Image.Image):
            return self._tf_pil(data, dh, ds, dv)
        elif isinstance(data, np.ndarray):
            return self._tf_cv(data, dh, ds, dv)
        else:
            log.error(f'HSVShift only works with <PIL images> or <OpenCV images> [{type(data)}]')
            return data     # Pass on data to not destroy pipeline with annos

    @staticmethod
    def _tf_pil(img, dh, ds, dv):
        """ Random hsv shift """
        img = img.convert('HSV')
        channels = list(img.split())

        def wrap_hue(x):
            x += int(dh * 255)
            if x > 255:
                x -= 255
            elif x < 0:
                x += 255
            return x

        channels[0] = channels[0].point(wrap_hue)
        channels[1] = channels[1].point(lambda i: min(255, max(0, int(i*ds))))
        channels[2] = channels[2].point(lambda i: min(255, max(0, int(i*dv))))

        img = Image.merge(img.mode, tuple(channels))
        img = img.convert('RGB')
        return img

    @staticmethod
    def _tf_cv(img, dh, ds, dv):
        """ Random hsv shift """
        img = img.astype(np.float32) / 255.0
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        def wrap_hue(x):
            x[x >= 360.0] -= 360.0
            x[x < 0.0] += 360.0
            return x

        img[:, :, 0] = wrap_hue(hsv[:, :, 0] + (360.0 * dh))
        img[:, :, 1] = np.clip(ds * img[:, :, 1], 0.0, 1.0)
        img[:, :, 2] = np.clip(dv * img[:, :, 2], 0.0, 1.0)

        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        img = (img * 255).astype(np.uint8)
        return img


class RandomJitter(BaseMultiTransform):
    """ Add random jitter to an image, by randomly cropping (or adding borders) to each side.

    Args:
        jitter (Number [0-1]): Indicates how much of the image we can crop
        crop_anno(Boolean, optional): Whether we crop the annotations inside the image crop; Default **False**
        intersection_threshold(number or list, optional): Argument passed on to :class:`brambox.boxes.util.modifiers.CropModifier`

    Note:
        Create 1 RandomCrop object and use it for both image and annotation transforms.
        This object will save data from the image transform and use that on the annotation transform.
    """
    def __init__(self, jitter, crop_anno=False, intersection_threshold=0.001, fill_color=127):
        self.jitter = jitter
        self.crop_anno = crop_anno
        self.fill_color = fill_color
        self.crop_modifier = bbb.CropModifier(float('Inf'), intersection_threshold)

    def _get_crop(self, im_w, im_h):
        dw, dh = int(im_w*self.jitter), int(im_h*self.jitter)
        crop_left = random.randint(-dw, dw)
        crop_right = random.randint(-dw, dw)
        crop_top = random.randint(-dh, dh)
        crop_bottom = random.randint(-dh, dh)
        crop = (crop_left, crop_top, im_w-crop_right, im_h-crop_bottom)

        self.crop_modifier.area = crop
        return crop

    def _tf_pil(self, img):
        """ Take random crop from image """
        im_w, im_h = img.size
        crop = self._get_crop(im_w, im_h)
        crop_w = crop[2] - crop[0]
        crop_h = crop[3] - crop[1]
        img_np = np.array(img)
        channels = img_np.shape[2] if len(img_np.shape) > 2 else 1

        img = img.crop((max(0, crop[0]), max(0, crop[1]), min(im_w, crop[2]-1), min(im_h, crop[3]-1)))
        img_crop = Image.new(img.mode, (crop_w, crop_h), color=(self.fill_color,)*channels)
        img_crop.paste(img, (max(0, -crop[0]), max(0, -crop[1])))

        return img_crop

    def _tf_cv(self, img):
        """ Take random crop from image """
        im_h, im_w = img.shape[:2]
        crop = self._get_crop(im_w, im_h)

        crop_w = crop[2] - crop[0]
        crop_h = crop[3] - crop[1]
        img_crop = np.ones((crop_h, crop_w) + img.shape[2:], dtype=img.dtype) * self.fill_color

        src_x1 = max(0, crop[0])
        src_x2 = min(crop[2], im_w)
        src_y1 = max(0, crop[1])
        src_y2 = min(crop[3], im_h)
        dst_x1 = max(0, -crop[0])
        dst_x2 = crop_w - max(0, crop[2]-im_w)
        dst_y1 = max(0, -crop[1])
        dst_y2 = crop_h - max(0, crop[3]-im_h)
        img_crop[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]

        return img_crop

    def _tf_anno(self, annos):
        """ Change coordinates of an annotation, according to the previous crop """
        if self.crop_anno:
            bbb.modify(annos, [self.crop_modifier])
        else:
            crop = self.crop_modifier.area
            for i in range(len(annos)-1, -1, -1):
                anno = annos[i]
                x1 = max(crop[0], anno.x_top_left)
                x2 = min(crop[2], anno.x_top_left+anno.width)
                y1 = max(crop[1], anno.y_top_left)
                y2 = min(crop[3], anno.y_top_left+anno.height)
                w = x2-x1
                h = y2-y1

                if self.crop_modifier.inter_area:
                    ratio = ((w * h) / (anno.width * anno.height)) < self.crop_modifier.inter_thresh
                else:
                    ratio = (w / anno.width) < self.crop_modifier.inter_thresh[0] or (h / anno.height) < self.crop_modifier.inter_thresh[1]
                if w <= 0 or h <= 0 or ratio:
                    del annos[i]
                    continue

                annos[i].x_top_left -= crop[0]
                annos[i].y_top_left -= crop[1]

        return annos


class RandomRotate(BaseMultiTransform):
    """ Randomly rotate the image/annotations.
    For the annotations we take the smallest possible rectangle that fits the rotated rectangle.

    Args:
        jitter (Number [0-180]): Random number between -jitter,jitter degrees is used to rotate the image

    Note:
        Create 1 RandomRotate object and use it for both image and annotation transforms.
        This object will save data from the image transform and use that on the annotation transform.
    """
    def __init__(self, jitter):
        self.jitter = jitter
        self.angle = None
        self.im_w = None
        self.im_h = None

    def _get_rotate(self, im_w, im_h):
        self.im_w = im_w
        self.im_h = im_h
        self.angle = random.randint(-self.jitter, self.jitter)

    def _tf_pil(self, img):
        im_w, im_h = img.size
        self._get_rotate(im_w, im_h)
        return img.rotate(self.angle)

    def _tf_cv(self, img):
        im_h, im_w = img.shape[:2]
        self._get_rotate(im_w, im_h)
        M = cv2.getRotationMatrix2D((im_w/2, im_h/2), self.angle, 1)
        return cv2.warpAffine(img, M, (im_w, im_h))

    def _tf_anno(self, annos):
        cx, cy = self.im_w/2, self.im_h/2
        rad = math.radians(-self.angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        for anno in annos:
            # Rotate anno
            x1_c = anno.x_top_left - cx
            y1_c = anno.y_top_left - cy
            x2_c = x1_c + anno.width
            y2_c = y1_c + anno.height

            x1_r = (x1_c * cos_a - y1_c * sin_a) + cx
            y1_r = (x1_c * sin_a + y1_c * cos_a) + cy
            x2_r = (x2_c * cos_a - y1_c * sin_a) + cx
            y2_r = (x2_c * sin_a + y1_c * cos_a) + cy
            x3_r = (x2_c * cos_a - y2_c * sin_a) + cx
            y3_r = (x2_c * sin_a + y2_c * cos_a) + cy
            x4_r = (x1_c * cos_a - y2_c * sin_a) + cx
            y4_r = (x1_c * sin_a + y2_c * cos_a) + cy

            # Max rect box
            x1_n = min(x1_r, x2_r, x3_r, x4_r)
            y1_n = min(y1_r, y2_r, y3_r, y4_r)
            x2_n = max(x1_r, x2_r, x3_r, x4_r)
            y2_n = max(y1_r, y2_r, y3_r, y4_r)

            anno.x_top_left = x1_n
            anno.y_top_left = y1_n
            anno.width = x2_n - x1_n
            anno.height = y2_n - y1_n

        return annos


#
#   Util
#
class BramboxToTensor(BaseTransform):
    """ Converts a list of brambox annotation objects to a tensor.

    Args:
        dimension (tuple, optional): Default size of the transformed images, expressed as a (width, height) tuple; Default **None**
        dataset (lightnet.data.Dataset, optional): Dataset that uses this transform; Default **None**
        max_anno (Number, optional): Maximum number of annotations in the list; Default **50**
        class_label_map (list, optional): class label map to convert class names to an index; Default **None**

    Return:
        torch.Tensor: tensor of dimension [max_anno, 5] containing [class_idx,center_x,center_y,width,height] for every detection

    Warning:
        If no class_label_map is given, this function will first try to convert the class_label to an integer. If that fails, it is simply given the number 0.
    """
    def __init__(self, dimension=None, dataset=None, max_anno=50, class_label_map=None):
        self.dimension = dimension
        self.dataset = dataset
        self.max_anno = max_anno
        self.class_label_map = class_label_map

        if self.dimension is None and self.dataset is None:
            raise ValueError('This transform either requires a dimension or a dataset to infer the dimension')
        if self.class_label_map is None:
            log.warn('No class_label_map given. If the class_labels are not integers, they will be set to zero.')

    def __call__(self, data):
        if self.dataset is not None:
            dim = self.dataset.input_dim
        else:
            dim = self.dimension
        return self.apply(data, dim, self.max_anno, self.class_label_map)

    @classmethod
    def apply(cls, data, dimension, max_anno=None, class_label_map=None):
        if not isinstance(data, collections.Sequence):
            raise TypeError(f'BramboxToTensor only works with <brambox annotation list> [{type(data)}]')

        anno_np = np.array([cls._tf_anno(anno, dimension, class_label_map) for anno in data], dtype=np.float32)

        if max_anno is not None:
            anno_len = len(data)
            if anno_len > max_anno:
                raise ValueError(f'More annotations than maximum allowed [{anno_len}/{max_anno}]')

            z_np = np.zeros((max_anno-anno_len, 5), dtype=np.float32)
            z_np[:, 0] = -1

            if anno_len > 0:
                return torch.from_numpy(np.concatenate((anno_np, z_np)))
            else:
                return torch.from_numpy(z_np)
        else:
            return torch.from_numpy(anno_np)

    @staticmethod
    def _tf_anno(anno, dimension, class_label_map):
        """ Transforms brambox annotation to list """
        net_w, net_h = dimension

        if class_label_map is not None:
            cls = class_label_map.index(anno.class_label)
        else:
            try:
                cls = int(anno.class_label)
            except ValueError:
                cls = 0

        cx = (anno.x_top_left + (anno.width / 2)) / net_w
        cy = (anno.y_top_left + (anno.height / 2)) / net_h
        w = anno.width / net_w
        h = anno.height / net_h
        return [cls, cx, cy, w, h]
