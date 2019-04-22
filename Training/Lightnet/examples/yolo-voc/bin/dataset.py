import copy
import logging
from PIL import Image
import torch
from torchvision import transforms as tf
import brambox.boxes as bbb
import lightnet as ln

__all__ = ['VOCData']
log = logging.getLogger('lightnet.VOC.dataset')

def identify_file(img_id):
    return 'data/' + img_id + '.png'

class VOCData(ln.models.BramboxDataset):
    def __init__(self, anno, params, augment):
        def identify(img_id):
            return f'data/VOCdevkit/{img_id}.jpg'

        self.filter = params.filter_anno
        if not self.filter in ('ignore', 'rm', 'none'):
            log.error(f'{self.filter} is not one of (ignore, rm, none). Choosing default "none" value')

        lb  = ln.data.transform.Letterbox(dataset=self)
        img_tf = ln.data.transform.Compose([lb, tf.ToTensor()])
        anno_tf = ln.data.transform.Compose([lb])

        if augment:
            rf  = ln.data.transform.RandomFlip(params.flip)
            rc  = ln.data.transform.RandomJitter(params.jitter, True, 0.1)
            hsv = ln.data.transform.RandomHSV(params.hue, params.saturation, params.value)
            img_tf[0:0] = [hsv, rc, rf]
            anno_tf[0:0] = [rc, rf]

        super().__init__('anno_pickle', anno, params.input_dimension, params.class_label_map, identify, img_tf, anno_tf)

    @ln.models.BramboxDataset.resize_getitem
    def __getitem__(self, index):
        img, anno = super().__getitem__(index)

        # Ignore difficult annotations
        if self.filter == 'ignore':
            for a in anno:
                a.ignore = a.difficult
        elif self.filter == 'rm':
            for i in range(len(anno)-1, -1, -1):
                if anno[i].difficult:
                    del anno[i]

        return img, anno
