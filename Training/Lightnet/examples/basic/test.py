#!/usr/bin/env python
#
#   Copyright EAVISE
#   Example: Test the lightnet tiny yolo network on a test data set and compute the PR/mAP metric
#            This example script uses darknet type annotations
#            For simplicity, this script does not use the HyperParameters class, though in practice it might be easier to do so.
#            Taks a look at the yolo-voc example to see how to separate your HyperParameters in a config file.
#

import os
import argparse
import logging
from pathlib import Path
from statistics import mean
import numpy as np
from tqdm import tqdm
import torch
import brambox.boxes as bbb
import lightnet as ln

log = logging.getLogger('lightnet.test')
ln.logger.setLogFile('test.log', filemode='w')
#ln.logger.setConsoleLevel(logging.DEBUG)

# Parameters
WORKERS = 4
PIN_MEM = True
TESTFILE = '.sandbox/data/files.data'   # Testing dataset files

CLASS_LABELS = ['person']
NETWORK_SIZE = [416, 416]
CONF_THRESH = 0.01
NMS_THRESH = 0.5

BATCH = 64 
MINI_BATCH = 8


def test(weight, device, save_det):
    log.debug('Creating network')
    net = ln.models.TinyYolo(len(CLASS_LABELS), weight, CONF_THRESH, NMS_THRESH)
    net.postprocess.append(ln.data.transform.TensorToBrambox(NETWORK_SIZE, CLASS_LABELS))
    net = net.to(device)
    net.eval()

    log.debug('Creating dataset')
    loader = torch.utils.data.DataLoader(
        ln.models.DarknetDataset(TESTFILE, augment=False, input_dimension=NETWORK_SIZE, class_label_map=CLASS_LABELS),
        batch_size = MINI_BATCH,
        shuffle = False,
        drop_last = False,
        num_workers = WORKERS,
        pin_memory = PIN_MEM,
        collate_fn = ln.data.list_collate,
    )

    log.debug('Running network')
    tot_loss = []
    coord_loss = []
    conf_loss = []
    cls_loss = []
    anno, det = {}, {}
    num_det = 0

    with torch.no_grad():
        for idx, (data, box) in enumerate(tqdm(loader, total=len(loader))):
            data = data.to(device)
            output, loss = net(data, box)

            tot_loss.append(net.loss.loss_tot.item()*len(box))
            coord_loss.append(net.loss.loss_coord.item()*len(box))
            conf_loss.append(net.loss.loss_conf.item()*len(box))
            if net.loss.loss_cls is not None:
                cls_loss.append(net.loss.loss_cls.item()*len(box))

            key_val = len(anno)
            anno.update({loader.dataset.keys[key_val+k]: v for k,v in enumerate(box)})
            det.update({loader.dataset.keys[key_val+k]: v for k,v in enumerate(output)})

    log.debug('Computing statistics')

    pr = bbb.pr(det, anno)
    m_ap = bbb.ap(*pr)*100
    tot = sum(tot_loss)/len(anno)
    coord = sum(coord_loss)/len(anno)
    conf = sum(conf_loss)/len(anno)
    if len(cls_loss) > 0:
        cls = sum(cls_loss)/len(anno)
        log.test(f'{net.seen//BATCH} mAP:{m_ap:.2f}% Loss:{tot:.5f} (Coord:{coord:.2f} Conf:{conf:.2f} Cls:{cls:.2f})')
    else:
        log.test(f'{net.seen//BATCH} mAP:{m_ap:.2f}% Loss:{tot:.5f} (Coord:{coord:.2f} Conf:{conf:.2f})')

    if save_det is not None:
        # Note: These detection boxes are the coordinates for the letterboxed images,
        #       you need ln.data.transform.ReverseLetterbox to have the right ones.
        #       Alternatively, you can save the letterboxed annotations, and use those for statistics later on!
        bbb.generate('det_pickle', det, Path(arguments.save_det).with_suffix('.pkl'))
        #bbb.generate('anno_pickle', det, Path('anno-letterboxed_'+arguments.save_det).with_suffix('.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a lightnet network')
    parser.add_argument('weight', help='Path to weight file', default=None)
    parser.add_argument('-c', '--cuda', action='store_true', help='Use cuda')
    parser.add_argument('-s', '--save_det', help='Save detections as a brambox pickle file', default=None)
    args = parser.parse_args()

    # Parse arguments
    device = torch.device('cpu')
    if args.cuda:
        if torch.cuda.is_available():
            log.debug('CUDA enabled')
            device = torch.device('cuda')
        else:
            log.error('CUDA not available')

    # Test
    test(args.weight, device, args.save_det)
