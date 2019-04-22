#!/usr/bin/env python
import argparse
import logging
import copy
import time
from pathlib import Path
from statistics import mean
import torch
import numpy as np
from tqdm import tqdm
import lightnet as ln
import brambox.boxes as bbb
from dataset import *

log = logging.getLogger('lightnet.VOC.test')


class TestEngine:
    workers = 8
    pin_mem = True
    coco_metric = False

    def __init__(self, params, **kwargs):
        self.params = params
        self.device = kwargs['device']
        self.loss = kwargs['loss']
        self.fast_pr = kwargs['fast_pr']
        self.network = params.network
        self.network.eval()
        self.network.to(self.device)

        self.test_dataloader = torch.utils.data.DataLoader(
            VOCData(params.test_set, params, False),
            batch_size = params.mini_batch_size,
            shuffle = False,
            drop_last = False,
            num_workers = 8,
            pin_memory = True,
            collate_fn = ln.data.list_collate,
        )

    def __call__(self, csv_file):
        if self.loss == 'none':
            anno, det = self.test_none()
        else:
            anno, det = self.test_loss()

        if self.coco_metric:
            m_ap = []
            for i in range(50, 95, 5):
                m_ap.append(self.ap(det, anno, i/100))
            m_ap = round(mean(m_ap), 2)
            if self.csv_file:
                log.error('CSV file is not possible with the coco metric')
        else:
            m_ap = self.ap(det, anno, csv=csv_file)

        print(f'mAP: {m_ap:.2f}%')

    def ap(self, det, anno, iou=.5, csv=None):
        if csv is not None:
            base_path = Path(csv)

        if self.fast_pr:
            pr = bbb.pr(det, anno, iou)

            if csv is not None:
                np.savetxt(str(base_path), np.array(pr), delimiter=',')

            return round(100 * bbb.ap(*pr), 2)
        else:
            aps = []
            for c in tqdm(self.params.class_label_map):
                anno_c = bbb.filter_discard(copy.deepcopy(anno), [ lambda a: a.class_label == c ])
                det_c  = bbb.filter_discard(copy.deepcopy(det), [ lambda d: d.class_label == c ])
                pr = bbb.pr(det_c, anno_c)

                if csv is not None:
                    np.savetxt(str(base_path.with_name(base_path.stem + f'_{c}' + base_path.suffix)), np.array(pr), delimiter=',')

                aps.append(bbb.ap(*pr))

            return round(100 * mean(aps), 2)

    def test_none(self):
        anno, det = {}, {}

        pp = self.network.postprocess
        self.network.postprocess = None

        with torch.no_grad():
            t_net = 0
            t_pp = 0
            for idx, (data, target) in enumerate(self.test_dataloader):
                data = data.to(self.device)
                t1 = time.perf_counter()
                output = self.network(data)
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                output = pp(output)
                torch.cuda.synchronize()
                t3 = time.perf_counter()

                t_net += t2 - t1
                t_pp += t3 - t2

                base_idx = idx*self.params.mini_batch_size
                anno.update({self.test_dataloader.dataset.keys[base_idx+k]: v for k,v in enumerate(target)})
                det.update({self.test_dataloader.dataset.keys[base_idx+k]: v for k,v in enumerate(output)})


            t_net_img = t_net * 1000 / len(self.test_dataloader.dataset)
            t_pp_img = t_pp * 1000 / len(self.test_dataloader.dataset)
            t_tot_img = t_net_img + t_pp_img
            log.info(f'Time:{t_tot_img:.2f}ms/img (Network:{t_net_img:.3f} Post:{t_pp_img:.3f})')

            self.network.postprocess = pp

            return anno, det

    def test_loss(self):
        loss_dict = {'tot': [], 'coord': [], 'conf': [], 'cls': []}
        anno, det = {}, {}

        with torch.no_grad():
            for idx, (data, target) in tqdm(enumerate(self.test_dataloader), total=len(self.test_dataloader)):
                data = data.to(self.device)
                output, loss = self.network(data, target)

                loss_dict['tot'].append(self.network.loss.loss_tot.item()*len(target))
                loss_dict['coord'].append(self.network.loss.loss_coord.item()*len(target))
                loss_dict['conf'].append(self.network.loss.loss_conf.item()*len(target))
                loss_dict['cls'].append(self.network.loss.loss_cls.item()*len(target))
                base_idx = idx*self.params.mini_batch_size
                anno.update({self.test_dataloader.dataset.keys[base_idx+k]: v for k,v in enumerate(target)})
                det.update({self.test_dataloader.dataset.keys[base_idx+k]: v for k,v in enumerate(output)})

        loss_tot = sum(loss_dict['tot'])/len(anno)
        loss_coord = sum(loss_dict['coord'])/len(anno)
        loss_conf = sum(loss_dict['conf'])/len(anno)
        loss_cls = sum(loss_dict['cls'])/len(anno)
        if self.loss == 'percent':
            loss_coord *= 100 / loss_tot
            loss_conf *= 100 / loss_tot
            loss_cls *= 100 / loss_tot
            log.info(f'Loss:{loss_tot:.5f} (Coord:{loss_coord:.2f}% Conf:{loss_conf:.2f}% Class:{loss_cls:.2f}%)')
        else:
            log.info(f'Loss:{loss_tot:.5f} (Coord:{loss_coord:.2f} Conf:{loss_conf:.2f} Class:{loss_cls:.2f})')

        return anno, det


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test trained network')
    parser.add_argument('weight', help='Path to weight file', default=None)
    parser.add_argument('--csv', help='Path for the csv file with the results', default=None)
    parser.add_argument('-n', '--network', help='network config file')
    parser.add_argument('-c', '--cuda', action='store_true', help='Use cuda')
    parser.add_argument('-f', '--fast-pr', action='store_true', help='Use faster but less accurate PR computation method')
    parser.add_argument('-l', '--loss', help='How to display loss', choices=['abs', 'percent', 'none'], default='abs')
    parser.add_argument('-t', '--thresh', help='Detection Threshold', type=float, default=None)
    parser.add_argument('-s', '--save', help='File to store network weights', default=None)

    args = parser.parse_args()

    device = torch.device('cpu')
    if args.cuda:
        if torch.cuda.is_available():
            log.debug('CUDA enabled')
            device = torch.device('cuda')
        else:
            log.error('CUDA not available')

    params = ln.engine.HyperParameters.from_file(args.network)
    if args.weight is not None:
        if args.weight.endswith('.state.pt'):
            params.load(args.weight)
        else:
            params.network.load(args.weight)
    if args.thresh is not None:
        params.network.postprocess[0].conf_thresh = args.thresh

    if args.save is not None:
        params.network.save(args.save)

    # Start test
    eng = TestEngine(params, device=device, loss=args.loss, fast_pr=args.fast_pr)
    eng(args.csv)
