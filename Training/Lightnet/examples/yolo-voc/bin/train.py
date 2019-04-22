#!/usr/bin/env python
import os
import logging
import time
from math import isinf
from statistics import mean
import argparse
import torch
import visdom
import numpy as np
import brambox.boxes as bbb
import lightnet as ln
from lightnet.engine import Engine
from dataset import *

log = logging.getLogger('lightnet.VOC.train')


class TrainEngine(Engine):
    def start(self):
        self.params.to(self.device)
        if self.valid_loader is not None:
            self.epoch_end()(self.test)

        self.train_loss = {'tot': [], 'coord': [], 'conf': [], 'cls': []}
        self.plot_train_loss = ln.engine.LinePlotter(self.visdom, 'train_loss', opts=dict(xlabel='Batch', ylabel='Loss', title='Training Loss', showlegend=True, legend=['Total loss', 'Coordinate loss', 'Confidence loss', 'Class loss']))
        self.plot_valid_loss = ln.engine.LinePlotter(self.visdom, 'valid_loss', name='Total loss', opts=dict(xlabel='Batch', ylabel='Loss', title='Validation Loss', showlegend=True))
        self.plot_lr = ln.engine.LinePlotter(self.visdom, 'learning_rate', name='Learning Rate', opts=dict(xlabel='Batch', ylabel='Learning Rate', title='Learning Rate Schedule'))
        self.plot_valid_pr = ln.engine.LinePlotter(self.visdom, 'valid_pr', name='latest', opts=dict(xlabel='Recall', ylabel='Precision', title='Validation PR', xtickmin=0, xtickmax=1, ytickmin=0, ytickmax=1, showlegend=True))
        self.best_map = 0

        self.dataloader.change_input_dim()
        self.optimizer.zero_grad()

    def process_batch(self, data):
        data, target = data
        data = data.to(self.device)

        loss = self.network(data, target)
        loss.backward()

        self.train_loss['tot'].append(self.network.loss.loss_tot.item())
        self.train_loss['coord'].append(self.network.loss.loss_coord.item())
        self.train_loss['conf'].append(self.network.loss.loss_conf.item())
        self.train_loss['cls'].append(self.network.loss.loss_cls.item())

    def train_batch(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

        tot = mean(self.train_loss['tot'])
        coord = mean(self.train_loss['coord'])
        conf = mean(self.train_loss['conf'])
        cls = mean(self.train_loss['cls'])
        self.train_loss = {'tot': [], 'coord': [], 'conf': [], 'cls': []}

        self.plot_train_loss(np.array([[tot, coord, conf, cls]]), np.array([self.batch]))
        self.log(f'{self.batch} Loss:{tot:.5f} (Coord:{coord:.2f} Conf:{conf:.2f} Cls:{cls:.2f})')

        if isinf(tot):
            log.error('Infinite loss')
            self.sigint = True
            return

        self.scheduler.step(self.batch, epoch=self.batch)
        self.plot_lr(np.array([self.optimizer.param_groups[0]['lr'] * self.batch_size]), np.array([self.batch]))

    @Engine.batch_end(5000)
    def backup(self):
        self.params.save(os.path.join(self.backup_folder, f'weights_{self.batch}.state.pt'))
        log.info(f'Saved backup')

    @Engine.batch_end(10)
    def resize(self):
        self.dataloader.change_input_dim()

    def test(self):
        log.info('Start testing')
        self.network.eval()
        tot_loss = 0
        anno, det = {}, {}

        with torch.no_grad():
            for idx, (data, target) in enumerate(self.valid_loader):
                data = data.to(self.device)
                output, loss = self.network(data, target)
                tot_loss += loss.item()*len(target)

                key_val = len(anno)
                anno.update({key_val+k: v for k,v in enumerate(target)})
                det.update({key_val+k: v for k,v in enumerate(output)})

                if self.sigint:
                    self.network.train()
                    return

        pr = bbb.pr(det, anno)
        m_ap = round(bbb.ap(*pr)*100, 2)
        loss = tot_loss/len(anno)
        self.log(f'Loss:{loss:.5f} mAP:{m_ap}%')
        self.plot_valid_loss(np.array([loss]), np.array([self.batch]))
        self.plot_valid_pr(np.array(pr[0]), np.array(pr[1]), update='replace')

        if m_ap > self.best_map:
            if self.best_map > 0:
                self.plot_valid_pr(None, name=f'best - {self.best_map}%', update='remove', opts=None)
            self.best_map = m_ap
            self.network.save(os.path.join(self.backup_folder, 'best_map.pt'))
            self.plot_valid_pr(np.array(pr[0]), np.array(pr[1]), name=f'best - {self.best_map}%', update='new', opts=dict(legend=[f'best - {self.best_map}%']))

        self.network.train()

    def quit(self):
        if self.batch >= self.max_batches:
            self.params.network.save(os.path.join(self.backup_folder, 'final.pt'))
            return True
        elif self.sigint:
            self.params.save(os.path.join(self.backup_folder, 'backup.state.pt'))
            return True
        else:
            return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train network')
    parser.add_argument('weight', help='Path to weight file', default=None, nargs='?')
    parser.add_argument('-n', '--network', help='network config file')
    parser.add_argument('-b', '--backup', metavar='folder', help='Backup folder', default='./backup')
    parser.add_argument('-c', '--cuda', action='store_true', help='Use cuda')
    parser.add_argument('-v', '--visdom', action='store_true', help='Visualize training data with visdom')
    parser.add_argument('-e', '--visdom_env', help='Visdom environment to plot to', default='main')
    parser.add_argument('-p', '--visdom_port', help='Port of the visdom server', type=int, default=8097)
    args = parser.parse_args()

    # Parse arguments
    device = torch.device('cpu')
    if args.cuda:
        if torch.cuda.is_available():
            log.debug('CUDA enabled')
            device = torch.device('cuda')
        else:
            log.error('CUDA not available')

    if not os.path.isdir(args.backup):
        if not os.path.exists(args.backup):
            log.warn('Backup folder does not exist, creating...')
            os.makedirs(args.backup)
        else:
            raise ValueError('Backup path is not a folder')
    
    if args.visdom:
        visdom = visdom.Visdom(port=args.visdom_port, env=args.visdom_env)
    else:
        visdom = None

    params = ln.engine.HyperParameters.from_file(args.network)
    if args.weight is not None:
        if args.weight.endswith('.state.pt'):
            params.load(args.weight)
        else:
            params.network.load(args.weight)

    # Dataloaders
    train_loader = ln.data.DataLoader(
        VOCData(params.train_set, params, True),
        batch_size = params.mini_batch_size,
        shuffle = True,
        drop_last = True,
        num_workers = 8,
        pin_memory = True,
        collate_fn = ln.data.list_collate,
    )

    if params.valid_set is not None:
        valid_loader = torch.utils.data.DataLoader(
            VOCData(params.valid_set, params, False),
            batch_size = params.mini_batch_size,
            shuffle = False,
            drop_last = False,
            num_workers = 8,
            pin_memory = True,
            collate_fn = ln.data.list_collate,
        )
    else:
        valid_loader = None

    # Start training
    eng = TrainEngine(
        params, train_loader,
        valid_loader=valid_loader, device=device, visdom=visdom, backup_folder=args.backup
    )
    b1 = eng.batch
    t1 = time.time()
    eng()
    t2 = time.time()
    b2 = eng.batch
    log.info(f'Training {b2-b1} batches took {t2-t1:.2f} seconds [{(t2-t1)/(b2-b1):.3f} sec/batch]')
