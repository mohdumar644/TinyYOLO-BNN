#!/usr/bin/env python
#
#   Copyright EAVISE
#   Example: Train the tiny yolo network using the lightnet engine
#            This example script uses darknet type annotations
#

import os
import argparse
import logging
from statistics import mean
import torch
import brambox.boxes as bbb
import lightnet as ln
from lightnet.engine import Engine

log = logging.getLogger('lightnet.train')
ln.logger.setLogFile('train.log', ('TRAIN', 'TEST'), filemode='w')
#ln.logger.setConsoleLevel(logging.DEBUG)

# Parameters
params = ln.engine.HyperParameters(
    workers = 4,
    pin_mem = True,
    trainfile = '.sandbox/data/files.data',     # File with training image paths
    validfile = '.sandbox/data/files.data',     # File with validation image paths
            
    class_labels = ['person'],
    img_size = [960, 540],
    network_size = [416, 416],
    batch_size = 64,
    mini_batch_size = 8,
    max_batches = 45000,

    _conf_thresh = 0.01,
    _nms_thresh = 0.5
)

params.network = ln.models.TinyYolo(
    num_classes=len(params.class_labels),
    conf_thresh=params.conf_thresh,
    nms_thresh=params.nms_thresh,
)
params.network.postprocess.append(ln.data.transform.TensorToBrambox(
    params.network_size,
    params.class_labels
))

params.add_optimizer(torch.optim.SGD(
    params.network.parameters(),
    lr = .001 / params.batch_size,
    momentum = .9,
    weight_decay = .0005 * params.batch_size,
    dampening = 0,
))

params.add_scheduler(torch.optim.lr_scheduler.MultiStepLR(
    params.optimizers[0],
    milestones = [25000, 35000],
    gamma = .1,
))


class TrainingEngine(Engine):
    """ This is a custom engine for this training cycle """
    def start(self):
        self.params.to(self.device)
        self.dataloader.change_input_dim()

        self.classes = len(self.class_labels)
        self.train_loss = {'tot': [], 'coord': [], 'conf': [], 'cls': []}

    def process_batch(self, data):
        data, target = data
        data = data.to(self.device)
        loss = self.network(data, target)
        loss.backward()

        self.train_loss['tot'].append(self.network.loss.loss_tot.item())
        self.train_loss['coord'].append(self.network.loss.loss_coord.item())
        self.train_loss['conf'].append(self.network.loss.loss_conf.item())
        if self.classes > 1:
            self.train_loss['cls'].append(self.network.loss.loss_cls.item())

    def train_batch(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

        tot = mean(self.train_loss['tot'])
        coord = mean(self.train_loss['coord'])
        conf = mean(self.train_loss['conf'])
        if self.classes > 1:
            cls = mean(self.train_loss['cls'])
        self.train_loss = {'tot': [], 'coord': [], 'conf': [], 'cls': []}

        if self.classes > 1:
            self.log(f'{self.batch} Loss:{tot:.5f} (Coord:{coord:.2f} Conf:{conf:.2f} Cls:{cls:.2f})')
        else:
            self.log(f'{self.batch} Loss:{tot:.5f} (Coord:{coord:.2f} Conf:{conf:.2f})')

    @Engine.batch_end(10)
    def resize(self):
        self.dataloader.change_input_dim()

    @Engine.batch_end(1000)
    def backup(self):
        self.params.save(os.path.join(self.backup_folder, f'weights_{self.batch}.state.pt'))
        log.info(f'Saved backup')

    @Engine.epoch_end()
    def test(self):
        log.info('Start testing')
        self.network.eval()
        tot_loss = []
        anno, det = {}, {}

        with torch.no_grad():
            for idx, (data, target) in enumerate(self.valid_loader):
                data = data.to(self.device)
                output, loss = self.network(data, target)
                tot_loss.append(loss.item()*len(target))

                key_val = len(anno)
                anno.update({key_val+k: v for k,v in enumerate(target)})
                det.update({key_val+k: v for k,v in enumerate(output)})

                if self.sigint:
                    self.network.train()
                    return

        m_ap = bbb.ap(*bbb.pr(det, anno)) * 100
        loss = sum(tot_loss) / len(self.valid_loader.dataset)
        self.log(f'Loss:{loss:.5f} mAP:{m_ap}%')

    def quit(self):
        if self.batch >= self.max_batches:
            self.network.save(os.path.join(self.backup_folder, 'final.pt'))
            return True
        elif self.sigint:
            self.params.save(os.path.join(self.backup_folder, 'backup.state.pt'))
            return True
        else:
            return False
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a lightnet network')
    parser.add_argument('weight', help='Path to weight file', default=None)
    parser.add_argument('-b', '--backup', help='Backup folder', default='./backup')
    parser.add_argument('-c', '--cuda', action='store_true', help='Use cuda')
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

    if args.weight is not None:
        if args.weight.endswith('.state.pt'):
            params.load(args.weight)
        else:
            params.network.load(args.weight)

    # Dataloaders
    train_loader = ln.data.DataLoader(
        ln.models.DarknetDataset(params.trainfile, input_dimension=params.network_size, class_label_map=params.class_labels),
        batch_size = params.mini_batch_size,
        shuffle = True,
        drop_last = True,
        num_workers = params.workers,
        pin_memory = params.pin_mem,
        collate_fn = ln.data.list_collate,
    )
    valid_loader = torch.utils.data.DataLoader(
        ln.models.DarknetDataset(params.validfile, False, input_dimension=params.network_size, class_label_map=params.class_labels),
        batch_size = params.mini_batch_size,
        shuffle = False,
        drop_last = False,
        num_workers = params.workers,
        pin_memory = params.pin_mem,
        collate_fn = ln.data.list_collate,
    )

    # Train
    engine = TrainEngine(params, train_loader, valid_loader=valid_loader, device=device, backup_folder=args.backup)
    engine()
