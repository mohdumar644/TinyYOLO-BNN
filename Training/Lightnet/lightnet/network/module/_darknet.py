#
#   Base network structure for Darknet networks
#   Copyright EAVISE
#

import os
import collections
import logging
import numpy as np
import torch
import torch.nn as nn
from ._lightnet import Lightnet
from ..layer._darknet import *

__all__ = ['Darknet']
log = logging.getLogger(__name__)


class Darknet(Lightnet):
    """ This network module provides functionality to load darknet weight files.
    """
    def __init__(self):
        super().__init__()
        self.header = [0, 2, 0]

    def load(self, weights_file, strict=False):
        """ This function will load the weights from a file.
        If the file extension is ``.pt``, it will be considered as a `pytorch pickle file <http://pytorch.org/docs/stable/notes/serialization.html#recommended-approach-for-saving-a-model>`_.
        Otherwise, the file is considered to be a darknet binary weight file.

        Args:
            weights_file (str): path to file
            strict (Boolean, optional): Whether the weight file should contain all layers of the model; Default **False**

        Note:
            The ``strict`` parameter only works for pytorch pickle files.
            See :class:`~lightnet.network.module.Lightnet` for more information.
        """
        if os.path.splitext(weights_file)[1] == '.pt':
            log.debug('Loading weights from pytorch file')
            super().load(weights_file, strict)
        else:
            log.debug('Loading weights from darknet file')
            self._load_darknet_weights(weights_file)

    def save(self, weights_file, remap=None):
        """ This function will save the weights to a file.
        If the file extension is ``.pt``, it will be considered as a `pytorch pickle file <http://pytorch.org/docs/stable/notes/serialization.html#recommended-approach-for-saving-a-model>`_.
        Otherwise, the file is considered to be a darknet binary weight file.

        Args:
            weights_file (str): path to file
            remap (list, optional): list of remapping tuples, to be able to use the weights from one model in another; Default **None**

        Note:
            The ``remap`` parameter only works for pytorch pickle files.
            See :class:`~lightnet.network.module.Lightnet` for more information.
        """
        if os.path.splitext(weights_file)[1] == '.pt':
            log.debug('Saving weights to pytorch file')
            super().save(weights_file, remap)
        else:
            log.debug('Saving weights to darknet file')
            self._save_darknet_weights(weights_file)

    def _load_darknet_weights(self, weights_file):
        weights = WeightLoader(weights_file)
        self.header = weights.header
        if hasattr(self.loss, 'seen'):
            self.loss.seen = torch.tensor(weights.seen)

        for module in self.layer_loop():
            try:
                weights.load_layer(module)
                log.debug(f'Layer loaded: {module}')
                if weights.start >= weights.size:
                    log.debug(f'Finished loading weights [{weights.start}/{weights.size} weights]')
                    break
            except NotImplementedError:
                log.debug(f'Layer skipped: {module.__class__.__name__}')

    def _save_darknet_weights(self, weights_file):
        if hasattr(self.loss, 'seen'):
            seen = self.loss.seen.item()
        else:
            seen = 0
        weights = WeightSaver(self.header, seen)

        for module in self.layer_loop():
            try:
                weights.save_layer(module)
                log.debug(f'Layer saved: {module}')
            except NotImplementedError:
                log.debug(f'Layer skipped: {module.__class__.__name__}')

        weights.write_file(weights_file)


class WeightLoader:
    """ Load darknet weight files into pytorch layers """
    def __init__(self, filename):
        with open(filename, 'rb') as fp:
            self.header = np.fromfile(fp, count=3, dtype=np.int32).tolist()
            ver_num = self.header[0]*100+self.header[1]*10+self.header[2]
            log.debug(f'Loading weight file: version {self.header[0]}.{self.header[1]}.{self.header[2]}')

            if ver_num <= 19:
                log.warn('Weight file uses sizeof to compute variable size, which might lead to undefined behaviour. (choosing int=int32, float=float32)')
                self.seen = int(np.fromfile(fp, count=1, dtype=np.int32)[0])
            elif ver_num <= 29:
                log.warn('Weight file uses sizeof to compute variable size, which might lead to undefined behaviour. (choosing int=int32, float=float32, size_t=int64)')
                self.seen = int(np.fromfile(fp, count=1, dtype=np.int64)[0])
            else:
                log.error('New weight file syntax! Loading of weights might not work properly. Please submit an issue with the weight file version number. [Run with DEBUG logging level]')
                self.seen = int(np.fromfile(fp, count=1, dtype=np.int64)[0])

            self.buf = np.fromfile(fp, dtype=np.float32)

        self.start = 0
        self.size = self.buf.size

    def load_layer(self, layer):
        """ Load weights for a layer from the weights file """
        if type(layer) == nn.Conv2d:
            self._load_conv(layer)
        elif type(layer) == Conv2dBatchReLU:
            self._load_convbatch(layer)
        elif type(layer) == nn.Linear:
            self._load_fc(layer)
        else:
            raise NotImplementedError(f'The layer you are trying to load is not supported [{type(layer)}]')

    def _load_conv(self, model):
        num_b = model.bias.numel()
        model.bias.data.copy_(torch.from_numpy(self.buf[self.start:self.start+num_b])
                                   .view_as(model.bias.data))
        self.start += num_b

        num_w = model.weight.numel()
        model.weight.data.copy_(torch.from_numpy(self.buf[self.start:self.start+num_w])
                                     .view_as(model.weight.data))
        self.start += num_w

    def _load_convbatch(self, model):
        num_b = model.layers[1].bias.numel()
        model.layers[1].bias.data.copy_(torch.from_numpy(self.buf[self.start:self.start+num_b])
                                             .view_as(model.layers[1].bias.data))
        self.start += num_b
        model.layers[1].weight.data.copy_(torch.from_numpy(self.buf[self.start:self.start+num_b])
                                               .view_as(model.layers[1].weight.data))
        self.start += num_b
        model.layers[1].running_mean.copy_(torch.from_numpy(self.buf[self.start:self.start+num_b])
                                                .view_as(model.layers[1].running_mean))
        self.start += num_b
        model.layers[1].running_var.copy_(torch.from_numpy(self.buf[self.start:self.start+num_b])
                                               .view_as(model.layers[1].running_var))
        self.start += num_b

        num_w = model.layers[0].weight.numel()
        model.layers[0].weight.data.copy_(torch.from_numpy(self.buf[self.start:self.start+num_w])
                                               .view_as(model.layers[0].weight.data))
        self.start += num_w

    def _load_fc(self, model):
        num_b = model.bias.numel()
        model.bias.data.copy_(torch.from_numpy(self.buf[self.start:self.start+num_b])
                                   .view_as(model.bias.data))
        self.start += num_b

        num_w = model.weight.numel()
        model.weight.data.copy_(torch.from_numpy(self.buf[self.start:self.start+num_w])
                                     .view_as(model.weight.data))
        self.start += num_w


class WeightSaver:
    """ Save darknet weight files from pytorch layers """
    def __init__(self, header, seen):
        self.weights = []
        self.header = np.array(header, dtype=np.int32)
        ver_num = self.header[0]*100+self.header[1]*10+self.header[2]
        if ver_num <= 19:
            self.seen = np.int32(seen)
        elif ver_num <= 29:
            self.seen = np.int64(seen)
        else:
            log.error('New weight file syntax! Saving of weights might not work properly. Please submit an issue with the weight file version number. [Run with DEBUG logging level]')
            self.seen = np.int64(seen)

    def write_file(self, filename):
        """ Save the accumulated weights to a darknet weightfile """
        log.debug(f'Writing weight file: version {self.header[0]}.{self.header[1]}.{self.header[2]}')
        with open(filename, 'wb') as fp:
            self.header.tofile(fp)
            self.seen.tofile(fp)
            for np_arr in self.weights:
                np_arr.tofile(fp)
        log.info(f'Weight file saved as {filename}')

    def save_layer(self, layer):
        """ save weights for a layer """
        if type(layer) == nn.Conv2d:
            self._save_conv(layer)
        elif type(layer) == Conv2dBatchReLU:
            self._save_convbatch(layer)
        elif type(layer) == nn.Linear:
            self._save_fc(layer)
        else:
            raise NotImplementedError(f'The layer you are trying to save is not supported [{type(layer)}]')

    def _save_conv(self, model):
        self.weights.append(model.bias.cpu().data.numpy())
        self.weights.append(model.weight.cpu().data.numpy())

    def _save_convbatch(self, model):
        self.weights.append(model.layers[1].bias.cpu().data.numpy())
        self.weights.append(model.layers[1].weight.cpu().data.numpy())
        self.weights.append(model.layers[1].running_mean.cpu().numpy())
        self.weights.append(model.layers[1].running_var.cpu().numpy())
        self.weights.append(model.layers[0].weight.cpu().data.numpy())

    def _save_fc(self, model):
        self.weights.append(model.bias.cpu().data.numpy())
        self.weights.append(model.weight.cpu().data.numpy())
