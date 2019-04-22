#
#   Base engine class
#   Copyright EAVISE
#

import sys
import logging
import signal
from statistics import mean
from abc import ABC, abstractmethod
import torch

import lightnet as ln

__all__ = ['Engine']
log = logging.getLogger(__name__)


class Engine(ABC):
    """ This class removes the boilerplate code needed for writing your training cycle. |br|
    Here is the code that runs when the engine is called:

    .. literalinclude:: /../lightnet/engine/_engine.py
       :language: python
       :pyobject: Engine.__call__
       :dedent: 4

    Args:
        params (lightnet.engine.HyperParameters): Serializable hyperparameters for the engine to work with
        dataloader (torch.utils.data.DataLoader, optional): Dataloader for the training data
        **kwargs (dict, optional): Keywords arguments that will be set as attributes of the engine

    Attributes:
        self.sigint: Boolean value indicating whether a SIGINT (CTRL+C) was send; Default **False**
        self.*: All values of the :class:`~lightnet.engine.HyperParameters` can be accessed in this class as well

    Note:
        This engine allows to define hook functions to run at certain points in the training *(epoch_start, epoch_end, batch_start, batch_end)*.
        The functions can be defined as class methods of your engine without any extra arguments or as separate functions that take the engine as a single argument.

        There are different functions to register a hook and they can be used as decorator functions or called straight away in code:

        >>> class TrainingEngine(ln.engine.Engine):
        ...     def start(self):
        ...         pass
        ...
        ...     @ln.engine.Engine.epoch_end
        ...     def backup(self):
        ...         pass    # This method will be executed at the end of every epoch
        ...
        ...     @ln.engine.Engine.batch_start(100)
        ...     def update_hyperparams(self):
        ...         pass    # This method will be executed at the start of every 100th batch
        ...
        >>> # Create TrainingEngine object and run it

        >>> def backup(engine):
        ...     pass    # This function will be executed at the end of every Xth batch defined by a backup_rate variable at runtime
        ...
        >>> @ln.engine.Engine.epoch_start
        ... def select_data_subset(engine):
        ...     pass    # This function will be executed at the start of every epoch
        ...
        >>> class TrainingEngine(ln.engine.Engine):
        ...     def start(self):
        ...         if hasattr(self, 'backup_rate') and self.backup_rate is not None:
        ...             self.batch_start(self.backup_rate)(backup)
        ...
        >>> # Create TrainingEngine object and run it

    """
    _init_done = False
    _epoch_start = {}
    _epoch_end = {}
    _batch_start = {}
    _batch_end = {}

    def __init__(self, params, dataloader, **kwargs):
        self.params = params
        if dataloader is not None:
            self.dataloader = dataloader
        else:
            log.warn('No dataloader given, make sure to have a self.dataloader property for this engine to work with.')

        # Sigint handling
        self.sigint = False
        signal.signal(signal.SIGINT, self.__sigint_handler)

        # Logging
        self.__log = ln.logger

        # Set attributes
        for key in kwargs:
            if not hasattr(self, key):
                setattr(self, key, kwargs[key])
            else:
                log.warn(f'{key} attribute already exists on engine.')

        self._init_done = True

    def __call__(self):
        """ Start the training cycle. """
        self.start()

        log.info('Start training')
        self.network.train()

        while True:
            # Epoch Start
            self.epoch += 1
            self._run_hooks(self.epoch, self._epoch_start)

            loader = self.dataloader
            for idx, data in enumerate(loader):
                # Batch Start
                if (idx + 1) % self.batch_subdivisions == 0:
                    self.batch += 1
                    self._run_hooks(self.batch, self._batch_start)

                # Forward and backward on (mini-)batches
                self.process_batch(data)
                if (idx + 1) % self.batch_subdivisions != 0:
                    continue

                # Optimizer step
                self.train_batch()

                # Batch End
                self._run_hooks(self.batch, self._batch_end)

                # Check if we need to stop training
                if self.quit() or self.sigint:
                    self.epoch -= 1     # Did not finish this epoch
                    log.info('Reached quitting criteria')
                    return

                # Not enough mini-batches left to have an entire batch
                if (len(loader) - idx) <= self.batch_subdivisions:
                    break

            # Epoch End
            self._run_hooks(self.epoch, self._epoch_end)

            # Check if we need to stop training
            if self.quit() or self.sigint:
                log.info('Reached quitting criteria')
                return

    def __getattr__(self, name):
        if hasattr(self.params, name):
            return getattr(self.params, name)
        else:
            raise AttributeError(f'{name} attribute does not exist')

    def __setattr__(self, name, value):
        if self._init_done and name not in dir(self) and hasattr(self.params, name):
            setattr(self.params, name, value)
        else:
            super().__setattr__(name, value)

    def __sigint_handler(self, signal, frame):
        if not self.sigint:
            log.debug('SIGINT caught. Waiting for gracefull exit')
            self.sigint = True

    def log(self, msg):
        """ Log messages about training and testing.
        This function will automatically prepend the messages with **TRAIN** or **TEST**.

        Args:
            msg (str): message to be printed
        """
        if self.network.training:
            self.__log.train(msg)
        else:
            self.__log.test(msg)

    def _run_hooks(self, value, hooks):
        """ Internal method that will execute registered hooks. """
        keys = list(hooks.keys())
        for k in keys:
            if value % k == 0:
                for fn in hooks[k]:
                    if hasattr(fn, '__self__'):
                        fn()
                    else:
                        fn(self)

    @classmethod
    def epoch_start(cls, interval=1):
        """ Register a hook to run at the start of an epoch.

        Args:
            interval (int, optional): Number dictating how often to run the hook; Default **1**
        """
        def decorator(fn):
            if interval in cls._epoch_start:
                cls._epoch_start[interval].append(fn)
            else:
                cls._epoch_start[interval] = [fn]
            return fn

        return decorator

    @classmethod
    def epoch_end(cls, interval=1):
        """ Register a hook to run at the end of an epoch.

        Args:
            interval (int, optional): Number dictating how often to run the hook; Default **1**
        """
        def decorator(fn):
            if interval in cls._epoch_end:
                cls._epoch_end[interval].append(fn)
            else:
                cls._epoch_end[interval] = [fn]
            return fn

        return decorator

    @classmethod
    def batch_start(cls, interval=1):
        """ Register a hook to run at the start of a batch.

        Args:
            interval (int, optional): Number dictating how often to run the hook; Default **1**
        """
        def decorator(fn):
            if interval in cls._batch_start:
                cls._batch_start[interval].append(fn)
            else:
                cls._batch_start[interval] = [fn]
            return fn

        return decorator

    @classmethod
    def batch_end(cls, interval=1):
        """ Register a hook to run at the end of a batch.

        Args:
            interval (int, optional): Number dictating how often to run the hook; Default **1**
        """
        def decorator(fn):
            if interval in cls._batch_end:
                cls._batch_end[interval].append(fn)
            else:
                cls._batch_end[interval] = [fn]
            return fn

        return decorator

    def start(self):
        """ First function that gets called when starting the engine. |br|
        Any required setup code can come in here.
        """
        pass

    @abstractmethod
    def process_batch(self, data):
        """ This function should contain the code to process the forward and backward pass of one (mini-)batch. """
        pass

    @abstractmethod
    def train_batch(self):
        """ This function should contain the code to update the weights of the network. |br|
        Statistical computations, performing backups at regular intervals, etc. also happen here.
        """
        pass

    def quit(self):
        """ This function gets called after every training epoch and decides if the training cycle continues.

        Return:
            Boolean: Whether are not to stop the training cycle

        Note:
            This function gets called before checking the ``self.sigint`` attribute.
            This means you can also check this attribute in this function. |br|
            If it evaluates to **True**, you know the program will exit after this function and you can thus
            perform the necessary actions (eg. save final weights).
        """
        return False
