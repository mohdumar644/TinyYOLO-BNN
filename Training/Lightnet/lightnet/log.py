#
#   Lightnet logger: Logging functionality used within the lightnet package
#   Copyright EAVISE
#

import os
import types
import logging
import copy
from enum import Enum

__all__ = ['logger']


# Formatter
class ColorCode(Enum):
    """ Color Codes """
    RESET = '\033[00m'
    BOLD = '\033[01m'

    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    WHITE = '\033[37m'
    GRAY = '\033[1;30m'


class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, color=True, **kwargs):
        logging.Formatter.__init__(self, msg, **kwargs)
        self.color = color
        self.color_codes = {
            'CRITICAL': ColorCode.RED,
            'ERROR': ColorCode.RED,
            'TRAIN': ColorCode.BLUE,
            'TEST': ColorCode.BLUE,
            'DEPRECATED': ColorCode.YELLOW,
            'WARNING': ColorCode.YELLOW,
            'INFO': ColorCode.WHITE,
            'DEBUG': ColorCode.GRAY,
        }

    def format(self, record):
        record = copy.copy(record)
        levelname = record.levelname
        name = record.name
        if self.color:
            color = self.color_codes[levelname] if levelname in self.color_codes else ''
            record.levelname = f'{ColorCode.BOLD.value}{color.value}{levelname:10}{ColorCode.RESET.value}'
        else:
            record.levelname = f'{levelname:10}'
        return logging.Formatter.format(self, record)

    def setColor(self, value):
        """ Enable or disable colored output for this handler """
        self.color = value


# Filter
class LevelFilter(logging.Filter):
    def __init__(self, levels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.levels = levels

    def filter(self, record):
        if self.levels is None or record.levelname in self.levels:
            return True
        else:
            return False


# Logging levels
def deprecated(self, message, *args, **kwargs):
    if not hasattr(self, 'deprecated_msgs'):
        self.deprecated_msgs = []

    if self.isEnabledFor(35) and message not in self.deprecated_msgs:
        self.deprecated_msgs.append(message)
        self._log(35, message, args, **kwargs)


def test(self, message, *args, **kwargs):
    if self.isEnabledFor(38):
        self._log(38, message, args, **kwargs)


def train(self, message, *args, **kwargs):
    if self.isEnabledFor(39):
        self._log(39, message, args, **kwargs)


logging.addLevelName(35, 'DEPRECATED')
logging.Logger.deprecated = deprecated
logging.addLevelName(38, 'TEST')
logging.Logger.test = test
logging.addLevelName(39, 'TRAIN')
logging.Logger.train = train


# Console Handler
ch = logging.StreamHandler()
ch.setFormatter(ColoredFormatter('{levelname} {message}', style='{'))
if 'LN_LOGLVL' in os.environ:
    ch.setLevel(os.environ['LN_LOGLVL'])
    if os.environ['LN_LOGLVL'] == 'DEBUG':
        ch.setFormatter(ColoredFormatter('{levelname} [{name}] {message}', style='{'))
else:
    ch.setLevel(logging.INFO)


# File Handler
def createFileHandler(self, filename, levels=None, filemode='a'):
    """ Create a file to write log messages of certaing levels """
    fh = logging.FileHandler(filename=filename, mode=filemode)
    fh.setLevel(logging.NOTSET)
    fh.addFilter(LevelFilter(levels))
    fh.setFormatter(logging.Formatter('{levelname} [{name}] {message}', style='{'))
    logger.addHandler(fh)
    return fh


# Logger
logger = logging.getLogger('lightnet')
logger.setLevel(logging.DEBUG)
logger.addHandler(ch)
logger.setConsoleLevel = ch.setLevel
logger.setConsoleColor = ch.formatter.setColor
logger.setLogFile = types.MethodType(createFileHandler, logger)
