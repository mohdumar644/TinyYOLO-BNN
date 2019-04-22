#
#   Visualisation with visdom
#   Copyright EAVISE
#

import logging
import numpy as np
import brambox.boxes as bbb


__all__ = ['LinePlotter']
log = logging.getLogger(__name__)


class LinePlotter:
    """ Wrapper to easily plot curves and lines.

    Args:
        visdom (object): Visdom plotting object
        window (str, optional): Name of the window to plot lines; Default **None**
        env (str, optional): Name of the environment to plot into; Default **main**
        name (str, optional): Name of the trace to draw by default; Default **None**
        opts (dict, optional): Dictionary with default options; Default **{}**

    Note:
        If the visdom argument is None, this plotter will do nothing.
        This can be used to disable using the visdom plotter, without having to check for it in the application code.
    """
    def __init__(self, visdom, window=None, env=None, name=None, opts={}):
        self.vis = visdom
        self.win = window
        self.env = env
        self.name = name
        self.opts = opts

        self.traces = []
        if name is not None:
            self.traces.append(name)

        if self.vis is None:
            return
        if not self.vis.check_connection():
            log.error('No connection with visdom server')
            self.vis = None

    def __call__(self, y, x=None, opts={}, name=None, update='append'):
        """ Add point(s) to a trace or draw a new trace in the window.

        Args:
            y (numpy or torch array): Y-value(s) to plot
            x (numpy or torch array, optional): X-value(s) to plot the Y-value(s) at; Default **None**
            opts (dict, optional): Extra options to pass for this call; Default **{}**
            name (str, optional): Name of the trace to change; Default **Use init name**
            update (str, optional): What to do with new data; Default **append**

        Note:
            If opts is set to ``None``, no options will be passed (not even the default ones).
            This is sometimes necessary, like when you want to remove a trace.
        """
        if self.vis is None:
            return

        if name is None:
            name = self.name
        if name is not None and name not in self.traces:
            self.traces.append(name)
        if opts is not None:
            opts = dict(self.opts, **opts)

        if not self.vis.win_exists(self.win, self.env):
            if 'legend' not in opts:
                opts['legend'] = [name]
            self.vis.line(y, x, self.win, self.env, opts, name=name)
            log.debug(f'Created new visdom window [{self.win}]')
        else:
            self.vis.line(y, x, self.win, self.env, opts, update, name)
            log.debug(f'Updated visdom window [{self.win}]')

    def clear(self, name=None):
        """ Clear the traces that were used with this lineplotter.

        Args:
            name (str, optional): Name of the trace to clear; Default **all**
        """
        if self.vis is None:
            return

        if name is not None:
            self.vis.line(None, win=self.win, env=self.env, name=name, update='remove')
        else:
            for name in self.traces:
                self.vis.line(None, win=self.win, env=self.env, name=name, update='remove')

    def close(self):
        """ Close the visdom window. """
        if self.vis is None:
            return

        if self.vis.win_exists(self.win, self.env):
            self.vis.close(self.win, self.env)
