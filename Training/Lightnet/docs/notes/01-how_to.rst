Using Lightnet
==============
Lightnet was originally built to recreate darknet_ type networks in pytorch_.
It has however evolved towards a library to aid **0phoff** in his PhD research, and as such contains many more building blocks than only the ones needed for darknet.

The library is build in a hierarchical way.
You could choose to only use a small subset of the building blocks from a few subpackages and build up your own network architectures or use the entire library and just use the network models that are provided. |br|
The different subpackages of lightnet are:

- lightnet.network:
   This subpackage contains everything related to building networks.
   This means layers, loss functions, weight-loading, etc.
- ligthnet.data:
   In here you will find everything related to data-processing.
   This includes data augmentation, post-processing of network output, etc.
- lightnet.engine:
   This subpackage contains blocks related to the automation of training and testing.
   It has an engine that reduces the boilerplate code needed for training,
   functions for visualisation with visdom_, etc.
- lightnet.models:
   This subpackage has some network implementations that I felt like sharing.
   This subpackage can be handy to look at and learn how to use this library.

A user of lightnet could choose to only use the provided layers and loss functions of this package, and build his own modules, training algorithm, etc. with pytorch.
Another way of working is to use the :class:`network module <lightnet.network.Darknet>` provided by this package, and just build your own training and testing scripts.
Finally, you can also decide to use everything this package has to offer. This might mean you need a bit more time to get acquainted with everything, but does mean you will write less code at the end of the day! |br|

*Happy Coding!* ♥ |br|
**~0phoff**


.. include:: ../links.rst
