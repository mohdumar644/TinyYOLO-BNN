Models
======
.. automodule:: lightnet.models

Networks
--------
Darknet
~~~~~~~
These are network from the original darknet_ project, reimplemented in PyTorch.

   J. Redmon. Darknet: Open Source Neural Networks in C.
   Online *http://pjreddie.com/darknet/*, 2013-2016.

.. autoclass:: lightnet.models.Darknet19
.. autoclass:: lightnet.models.Yolo
.. autoclass:: lightnet.models.TinyYolo

.. .. autoclass:: lightnet.models.MobileNetYolo

Fusion
~~~~~~
These are the fusion networks, created whilst researching rgb+depth fusion.

   T. Ophoff and K. Van Beeck and T. Goedemé. Improving Real-Time Pedestrian Detectors with RGB+Depth Fusion.
   In *AVSS Workshop - MSS*, November 2018.

.. autoclass:: lightnet.models.YoloMidFusion
.. autoclass:: lightnet.models.YoloLateFusion

Data
----
.. autoclass:: lightnet.models.BramboxDataset
   :members: __getitem__
.. autoclass:: lightnet.models.DarknetDataset
   :members:


.. include:: ../links.rst
