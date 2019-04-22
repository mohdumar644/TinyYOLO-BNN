Data
====
.. automodule:: lightnet.data

Preprocessing
-------------
These classes perform data augmentation and conversion on your input.
They work just like the :mod:`torchvision transforms <pytorch:torchvision>`. |br|
First you create an object and then you call the object with the image or annotation object as parameter.
You can also call the ``apply()`` method on the classes to run the transformation once.

.. autoclass:: lightnet.data.transform.Crop
.. autoclass:: lightnet.data.transform.Letterbox
.. autoclass:: lightnet.data.transform.RandomFlip
.. autoclass:: lightnet.data.transform.RandomHSV
.. autoclass:: lightnet.data.transform.RandomJitter
.. autoclass:: lightnet.data.transform.RandomRotate
.. autoclass:: lightnet.data.transform.BramboxToTensor

Postprocessing
--------------
These classes parse the output of your networks to understandable data structures.
They work just like the :mod:`torchvision transforms <pytorch:torchvision>`. |br|
First you create an object and then you call the object with the network output as parameter.
You can also call the ``apply()`` method on the classes to run the transformation once.

.. autoclass:: lightnet.data.transform.GetBoundingBoxes
.. autoclass:: lightnet.data.transform.NonMaxSupression
.. autoclass:: lightnet.data.transform.TensorToBrambox
.. autoclass:: lightnet.data.transform.ReverseLetterbox

Data loading
------------
.. autoclass:: lightnet.data.Dataset
   :members: input_dim, resize_getitem
.. autoclass:: lightnet.data.DataLoader
   :members:
.. autofunction:: lightnet.data.list_collate

Util
----
Some random classes and functions that are used in the data subpackage.

.. autoclass:: lightnet.data.transform.Compose
.. autoclass:: lightnet.data.transform.util.BaseTransform
   :members:
.. autoclass:: lightnet.data.transform.util.BaseMultiTransform
   :members:


.. include:: ../links.rst
