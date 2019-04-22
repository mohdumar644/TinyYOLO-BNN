Examples
========
This page contains explanation with the the scripts located in the `examples folder`_ of the repository.


Basic examples
--------------
The scripts in the `basic` subfolder show how to use this library to train, test and utilize networks.

- detect.py
   This script shows how to initialize and use a network to perform detections on OpenCV images.
   Note that besides OpenCV, Pillow is also supported and is considered a better and faster alternative
   by the author of this package. The OpenCV example is just here for demonstration purposes.
- train.py
   This script shows how to use the :mod:`lightnet.engine` to train a network on some arbitrary data.
   This script uses the :class:`lightnet.models.DarknetDataset` dataset,
   which uses a file with paths to the images and assumes there are annotations for each image with the same name,
   but a ``.txt`` extension.
- test.py
   This script shows how to run a network on an entire testset and
   perform a statistical analysis of the results with brambox_.


Pascal VOC
----------
The scripts in the `yolo-voc` subfolder were build to test the results of lightnet on the Pascal VOC dataset and compare them with darknet.
We perform the same training and testing as explained on the `darknet website`_.

.. rubric:: Get the data

We train YOLO on all of the VOC data from 2007 and 2012.
To get all the data, make a directory to store it all, and execute the following commands:

.. code:: bash

   mkdir data
   cd data
   wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
   wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
   wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
   tar xf VOCtrainval_11-May-2012.tar
   tar xf VOCtrainval_06-Nov-2007.tar
   tar xf VOCtest_06-Nov-2007.tar
   cd ..

There will now be a *VOCdevkit* folder with all the data.

.. rubric:: Generating labels

We need to have the right labels for training and testing the network. |br|
While brambox (and thus lightnet) can work with Pascal VOC annotations,
we still need to group the data in a training and testing set.
Because we are converting this anyway, we take the opportunity to convert the annotations to a pickle format,
which will be faster to parse whilst training/testing. |br|
You can check whether to annotation conversion was succesfull, by running the **bbox_view.py** script from brambox.

.. code:: bash

   # Change the ROOT variable in labels.py to point to the root directory that contains VOCdevkit
   ./bin/labels.py
   bbox_view.py -lx .jpg anno_pickle ROOT/train.pkl ROOT/VOCdevkit
   bbox_view.py -lx .jpg anno_pickle ROOT/test.pkl ROOT/VOCdevkit

.. Note::
   There is no validation set.  
   We perform the same training cycle as darknet, and thus have no testing whilst training the network.
   This means there is no need for a separate testing and validation set,
   but also means we have no way to check how well the network performs whilst it is training.

.. rubric:: Get weights

For training, we use weights that are pretrained on ImageNet. |br|
See :ref:`accuracy` for more information on the difference between darknet and lightnet pretrained weights.

========= ===
Framework URL
========= ===
Darknet   https://pjreddie.com/media/files/darknet19_448.conv.23
--------- ---
Lightnet  https://mega.nz/#!ChsBkSQT!8Jpjzzi_tgPtd6gs079g4ea-XOUIr3LspOqAgk97hUA
========= ===

.. rubric:: Train model

Use the **train.py** script to train the model. You can use *train.py --help* for an explanation of the arguments and flags.

.. code:: bash

   # Adapt the model parameters inside of train.py to suite your needs
   ./bin/train.py -cv -n cfg/yolo.py <path/to/pretrained/weights>

.. rubric:: Test model

Use the **test.py** script to test the model. You can again use *test.py --help* for an explanation of the arguments and flags.

.. code:: bash

   # We use tqdm for a nice loading bar
   pip install tqdm 
   
   # Adapt the model parameters inside of test.py to suite your needs
   ./bin/test.py -c -n cfg/yolo.py backup/final.pt


.. include:: ../links.rst
.. _examples folder: https://gitlab.com/EAVISE/lightnet/tree/master/examples
.. _darknet website: https://pjreddie.com/darknet/yolov2/#train-voc
