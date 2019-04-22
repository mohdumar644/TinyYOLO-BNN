.. Lightnet documentation master file, created by
   sphinx-quickstart on Fri Dec  8 09:41:38 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:gitlab_url: https://gitlab.com/EAVISE/lightnet

Lightnet documentation
======================
Lightnet is a library for pytorch_, that makes it easier to create CNN's.
It was mainly created to implement darknet_ networks in python.

Credits
-------
Credits where credits are due. I get a lot *-if not most-* of my ideas from other papers and repositories. |br|
Without the work of the following people, this library would have never become a reality.

- `pjreddie et al. <darknet_>`_ for their work on the original YOLO networks
- `The PyTorch team <pytorch_>`_ for creating a clear and powerfull library for deep learning.
- `marvis <pytorch-yolo2_>`_ for his implementation of YOLO in pytorch. I took a lot of parts from this repository, and tweaked them to my needs.
- `longcw <yolo2-pytorch_>`_ for his implementation of YOLO in pytorch. I used his code to get a correct region loss.
- `Andrew G. Howard et al. <mobilenets_>`_ for their work on efficient MobileNets.

Cite
----
If you use Lightnet in your research, please cite it.

.. code:: bibtex

   @misc{lightnet18,
     author = {Tanguy Ophoff},
     title = {Lightnet: Building Blocks to Recreate Darknet Networks in Pytorch},
     howpublished = {\url{https://gitlab.com/EAVISE/lightnet}},
     year = {2018}
   }

Table of Contents
=================
.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Notes

   notes/*

.. toctree::
   :maxdepth: 2
   :caption: API

   lightnet.data <api/data>
   lightnet.network <api/network>
   lightnet.engine <api/engine>
   lightnet.models <api/models>
   lightnet.log <api/log>


Indices and tables
==================
* :ref:`genindex`
* :ref:`search`

.. include:: links.rst
