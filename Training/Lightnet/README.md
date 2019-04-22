LIGHTNET
========
<img src="docs/.static/lightnet.png" alt="Logo" width="250" height="250">  

Building blocks to recreate Darknet networks in Pytorch  
[![Version][version-badge]][documentation-url]
[![PyTorch][pytorch-badge]][pytorch-url]
[![Pipeline][pipeline-badge]][pipeline-badge]



## Why another framework
[pytorch-yolo2](https://github.com/marvis/pytorch-yolo2) is working perfectly fine, but does not easily allow a user to modify an existing network.
This is why I decided to create a library, that gives the user all the necessary building blocks, to recreate any darknet network.  
This library has everything you need to control your network, weight loading & saving, datasets, dataloaders and data augmentation.

## Installing
First install [PyTorch and Torchvision](http://pytorch.org/).  
Then clone this repository and run one of the following commands:
```bash
# If you just want to use Lightnet
pip install -r requirements.txt

# If you want to develop Lightnet
pip install -r develop.txt
```
> This project is python 3.6 and higher so on some systems you might want to use 'pip3.6' instead of 'pip'

## How to use
[Click Here](https://eavise.gitlab.io/lightnet) for the API documentation and guides on how to use this library.  
The _examples_ folder contains code snippets to train and test networks with lightnet. For examples on how to implement your own networks, you can take a look at the files in _lightnet/models_.
>If you are using a different version than the latest,
>you can generate the documentation yourself by running `make clean html` in the _docs_ folder.
>This does require some dependencies, like Sphinx.
>The easiest way to install them is by using the __-r develop.txt__ option when installing lightnet.

## Cite
If you use Lightnet in your research, please cite it.
```
@misc{lightnet18,
  author = {Tanguy Ophoff},
  title = {Lightnet: Building Blocks to Recreate Darknet Networks in Pytorch},
  howpublished = {\url{https://gitlab.com/EAVISE/lightnet}},
  year = {2018}
}
```

## Main Contributors
Here is a list of people that made noteworthy contributions and helped to get this project where it stands today!
  - [Tanguy Ophoff](https://gitlab.com/0phoff)
  - [John Crall](https://gitlab.com/Erotemic)


[version-badge]: https://img.shields.io/badge/version-0.4.0-blue.svg
[pytorch-badge]: https://img.shields.io/badge/PyTorch-0.4.1-F05732.svg
[pytorch-url]: https://pytorch.org
[pipeline-badge]: https://gitlab.com/EAVISE/lightnet/badges/master/pipeline.svg
[documentation-url]: https://eavise.gitlab.io/lightnet
