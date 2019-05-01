
# TinyYOLO-BNN

This is an example of a Quantized Tiny YOLO v2 on FPGA using the Xilinx FINN framework, specifically [BNN-PYNQ](https://github.com/Xilinx/BNN-PYNQ).

The board targeted is ZC706 (Zynq 7z045).

You need an understanding of that repo to run this example successfully.


### Training Algorithm

We use W1A4.

We use [DoReFa-Net](https://arxiv.org/abs/1606.06160) like training, except that we use channel-wise mean for the weights as in [XNOR-Net](https://arxiv.org/abs/1603.05279).

Please refer to the [QNN-MO-PYNQ Tiny YOLO example](https://github.com/Xilinx/QNN-MO-PYNQ/blob/master/notebooks/tiny-yolo-image.ipynb) for the topology.

### Training Framework

We use a custom PyTorch implementation of Darknet called [Lightnet](https://gitlab.com/EAVISE/lightnet).

Follow the instructions at https://eavise.gitlab.io/lightnet/notes/02-examples.html#pascal-voc to prepare the PASCAL VOC dataset for use with Lightnet.

Launch training from the `Training/Lightnet/examples/yolo-voc` folder with

```
python3 bin/train.py -c -n cfg/tinyyolo.py  /path/to/pretrained/weights/if/present.pt
```

To train on your custom datasets, please refer to the Lightnet documentation.

To draw boxes on a test image, use the `Training/Lightnet/examples/basic/detect.py`.

For evaluating the mAP score on the test dataset, 
```
python3 bin/test.py -c -n cfg/tinyyolo.py   /path/to/trained/weights.pt -f
```

### Packing Weights

We have included weights having 51.5% mAP score for the PASCAL VOC dataset, at `Training/Weights-Packing/pretrained.pt`.

The `.pt` to `.npz` converter and Finnthesizer are also present in the same directory. Call `gen-weights.py` from there to create the .bin files.

Such files for the pretrained weights are provided at `BNN-PYNQ/bnn/params/tinyyolo/cnvW1A1`.


## Hardware Description





We have modified the original BNN-PYNQ to support DoReFa-Net trained networks instead of their vanilla BNN WnAn like networks.


The library may be hard-coded for W1A4.

The hardware description is found in `./BNN-PYNQ/bnn/src/network/cnvW1A1`.

## Deployment

Copy the bitstream and the parameters to the board. A shared library object for the network must also be compiled as per the new configuration for correct weight loading.

Run the notebook 'Inference/TinyYOLO_LOOP.ipynb'

Please note that we use the shared object and its python interface to only load the accelerator weights. Rest of the inference steps are done using Python `pynq` package.
 
Moreover, we do not use a custom-written post-processor for YOLO, not the one in Darknet. 

## References

As linked above.
