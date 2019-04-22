import lightnet as ln
import torch

__all__ = ['params']


params = ln.engine.HyperParameters( 
    # Network
    class_label_map = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
    input_dimension = (416, 416),
    batch_size = 64,
    mini_batch_size = 8,
    max_batches = 80200,

    # Dataset
    _train_set = 'data/train.pkl',
    _valid_set = None,
    _test_set = 'data/test.pkl',
    _filter_anno = 'ignore',

    # Data Augmentation
    jitter = .3,
    flip = .5,
    hue = .1,
    saturation = 1.5,
    value = 1.5,
)

# Network
def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')

params.network = ln.models.TinyYolo(
    len(params.class_label_map),
    conf_thresh = .001,
    nms_thresh = .5,
)
params.network.postprocess.append(ln.data.transform.TensorToBrambox(params.input_dimension, params.class_label_map))
params.network.apply(init_weights)

# Optimizers
params.add_optimizer(torch.optim.SGD(
    params.network.parameters(),
    lr = .001 / params.batch_size,
    momentum = .9,
    weight_decay = .0005 * params.batch_size,
    dampening = 0,
))

# Schedulers
burn_in = torch.optim.lr_scheduler.LambdaLR(
    params.optimizers[0],
    lambda b: (b / 1000) ** 4,
)
step = torch.optim.lr_scheduler.MultiStepLR(
    params.optimizers[0],
    milestones = [40000, 60000],
    gamma = .1,
)
params.add_scheduler(ln.engine.SchedulerCompositor(
#   batch   scheduler
    (0,     burn_in),
    (1000,  step),
))
