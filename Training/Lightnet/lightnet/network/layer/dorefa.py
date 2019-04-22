import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function

import numpy as np


def Binarize(tensor):
        E = tensor.abs().mean()        
        return tensor.sign() * E

def BinarizeSign(tensor):       
        return tensor.sign()  


def BinarizeMean(tensor):
        w = tensor.abs().mean(1).mean(1).mean(1)
        s = tensor.sign().clone()
        s = s * w[:,None,None,None]
        return s

  
def GetMeanXnor(tensor):
        w = tensor.abs().mean(1).mean(1).mean(1)
        return w
   

class Clamper(nn.Module):
    def __init__(self,minval,maxval):
        super(Clamper, self).__init__()
        self.minval = minval
        self.maxval = maxval
        
    def forward(self, x):
        return x.clamp_(self.minval, self.maxval)

 



class Quantizer(nn.Module):
    def __init__(self, k):
        super(Quantizer, self).__init__()
        self.numbits = k

    def forward(self, input): 
        # x = input.clamp_(0,1)
        return Quantize.apply(input, self.numbits)
  



class Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k):
        ctx.save_for_backward(input)
        n = float(2 ** k - 1) 
        input = input*n
        return input.round() / n 

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


def FF(x,k):
    t = 2**k
    t = float(t)
    x = torch.floor( torch.mul(x,t) )
    x = torch.div(x,t)
    x[x==1.] = 0.9921875
    return x


class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)


    def forward(self, input):
 
        if input.size(1) == 3:
            input.data = FF(input.data,7)

        if not hasattr(self.weight,'org'):      ## first time only
            self.weight.org=self.weight.data.clone()

        self.weight.data=BinarizeMean(self.weight.org)

        #wm = GetMeanXnor(self.weight.org)
        # self.weight.data=BinarizeSign(self.weight.org)        

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)
        
        # for idx,w in enumerate(wm):
        #     out[:,idx,:,:]  *= w

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out



class BinarizeConv2dLast(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2dLast, self).__init__(*kargs, **kwargs)


    def forward(self, input):

        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()            

        self.weight.data=BinarizeMean(self.weight.org)        

        # wm = GetMeanXnor(self.weight.org)
        # self.weight.data=BinarizeSign(self.weight.org)        

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        # for idx,w in enumerate(wm):
        #     out[:,idx,:,:]  *= w

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

