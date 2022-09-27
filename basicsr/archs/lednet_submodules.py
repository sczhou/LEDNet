#!/usr/bin/python
# -*- coding: utf-8 -*-
#
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn import functional as F

class KernelConv2D(nn.Module):
    def __init__(self, ksize=5, act=True):
        super(KernelConv2D, self).__init__()
        self.ksize = ksize
        self.act = act

    def forward(self, feat_in, kernel):
        channels = feat_in.size(1)
        N, kernels, H, W = kernel.size()
        pad = (self.ksize - 1) // 2

        feat_in = F.pad(feat_in, (pad, pad, pad, pad), mode="replicate")
        feat_in = feat_in.unfold(2, self.ksize, 1).unfold(3, self.ksize, 1)
        feat_in = feat_in.permute(0, 2, 3, 1, 4, 5).contiguous()
        feat_in = feat_in.reshape(N, H, W, channels, -1)

        kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, channels, -1)
        feat_out = torch.sum(feat_in * kernel, -1)
        feat_out = feat_out.permute(0, 3, 1, 2).contiguous()
        if self.act:
            feat_out = F.leaky_relu(feat_out, negative_slope=0.2, inplace=True)
        return feat_out

def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer


class Downsample(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]    
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])