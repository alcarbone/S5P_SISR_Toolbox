# -*- coding: utf-8 -*-
"""
Copyright (c) 
All rights reserved. This work should only be used for nonprofit purposes.

@author: 
    Alessia Carbone (alcarbone@unisa.it)
"""

"""
 Description:
     Very Deep Super-Resolution (VDSR) was proposed as a deeper alternative
     to SRCNN. 
     
     The VDSR requires pre-upsampling (the image is up-sampled before being fed to
     the network by using bicubic interpolation). 
     It uses ReLU as the activation function.
     
     It consists of the following operations:
         1) 1 Input layer -> 64 convolutions with a kernel of size channelsx3x3.
         2) 18 Hidden layers -> 64 convolutions with a kernel of size 64x3x3.
         3) 1 Output layer -> 1 convolution with a kernel of size 64x3x3.
           
 References:
     J. Kim, J. K. Lee and K. M. Lee, "Accurate Image Super-Resolution Using Very Deep Convolutional Networks," 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Las Vegas, NV, USA, 2016, pp. 1646-1654, doi: 10.1109/CVPR.2016.182    
"""

from math import sqrt

import torch
from torch import nn


class ConvReLU(nn.Module):
    def __init__(self, channels: int) -> None:
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False)
        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.relu(out)

        return out


class VDSR(nn.Module):
    def __init__(self) -> None:
        super(VDSR, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3), (1, 1), (1, 1), bias=False),
            nn.ReLU(True),
        ) # input layer

        trunk = []
        for _ in range(18): # number of hidden layers
            trunk.append(ConvReLU(64)) # hidden layer
        self.trunk = nn.Sequential(*trunk)

        self.conv2 = nn.Conv2d(64, 1, (3, 3), (1, 1), (1, 1), bias=False) # output layer

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.trunk(out)
        out = self.conv2(out)

        out = torch.add(out, identity)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(0.0, sqrt(2 / (module.kernel_size[0] * module.kernel_size[1] * module.out_channels)))