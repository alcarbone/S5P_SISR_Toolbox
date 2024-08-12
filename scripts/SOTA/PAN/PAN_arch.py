# -*- coding: utf-8 -*-
"""
Copyright (c) 
All rights reserved. This work should only be used for nonprofit purposes.

@author: 
    Hengyuan Zhao, Xiangtao Kong, Jingwen He, Yu Qiao, Chao Dong
"""

"""
 Description:
     Pixel Attention Network (PAN) was proposed as a more efficient alternative to 
     SRResNet. It uses a combination of spatial and channel attention called pixel
     attention but it uses very few parameters.
     
     The PAN does not require pre-upsampling (the image is post-upsampled by
     bilinear interpolation).  
     
     It consists of the following operations:
         1) Feature extraction layer -> SC-PA (it uses LeakyReLU as the activation function).
         2) Non-linear mapping module -> SC-PAs and PA blocks (it uses sigmoid as the activation function).
         3) Reconstruction module -> convs (it uses LeakyReLU as the activation function)
           
 References:
     H. Zhao, X. Kong, J. He, Y. Qiao, and C. Dong, ‘Efficient Image Super-Resolution Using Pixel Attention’. arXiv, Oct. 02, 2020. doi: 10.48550/arXiv.2010.01073.
"""

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

import scripts.SOTA.PAN.arch_util as arch_util

# Pixel Attention block
class PA(nn.Module):
    def __init__(self, nf):
        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)

        return out
    
class PAConv(nn.Module):
    def __init__(self, nf, k_size=3):
        super(PAConv, self).__init__()
        self.k2 = nn.Conv2d(nf, nf, 1) # nf convs of size nfx1x1
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # nf convs of size nfx3x3 
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # nf convs of size nfx3x3

    def forward(self, x):
        y = self.k2(x)
        y = self.sigmoid(y)
        out = torch.mul(self.k3(x), y)
        out = self.k4(out)
        return out
   
#SC-PA block    
class SCPA(nn.Module):
    def __init__(self, nf, reduction=2, stride=1, dilation=1):
        super(SCPA, self).__init__()
        group_width = nf // reduction
        
        self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, bias=False) # group-width convs of size nfx1x1
        self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False) # group-width convs of size nfx1x1
        
        self.k1 = nn.Sequential(
                    nn.Conv2d(
                        group_width, group_width, kernel_size=3, stride=stride,
                        padding=dilation, dilation=dilation,
                        bias=False) # group-width convs of size group_widthx3x3 
                    ) # sequential block 
        
        self.PAConv = PAConv(group_width) # PAConv block
        
        self.conv3 = nn.Conv2d(
            group_width * reduction, nf, kernel_size=1, bias=False) # nf convs of size (group-width*reduction)x1x1
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        residual = x
        out_a= self.conv1_a(x)
        out_b = self.conv1_b(x)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)
        out_a = self.k1(out_a)
        out_b = self.PAConv(out_b)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)
        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out += residual
        return out
    
    
class PAN(nn.Module):
    
    def __init__(self, in_nc, out_nc, nf, unf, nb, scale=4):
        super(PAN, self).__init__()
        
        # Feature extraction
        SCPA_block_f = functools.partial(SCPA, nf=nf, reduction=2) # SC-PA block 
        self.scale = scale
        
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True) # nf convs of size in_ncx3x3
        
        # Non-linear mapping
        self.SCPA_trunk = arch_util.make_layer(SCPA_block_f, nb) # SC-PA block 
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True) # nf convs of size nfx3x3
        
        self.upconv1 = nn.Conv2d(nf, unf, 3, 1, 1, bias=True) # unf convs of size nfx3x3
        self.att1 = PA(unf) # PA block
        self.HRconv1 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True) # unf convs of size nfx3x3
        
        if self.scale == 4: # if scale == 4, add PA block
            self.upconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True) # unf convs of size nfx3x3
            self.att2 = PA(unf) # PA block
            self.HRconv2 = nn.Conv2d(unf, unf, 3, 1, 1, bias=True) # unf convs of size nfx3x3
         
        # Reconstruction
        self.conv_last = nn.Conv2d(unf, out_nc, 3, 1, 1, bias=True) # out_nh convs of size unfx3x3
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x): 
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.SCPA_trunk(fea))
        fea = fea + trunk
        if self.scale == 2 or self.scale == 3:
            fea = self.upconv1(F.interpolate(fea, scale_factor=self.scale, mode='nearest'))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
        elif self.scale == 4:
            fea = self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.att1(fea))
            fea = self.lrelu(self.HRconv1(fea))
            fea = self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest'))
            fea = self.lrelu(self.att2(fea))
            fea = self.lrelu(self.HRconv2(fea))
        out = self.conv_last(fea)
        ILR = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        out = out + ILR
        return out
 