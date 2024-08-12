# -*- coding: utf-8 -*-
"""
Copyright (c) 
All rights reserved. This work should only be used for nonprofit purposes.

@author: 
    Alessia Carbone (alcarbone@unisa.it)
"""

"""
 Description:
     S5Net network (transposed convolution + 3 convolutional layers)
           
"""

from torch import nn

class S5Net(nn.Module):
    def __init__(self,n1,n2,n3,f1,f2,f3,c,dec_size,ratio,weights_deconv=None,weights_conv1=None,weights_conv2=None,weights_conv3=None,
                 biases_conv1=None,biases_conv2=None,biases_conv3=None):
        super(S5Net, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(c, c, kernel_size=dec_size, stride=ratio, padding = dec_size-1-int(ratio/2)-int((dec_size-1)/2), bias=False)
        self.conv1 = nn.Conv2d(c, n1, kernel_size=f1, padding_mode='replicate', padding = 'same')
        self.conv2 = nn.Conv2d(n1, n2, kernel_size=f2, padding_mode='replicate', padding = 'same')
        self.conv3 = nn.Conv2d(n3, c, kernel_size=f3, padding_mode='replicate', padding = 'same')
        self.relu = nn.ReLU(inplace=True)
        
        if(weights_conv1!=None): 
            self.deconv1.weight.data = weights_deconv
            self.conv1.weight.data = weights_conv1
            self.conv2.weight.data = weights_conv2
            self.conv3.weight.data = weights_conv3
        
        if(biases_conv1!=None):
            self.conv1.bias.data = biases_conv1
            self.conv2.bias.data = biases_conv2
            self.conv3.bias.data = biases_conv3

    def forward(self, x, ratio, device):
        new_dim = [x.shape[2]*ratio,x.shape[3]*ratio]
        x = self.deconv1.forward(x)
        x = x[:,:,0:new_dim[0],0:new_dim[1]]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x