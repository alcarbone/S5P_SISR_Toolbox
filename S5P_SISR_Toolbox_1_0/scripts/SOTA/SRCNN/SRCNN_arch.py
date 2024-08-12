# -*- coding: utf-8 -*-
"""
Copyright (c) 
All rights reserved. This work should only be used for nonprofit purposes.

@author: 
    Alessia Carbone (alcarbone@unisa.it)
"""

"""
 Description:
     Super-Resolution Convolutional Neural Network (SRCNN) was proposed as the first
     neural network for Super-Resolution. 
     
     The SRCNN requires pre-upsampling (the image is up-sampled before being fed to
     the network by using bicubic interpolation). 
     It uses ReLU as the activation function.
     
     It consists of the following operations:
         1) 1 Feature extraction layer -> n1 convolutions with a kernel of size cxf1xf1.
         2) 1 Non-linear-mapping layer -> n2 convolutions with a kernel of size n1xf2xf2.
         3) 1 Reconstruction layer -> c convolutions with a kernel of size n3xf3xf3.
           
 References:
     C. Dong, C. C. Loy, K. He, and X. Tang, ‘Image Super-Resolution Using Deep Convolutional Networks’, Jul. 2015, Accessed: Mar. 06, 2023. [Online]. Available: http://arxiv.org/abs/1501.00092
"""

from torch import nn

class SRCNN(nn.Module):
    def __init__(self,n1,n2,n3,f1=9,f2=5,f3=5,c=1,weights_conv1=None,weights_conv2=None,weights_conv3=None,
                 biases_conv1=None,biases_conv2=None,biases_conv3=None):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(c, n1, kernel_size=f1, padding_mode='replicate', padding = 'same') # feature extraction layer
        self.conv2 = nn.Conv2d(n1, n2, kernel_size=f2, padding_mode='replicate', padding = 'same') # non-linear mapping layer
        self.conv3 = nn.Conv2d(n3, c, kernel_size=f3, padding_mode='replicate', padding = 'same') # reconstruction layer
        self.relu = nn.ReLU(inplace=True)
        
        if(weights_conv1 is not None):       
            self.conv1.weight.data = weights_conv1
            self.conv2.weight.data = weights_conv2
            self.conv3.weight.data = weights_conv3
        
        if(biases_conv1 is not None):
            self.conv1.bias.data = biases_conv1
            self.conv2.bias.data = biases_conv2
            self.conv3.bias.data = biases_conv3

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
    
        