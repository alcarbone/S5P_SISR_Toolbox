# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )
"""
"""
 Description: 
           interp23_1 interpolates the image I_Interpolated using a polynomial with 44 coefficients interpolator. 
 
 Interface:
           image = interp23_1(image,ratio)

 Inputs:
           image:          Image to interpolate;
           ratio:          Scale ratio between MS and PAN. 

 Outputs:
           image:          Interpolated image.
 
 References:
         G. Vivone, M. Dalla Mura, A. Garzelli, and F. Pacifici, "A Benchmarking Protocol for Pansharpening: Dataset, Pre-processing, and Quality Assessment", IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 6102-6118, 2021.
         G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting Pansharpening With Classical and Emerging Pansharpening Methods", IEEE Geoscience and Remote Sensing Magazine, vol. 9, no. 1, pp. 53 - 81, March 2021.
"""

import numpy as np
from scipy import ndimage
import matlab.engine


def interp23_1(image, ratio):

    L = 44  
    
    r = image.shape[0]
    c = image.shape[1]
    
    if (np.size(image.shape) == 3):      
        b = image.shape[2]
    else:
        b = 1
    
    eng = matlab.engine.start_matlab()
    BaseCoeff = ratio * np.asarray(eng.fir1(L, 1. / ratio))
    BaseCoeff1 = np.zeros([BaseCoeff.shape[1]])
    BaseCoeff1[:] = BaseCoeff
    BaseCoeff = BaseCoeff1
    
    I1LRU = np.zeros((ratio * r, ratio * c, b))
    if (b==1 and np.size(image.shape) != 3):
        I1LRU[::ratio, ::ratio, 0] = image
    else:
        I1LRU[::ratio, ::ratio, :] = image
    
    for ii in range(b):
        t = I1LRU[:, :, ii]
        
        for j in range(0,I1LRU.shape[0]):
            I1LRU[j, :, ii]=ndimage.correlate(t[j,:],BaseCoeff,mode='wrap')
        for k in range(0,I1LRU.shape[1]):
            I1LRU[:, k, ii]=ndimage.correlate(t[:,k],BaseCoeff,mode='wrap')
        
    I_Interpolated = I1LRU
    
    return I_Interpolated