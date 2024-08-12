# -*- coding: utf-8 -*-
"""
Copyright (c) 
All rights reserved. This work should only be used for nonprofit purposes.

@author: 
    Alessia Carbone (alcarbone@unisa.it)
"""

"""
 Description:
           The process of super-resolution can be considered as the interpolation
           of the original LR image, once chosen an interpolator.
           
           Interface: SR_int = interp(LR,ratio,interp_type,tc=1)
          
           Inputs: LR: "low-resolution" image;
                   ratio: scaling ratio;
                   interp_type: interpolation type;
                   tc = sampling time (default is 1).
               
           Outputs: SR_int: SR output image.
           
 References:
       B. Aiazzi, S. Baronti, M. Selva, and L. Alparone, ‘Bi-cubic interpolation for shift-free pan-sharpening’, ISPRS Journal of Photogrammetry and Remote Sensing, vol. 86, pp. 65–76, Dec. 2013, doi: 10.1016/j.isprsjprs.2013.09.007.

"""

import sys
import numpy as np
from scripts.interp.kernel import kernel
from scripts.interp.interp23 import interp23
from scripts.interp.interp23_1 import interp23_1

def interp(LR, ratio, interp_type, tc=1):
    
    dim = LR.shape
    
    if (len(LR.shape) == 2):
        LR1 = np.zeros([LR.shape[0],LR.shape[1],1])
        LR1[:,:,0] = LR
        LR = LR1
        
        new_dim = [dim[0]*ratio,dim[1]*ratio,1]
    else:
        new_dim = [dim[0]*ratio,dim[1]*ratio,LR.shape[2]]
    
    SR_int = np.zeros(new_dim)
    
    if(interp_type == '23tap'): 
        
        if(ratio % 2 == 1):
            SR_int = interp23_1(LR,ratio)
        else:
            if(len(LR.shape) == 3 and LR.shape[2] == 1):
                LR1 = np.zeros((LR.shape[0],LR.shape[1]))
                LR1 = LR[:,:,0]
    
                SR_int = interp23(LR1,ratio)
                
                SR_int1 = np.zeros((SR_int.shape[0],SR_int.shape[1],1))
                SR_int1[:,:,0] = SR_int
                SR_int = SR_int1
            else: 
                SR_int = interp23(LR,ratio)
         
    else:
        
        ker = kernel(tc,ratio,interp_type)
        
        SR_int[ratio//2::ratio,ratio//2::ratio, :] = LR
        
        for ch in range(new_dim[2]):
            for i in range(new_dim[0]):
                SR_int[i,:,ch] = np.convolve(SR_int[i,:,ch],ker,'same')
            for j in range(new_dim[1]):
                SR_int[:,j,ch] = np.convolve(SR_int[:,j,ch],ker,'same')
                
    SR_int = SR_int.astype('float64')
        
    return SR_int
