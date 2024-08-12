# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )
"""

"""
 Description: 
           Spectral Angle Mapper (SAM).
 
 Interface:
           [SAM_index,SAM_map] = SAM(I1,I2)

 Inputs:
           I1:         First multispectral image;
           I2:         Second multispectral image.
 
 Outputs:
           SAM_index:  SAM index;
           SAM_map:    Image of SAM values.
 
 References:
         G. Vivone, M. Dalla Mura, A. Garzelli, and F. Pacifici, "A Benchmarking Protocol for Pansharpening: Dataset, Pre-processing, and Quality Assessment", IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 6102-6118, 2021.
         G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting Pansharpening With Classical and Emerging Pansharpening Methods", IEEE Geoscience and Remote Sensing Magazine, vol. 9, no. 1, pp. 53 - 81, March 2021.           
"""

import numpy as np
import math

def  SAM(I1,I2):
    
    M = I2.shape[0]
    N = I2.shape[1]
    
    
    if np.size(I2.shape) == 2 :
        
            SAM_index = 0
            SAM_map = np.zeros(I2.shape)
    
    else:
        
        prod_scal = np.zeros((M,N))
        norm_orig = np.zeros((M,N))
        norm_fusa = np.zeros((M,N))
        for iM in range(M):
            for iN in range(N):            
                h1 = I1[iM,iN,:]
                h2 = I2[iM,iN,:]
                prod_scal[iM,iN] = h1.flatten() @ h2.flatten()
                norm_orig[iM,iN] = h1.flatten() @ h1.flatten()
                norm_fusa[iM,iN] = h2.flatten() @ h2.flatten()
        
        
        prod_norm = np.sqrt(norm_orig * norm_fusa)
        prod_map = prod_norm
        prod_map[prod_map == 0] = 2 * 10**(-16)
        
        prod = prod_scal/prod_map
        prod[prod > 1] = 0.9999999
        
        SAM_map = np.arccos(prod)
        
        prod_scal = np.reshape(prod_scal, (M*N,1))
        prod_norm = np.reshape(prod_norm, (M*N,1))
        
        z = np.nonzero(prod_norm == 0)
        
        prod_scal[z]=[]
        prod_norm[z]=[]
        
        prod = prod_scal/prod_norm
        prod[prod > 1] = 0.9999999
        
        angolo = np.sum(np.arccos(prod))/(prod_norm.shape[0])
        
        SAM_index = angolo*180/math.pi
    
    
    
    return SAM_index, SAM_map