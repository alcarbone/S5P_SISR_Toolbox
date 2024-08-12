# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )
"""

"""
 Description: 
          Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS).

 Interface:
           ERGAS_index = ERGAS(I1,I2,ratio)

 Inputs:
           I1:             First multispectral image;
           I2:             Second multispectral image;
           ratio:          Scale ratio between MS and PAN. Pre-condition: Integer value.
 
 Outputs:
           ERGAS_index:    ERGAS index.
 
 References:
         G. Vivone, M. Dalla Mura, A. Garzelli, and F. Pacifici, "A Benchmarking Protocol for Pansharpening: Dataset, Pre-processing, and Quality Assessment", IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 6102-6118, 2021.
         G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting Pansharpening With Classical and Emerging Pansharpening Methods", IEEE Geoscience and Remote Sensing Magazine, vol. 9, no. 1, pp. 53 - 81, March 2021.    
"""

import numpy as np
import math

def ERGAS(I1,I2,ratio):

    I1 = I1.astype('float64')
    I2 = I2.astype('float64')

    Err = I1-I2

    ERGAS_index=0

    if np.size(I2.shape) == 2:
        
       ERGAS_index = ERGAS_index + np.mean(Err[:,:]**2)/(np.mean(I1))**2    
  
       ERGAS_index = (100/ratio) * math.sqrt(ERGAS_index) 
       
    else:
        for iLR in range(I1.shape[2]):
            ERGAS_index = ERGAS_index + np.mean(Err[:,:,iLR]**2, axis=(0, 1))/(np.mean(I1[:,:,iLR], axis=(0, 1)))**2 
         
        ERGAS_index = (100/ratio) * math.sqrt((1/I1.shape[2]) * ERGAS_index)
            
            
    return np.squeeze(ERGAS_index)