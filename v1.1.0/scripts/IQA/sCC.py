# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 
All rights reserved. This work should only be used for nonprofit purposes.

@author: 
"""

"""
 Description: 
          spatial Correlation Index (sCC).

 Interface:
           sCC_index = sCC(I1,I2)

 Inputs:
           I1:             First multispectral image;
           I2:             Second multispectral image;
           
 
 Outputs:
           sCC_index:    sCC index.
 
 References:
"""

import numpy as np
import math
from sklearn.metrics import mean_squared_error
from scipy.ndimage import sobel

def sCC(I1,I2):

    I1 = I1.astype('float64')
    I2 = I2.astype('float64')
    I_Lap_1 = np.zeros(I1.shape);
    I_Lap_2 = np.zeros(I2.shape);
    
    if np.size(I2.shape) == 2:
       
           I_Lap_1_x = sobel(I1,axis=1);
           I_Lap_1_y = sobel(I1,axis=0);
           I_Lap_1 = np.sqrt(I_Lap_1_x**2+I_Lap_1_y**2);
           
           I_Lap_2_x = sobel(I2,axis=1);
           I_Lap_2_y = sobel(I2,axis=0);
           I_Lap_2 = np.sqrt(I_Lap_2_x**2+I_Lap_2_y**2);
           
           SCCMap=(I_Lap_1*I_Lap_2)/np.sqrt(np.sum(I_Lap_1**2))/np.sqrt(np.sum(I_Lap_1**2))
    
    else:
 
        for idim in range(I1.shape[2]):
            I_Lap_1_x = sobel(I1[:,:,idim],axis=1);
            I_Lap_1_y = sobel(I1[:,:,idim],axis=0);
            I_Lap_1[:,:,idim] = np.sqrt(I_Lap_1_x**2+I_Lap_1_y**2);
        
        
            I_Lap_2 = np.zeros(I2.shape);
        for idim in range(I2.shape[2]):
            I_Lap_2_x = sobel(I2[:,:,idim],axis=1);
            I_Lap_2_y = sobel(I2[:,:,idim],axis=0);
            I_Lap_2[:,:,idim] = np.sqrt(I_Lap_2_x**2+I_Lap_2_y**2);
        
        SCCMap=np.sum(I_Lap_1*I_Lap_2,axis=2)/np.sqrt(np.sum(I_Lap_1**2))/np.sqrt(np.sum(I_Lap_1**2))


    sCC=np.sum(I_Lap_1*I_Lap_2);
    sCC = sCC/np.sqrt(np.sum(I_Lap_1**2));
    sCC = sCC/np.sqrt(np.sum(I_Lap_2**2));

    
    
           
    return sCC, SCCMap