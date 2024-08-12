# -*- coding: utf-8 -*-
"""

"""
"""
Copyright (c) 2020 
All rights reserved. This work should only be used for nonprofit purposes.

@author: 
    
"""
"""
 Description: 
          Root Mean Square Error (RMSE).

 Interface:
           RMSE_index = RMSE(I1,I2)

 Inputs:
           I1:             First multispectral image;
           I2:             Second multispectral image;
           
 
 Outputs:
           RMSE_index:    RMSE index.
 
 References:
"""

import numpy as np
import math
from sklearn.metrics import mean_squared_error

def RMSE(I1,I2):

    I1 = I1.astype('float64')
    I2 = I2.astype('float64')

    
    RMSE = np.sqrt(np.mean((I1-I2)**2))
    
    # RMSE = np.sqrt(mean_squared_error(I1,I2))       
            
    return RMSE