# -*- coding: utf-8 -*-
"""
Copyright (c) 
All rights reserved. This work should only be used for nonprofit purposes.

@author: 
    Alessia Carbone (alcarbone@unisa.it)
"""

"""
 Description:
     The process of super-resolution can be considered as the convolution
     of an enlarged version of the original LR image and an interpolation kernel.
     Note that the dimension of the kernel depends on the ratio value. 
     However, if ratio is even and the kernel degree is even too, a shift of half 
     pixel will occur in the image.
     
     Interface: k = kernel(tc,ratio,interp_type)
    
     Inputs: tc: sampling time;
             ratio: expansion factor;
             interp_type: the type of interpolation.
    
     Outputs: k: kernel values.
     
 References:
      [1] B. Aiazzi, S. Baronti, M. Selva, and L. Alparone, ‘Bi-cubic interpolation for shift-free pan-sharpening’, ISPRS Journal of Photogrammetry and Remote Sensing, vol. 86, pp. 65–76, Dec. 2013, doi: 10.1016/j.isprsjprs.2013.09.007.
          
"""

import sys
import math
import numpy as np

from scripts.interp.cubic import cubic

def kernel(tc,ratio,interp_type):   
    if (interp_type == 'cubic'):
        q = 4
        M = q*(ratio-1)+2*math.floor((q+1)/2)-1
        k = np.zeros(M)
        x = -((M-1)/2)*tc/ratio
        for i in range(len(k)):
            k[i] = cubic(x,-0.5)
            x = x + tc/ratio
    else:
        print('Interpolation not available', file=sys.stderr)
        return None   
    return k