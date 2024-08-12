# -*- coding: utf-8 -*-
"""
Copyright (c) 
All rights reserved. This work should only be used for nonprofit purposes.

@author: 
    Alessia Carbone (alcarbone@unisa.it)
"""

"""
 Description: 
     Lanczos kernel definition.
 
     Interface: y = lanczos(s,a)
    
     Inputs: s: x coordinate;
             a: half interval of definition of the kernel.
     
     Outputs: y: y coordinate.
          
"""

import math

def lanczos(s, a):
    if s == 0:
        return 1
    elif (s >= -a) & (s < a) & (s != 0):
        return ((a*math.sin(math.pi*s)*math.sin(math.pi*s/a))/(((math.pi)**2)*((s)**2)))
    else:
        return 0