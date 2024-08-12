# -*- coding: utf-8 -*-
"""
Copyright (c) 
All rights reserved. This work should only be used for nonprofit purposes.

@author: 
    Alessia Carbone (alcarbone@unisa.it)
"""

"""
 Description: 
     nearest-neighbour kernel definition.
 
     Interface: y = nearest(s)
    
     Inputs: s: x coordinate.
     
     Outputs: y: y coordinate.
          
"""
def nearest(s):
    if (abs(s) <= 1/2):
        return 1
    else:
        return 0