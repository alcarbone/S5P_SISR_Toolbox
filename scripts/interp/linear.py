# -*- coding: utf-8 -*-
"""
Copyright (c) 
All rights reserved. This work should only be used for nonprofit purposes.

@author: 
    Alessia Carbone (alcarbone@unisa.it)
"""

"""
 Description: 
     linear kernel definition.
 
     Interface: y = linear(s)
    
     Inputs: s: x coordinate.
     
     Outputs: y: y coordinate.
          
"""

def linear(s):
    if (abs(s) <= 1):
        return 1 - abs(s)
    else:
        return 0