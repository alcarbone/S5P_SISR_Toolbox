# -*- coding: utf-8 -*-
"""
Copyright (c) 
All rights reserved. This work should only be used for nonprofit purposes.

@author: 
    Alessia Carbone (alcarbone@unisa.it)
"""

"""
 Description: 
     quadratic kernel definition.
 
     Interface: y = quadratic(s)
    
     Inputs: s: x coordinate.
     
     Outputs: y: y coordinate.
          
"""

def quadratic(s):
    if abs(s) <= 1/2:
        return -(2*(abs(s)**2))+1
    elif (abs(s) > 1/2) & (abs(s) <= 3/2):
        return (abs(s)**2) - 5/2*abs(s) + 3/2
    else:
        return 0  