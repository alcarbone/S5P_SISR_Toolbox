# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )
"""

""" Q2n aux. function """
import numpy as np

def norm_blocco(x):
    
    a = np.mean(x)
    c = np.std(x, ddof=1)
    
    if (c==0):
    	c = 2.2204 * 10**(-16)
    
    y = ((x - a)/c) + 1
    
    return y, a, c