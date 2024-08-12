# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )
"""

""" Q2n aux. function """

import numpy as np

def onion_mult(onion1,onion2):

    N = onion1.size
    
    if (N > 1):      
        L = int(N/2)   
        
        a = onion1[0,0:L]
        b = onion1[0,L:onion1.shape[1]]
        b = np.concatenate(([b[0]],-b[1:b.shape[0]]))
        c = onion2[0,0:L]
        d = onion2[0,L:onion2.shape[1]]
        d = np.concatenate(([d[0]],-d[1:d.shape[0]]))
    
        if (N == 2):
            ris = np.concatenate((a*c-d*b, a*d+c*b))
        else:
            ris1 = onion_mult(np.reshape(a,(1,a.shape[0])),np.reshape(c,(1,c.shape[0])))
            ris2 = onion_mult(np.reshape(d,(1,d.shape[0])),np.reshape(np.concatenate(([b[0]],-b[1:b.shape[0]])),(1,b.shape[0])))
            ris3 = onion_mult(np.reshape(np.concatenate(([a[0]],-a[1:a.shape[0]])),(1,a.shape[0])),np.reshape(d,(1,d.shape[0])))
            ris4 = onion_mult(np.reshape(c,(1,c.shape[0])),np.reshape(b,(1,b.shape[0])))
    
            aux1 = ris1 - ris2
            aux2 = ris3 + ris4
    
            ris = np.concatenate((aux1,aux2))
    else:
        ris = onion1 * onion2

    return ris