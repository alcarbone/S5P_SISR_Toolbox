# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )
"""

""" Q2n aux. function """

import numpy as np

def onion_mult2D(onion1,onion2):

    N3 = onion1.shape[2]

    if (N3 > 1):
        L = int(N3/2)
        a = onion1[:,:,0:L]
        b = onion1[:,:,L:onion1.shape[2]]
        h = b[:,:,0]
        b = np.concatenate((h[:,:,None],-b[:,:,1:b.shape[2]]),axis=2)
        c = onion2[:,:,0:L]
        d = onion2[:,:,L:onion2.shape[2]]
        h = d[:,:,0]
        d = np.concatenate((h[:,:,None],-d[:,:,1:d.shape[2]]),axis=2)

        if (N3 == 2):
            ris = np.concatenate((a*c-d*b,a*d+c*b),axis=2)
        else:
            ris1 = onion_mult2D(a,c)
            h = b[:,:,0]
            ris2 = onion_mult2D(d,np.concatenate((h[:,:,None],-b[:,:,1:b.shape[2]]),axis=2))
            h = a[:,:,0]
            ris3 = onion_mult2D(np.concatenate((h[:,:,None],-a[:,:,1:a.shape[2]]),axis=2),d)
            ris4 = onion_mult2D(c,b)
            aux1 = ris1 - ris2
            aux2 = ris3 + ris4
            ris = np.concatenate((aux1,aux2), axis=2)
    else:
        ris = onion1 * onion2   

    return ris