# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )
"""

""" Q2n aux. function """

import numpy as np
from scripts.IQA.onion_mult2D import onion_mult2D
from scripts.IQA.onion_mult import onion_mult
from scripts.IQA.norm_blocco import norm_blocco

def onions_quality(dat1,dat2,size1):

    #dat1 = dat1.astype('float64')
    #dat2 = dat2.astype('float64')
    
    
    h = dat2[:,:,0]
    dat2 = np.concatenate((h[:,:,None],-dat2[:,:,1:dat2.shape[2]]),axis=2)
    
    N3 = dat1.shape[2]
    size2 = size1
    
    """ Block normalization """
    for i in range(N3):
      a1,s,t = norm_blocco(np.squeeze(dat1[:,:,i]))
      dat1[:,:,i] = a1
      
      if (s == 0):
          if (i == 0):
              dat2[:,:,i] = dat2[:,:,i] - s + 1
          else:
              dat2[:,:,i] = -(-dat2[:,:,i] - s + 1)
      else:
          if (i == 0):
              dat2[:,:,i] = (dat2[:,:,i] - s)/t + 1
          else:
              dat2[:,:,i] = -(((-dat2[:,:,i] - s)/t) + 1)    
    
    m1 = np.zeros((1,N3))
    m2 = np.zeros((1,N3))
    
    mod_q1m = 0
    mod_q2m = 0
    mod_q1 = np.zeros((size1,size2))
    mod_q2 = np.zeros((size1,size2))
    
    for i in range(N3):
        m1[0,i] = np.mean(np.squeeze(dat1[:,:,i]))
        m2[0,i] = np.mean(np.squeeze(dat2[:,:,i]))
        mod_q1m = mod_q1m + m1[0,i]**2
        mod_q2m = mod_q2m + m2[0,i]**2
        mod_q1 = mod_q1 + (np.squeeze(dat1[:,:,i]))**2
        mod_q2 = mod_q2 + (np.squeeze(dat2[:,:,i]))**2
    
    mod_q1m = np.sqrt(mod_q1m)
    mod_q2m = np.sqrt(mod_q2m)
    mod_q1 = np.sqrt(mod_q1)
    mod_q2 = np.sqrt(mod_q2)
    
    termine2 = mod_q1m * mod_q2m
    termine4 = mod_q1m**2 + mod_q2m**2
    int1 = (size1 * size2)/((size1 * size2)-1) * np.mean(mod_q1**2)
    int2 = (size1 * size2)/((size1 * size2)-1) * np.mean(mod_q2**2)
    termine3 = int1 + int2 - (size1 * size2)/((size1 * size2) - 1) * ((mod_q1m**2) + (mod_q2m**2))
    
    mean_bias = 2*termine2/termine4
    
    if (termine3==0):
        q = np.zeros((1,1,N3))
        q[:,:,N3-1] = mean_bias
    else:
        cbm = 2/termine3
        qu = onion_mult2D(dat1, dat2)
        qm = onion_mult(m1, m2)
        qv = np.zeros((1,N3))
        for i in range(N3):
            qv[0,i] = (size1 * size2)/((size1 * size2)-1) * np.mean(np.squeeze(qu[:,:,i]))
        q = qv - ((size1 * size2)/((size1 * size2) - 1.0)) * qm
        q = q * mean_bias * cbm
    return q