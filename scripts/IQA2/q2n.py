# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )
"""

"""
 Description: 
           Q2n index. 
 
 Interface:
           [Q2n_index, Q2n_index_map] = q2n(I_GT, I_F, Q_blocks_size, Q_shift)

 Inputs:
           I_GT:               Ground-Truth image;
           I_F:                Fused Image;
           Q_blocks_size:      Block size of the Q-index locally applied;
           Q_shift:            Block shift of the Q-index locally applied.

 Outputs:
           Q2n_index:          Q2n index;
           Q2n_index_map:      Map of Q2n values.

 References:
         G. Vivone, M. Dalla Mura, A. Garzelli, and F. Pacifici, "A Benchmarking Protocol for Pansharpening: Dataset, Pre-processing, and Quality Assessment", IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 6102-6118, 2021.
         G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting Pansharpening With Classical and Emerging Pansharpening Methods", IEEE Geoscience and Remote Sensing Magazine, vol. 9, no. 1, pp. 53 - 81, March 2021.
"""

import math
import numpy as np
from scripts.IQA.onions_quality import onions_quality

def q2n(I_GT, I_F, Q_blocks_size, Q_shift):

    if np.size(I_GT.shape) == 2 :
        
        Q2n_index_map = np.zeros(I_GT.shape)

        Q2n_index = 0
        
    else:
        N1 = I_GT.shape[0]
        N2 = I_GT.shape[1]
        N3 = I_GT.shape[2]
        
        size2 = Q_blocks_size
        
        stepx = math.ceil(N1/Q_shift)
        stepy = math.ceil(N2/Q_shift)
         
        if (stepy <= 0):
            stepy = 1
            stepx = 1
        
        est1 = (stepx - 1) * Q_shift + Q_blocks_size - N1
        est2 = (stepy - 1) * Q_shift + Q_blocks_size - N2
           
        if (est1 != 0) or (est2 != 0):
          refref = []
          fusfus = []
          
          for i in range(N3):
              a1 = np.squeeze(I_GT[:,:,0])
            
              ia1 = np.zeros((N1+est1,N2+est2))
              ia1[0:N1,0:N2] = a1
              ia1[:,N2:N2+est2] = ia1[:,N2-1:N2-est2-1:-1]
              ia1[N1:N1+est1,:] = ia1[N1-1:N1-est1-1:-1,:]
    
              if i == 0:
                  refref = ia1
              elif i == 1:
                  refref = np.concatenate((refref[:,:,None],ia1[:,:,None]),axis=2)
              else:
                  refref = np.concatenate((refref,ia1[:,:,None]),axis=2)
              
              if (i < (N3-1)):
                  I_GT = I_GT[:,:,1:I_GT.shape[2]]
              
          I_GT = refref
                
          for i in range(N3):
              a2 = np.squeeze(I_F[:,:,0])
              
              ia2 = np.zeros((N1+est1,N2+est2))
              ia2[0:N1,0:N2] = a2
              ia2[:,N2:N2+est2] = ia2[:,N2-1:N2-est2-1:-1]
              ia2[N1:N1+est1,:] = ia2[N1-1:N1-est1-1:-1,:]
              
              if i == 0:
                  fusfus = ia2
              elif i == 1:
                  fusfus = np.concatenate((fusfus[:,:,None],ia2[:,:,None]),axis=2)
              else:
                  fusfus = np.concatenate((fusfus,ia2[:,:,None]),axis=2)
              
              if (i < (N3-1)):
                  I_F = I_F[:,:,1:I_F.shape[2]]
              
          I_F = fusfus
          
        #I_F = np.uint16(I_F)
        #I_GT = np.uint16(I_GT)
        
        N1 = I_GT.shape[0]
        N2 = I_GT.shape[1]
        N3 = I_GT.shape[2]
        
        if (((math.ceil(math.log2(N3))) - math.log2(N3)) != 0):
            Ndif = (2**(math.ceil(math.log2(N3)))) - N3
            dif = np.zeros((N1,N2,Ndif))
            #dif = np.uint16(dif)
            I_GT = np.concatenate((I_GT, dif), axis = 2)
            I_F = np.concatenate((I_F, dif), axis = 2)
        
        N3 = I_GT.shape[2]
        
        valori = np.zeros((stepx,stepy,N3))
        
        for j in range(stepx):
            for i in range(stepy):
                o = onions_quality(I_GT[ (j*Q_shift):(j*Q_shift)+Q_blocks_size, (i*Q_shift):(i*Q_shift)+size2,:], I_F[ (j*Q_shift):(j*Q_shift)+Q_blocks_size,(i*Q_shift):(i*Q_shift)+size2,:], Q_blocks_size)
                valori[j,i,:] = o    
            
        Q2n_index_map = np.sqrt(np.sum(valori**2, axis=2))
    
        Q2n_index = np.mean(Q2n_index_map)
    
    return Q2n_index, Q2n_index_map