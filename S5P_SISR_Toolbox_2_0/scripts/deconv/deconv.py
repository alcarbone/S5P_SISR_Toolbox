# -*- coding: utf-8 -*-
"""
Copyright (c) 
All rights reserved. This work should only be used for nonprofit purposes.

@author: 
    Alessia Carbone (alcarbone@unisa.it)
"""

"""
 Description:
       The process of super-resolution can be considered as the solution of an
       optimization problem, once chosen a method of resolution. In the definition
       when both m and l are equal to 0, the regularization is a least-squares
       regularization. When l is equal to 0 and m is not equal to zero, the
       regularization is a Tikhonov regularization.
       
       Interface: SR, all_it, S1, S2 = deconv(LR,GT,ratio,GNyq_x,GNyq_y,method,iters,delta,m,l)
      
       Inputs: LR: "low-resolution" image;
               GT: GT image;
               ratio: scaling ratio;
               GNyq_x: Nyquist gain on x axis;
               GNyq_y: Nyquist gain on y axis;
               method: method to solve system;
               iters: maximum number of iterations (default is 100);
               delta: threshold to end the algorithm (default is 1e-3);
               m: mu shrinking parameter (default is 0);
               l: lambda smoothing parameter (default is 1).
           
       Outputs: SR: SR output image;
                all_it: last iteration for each band;
                S1, S2: up-sampling matrices for rows and columns.
       
 References:
       P. H. C. Eilers and C. Ruckebusch, ‘Fast and simple super-resolution with single images’, Sci Rep, vol. 12, no. 1, Art. no. 1, Jul. 2022, doi: 10.1038/s41598-022-14874-8.

"""

import numpy as np
from scripts.deconv.cga import cg2
from scripts.resize_image import genPSF, psf_BCCB
 
def deconv(LR,GT,ratio,GNyq_x,GNyq_y,method,iters=100,delta=1e-3,m=0,l=1):
    
    N = 41
    
    if (len(LR.shape) == 2):
        LR1 = np.zeros([LR.shape[0],LR.shape[1],1])
        LR1[:,:,0] = LR
        LR = LR1
        
        GT1 = np.zeros([GT.shape[0],GT.shape[1],1])
        GT1[:,:,0] = GT
        GT = GT1
        
        nbands = 1
    else:
        nbands = LR.shape[2]
        
    dim = LR.shape 
    new_dim = GT.shape
    
    C1 = np.zeros([dim[0],new_dim[0]])  
    for jj in range(dim[0]):
        C1[jj,jj*ratio+ratio//2] = 1
    
    C2 = np.zeros([dim[1],new_dim[1]])  
    for jj in range(dim[1]):
        C2[jj,jj*ratio+ratio//2] = 1 
        
    h = genPSF(ratio, N, GNyq_x, GNyq_y, nbands)
        
    h_1D_el = np.zeros((1,h.shape[0]))
    h_1D_az = np.zeros((1,h.shape[0]))
    
    SR = np.zeros(new_dim) 
    
    D1 = np.diff(np.identity(new_dim[0]),append=0)  
    D1 = np.transpose(D1)
    D2 = np.diff(np.identity(new_dim[1]),append=0)  
    D2 = np.transpose(D2)
    
    all_it = [0]*nbands
    
    for idim in range(SR.shape[2]):
        
        h_1D_el[0,:] = h[(h.shape[0]-1)//2,:,idim]/np.sqrt(h[(h.shape[0]-1)//2,(h.shape[0]-1)//2,idim])
        h_1D_az[0,:] = h[:,(h.shape[1]-1)//2,idim]/np.sqrt(h[(h.shape[1]-1)//2,(h.shape[1]-1)//2,idim])
        
        h_1D_el = h_1D_el/sum(h_1D_el[0,:])
        h_1D_az = h_1D_az/sum(h_1D_az[0,:])
        
        H_MTF_el = psf_BCCB(new_dim[0],h_1D_el,1)
        H_MTF_az = psf_BCCB(new_dim[1],h_1D_az,1)
         
        S1 = np.dot(C1,H_MTF_el)  
        S2 = np.dot(C2,H_MTF_az)
        
        if method == 'cga':
            it,SR[:,:,idim] = cg2(S1,S2,D1,D2,LR[:,:,idim],m,l,delta,iters)
            
        all_it[idim] = it
        
    return SR, all_it, S1, S2