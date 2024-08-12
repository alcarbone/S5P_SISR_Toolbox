# -*- coding: utf-8 -*-
"""
Copyright (c) 
All rights reserved. This work should only be used for nonprofit purposes.

@author: 
    Alessia Carbone (alcarbone@unisa.it)
"""

"""
 Description: 
     The value of X is computed through the conjugate gradients 
     algorithm generally used to solve linear systems of equations.
 
     Interface: it, img_SR = cg2(S1,S2,D1,D2,img_LR,k,l,delta,iterations,GT)
    
     Inputs: S1: S matrix for rows;
             S2: S matrix for cols;
             D1: first differences matrix for rows;
             D2: first differences matrix for cols;
             img_LR: "low-resolution" image;
             m: mu shrinking parameter (default is 0);
             l: lambda smoothing parameter (default is 1);
             delta: threshold to end the algorithm (default is 1e-3);
             iterations: maximum number of iterations (default is 100).
         
     Outputs: img_SR: estimated image;
              it: last iteration.
     
 References:
     P. H. C. Eilers and C. Ruckebusch, ‘Fast and simple super-resolution with single images’, Sci Rep, vol. 12, no. 1, Art. no. 1, Jul. 2022, doi: 10.1038/s41598-022-14874-8.
          
"""

import numpy as np
from scripts.interp.interp23 import interp23
from scripts.interp.interp23_1 import interp23_1


def cg2(S1,S2,D1,D2,img_LR,m=0,l=1,delta=1e-3,iterations=100):
    
    G1 = np.dot(np.transpose(S1),S1) # S1'*S1 
    G2 = np.dot(np.transpose(S2),S2) # S2'*S2
    
    n1 = G1.shape[1]
    n2 = G2.shape[1]
    
    img_SR = np.zeros([n1,n2]) 
    norm_LR = np.sqrt(np.sum(img_LR**2))
    
    V1 = l*np.dot(np.transpose(D1),D1) # l*D1'*D1
    V2 = l*np.dot(np.transpose(D2),D2) # l*D2'*D2
    
    U = np.dot(np.dot(np.transpose(S1),img_LR),S2) # S1'*img_LR*S2 
    if((S1.shape[1]//S1.shape[0])%2 == 0):
        X = interp23(img_LR,S1.shape[1]//S1.shape[0]) # the initial image is the interpolation of img_LR
    else:
        X = interp23_1(img_LR,S1.shape[1]//S1.shape[0]) # the initial image is the interpolation of img_LR
        X = X.squeeze()
    R = U-(np.dot(np.dot(G1,X),G2) + np.dot(np.dot(V1,X),np.transpose(V2)) + m*X )
    P = R 
    for it in range(iterations):
        Q = np.dot(np.dot(G1,P),G2) + np.dot(np.dot(V1,P),np.transpose(V2))  + m*P 
        alpha = np.sum(R**2)/(np.sum(P*Q)+10**(-15))
        X = X + alpha*P  # new estimation for x based on alpha  
        Rnew = R - alpha*Q  # residual computation
        rs1 = np.sum(np.sum(R[:,:]**2))
        rs2 = np.sum(np.sum(Rnew[:,:]**2))
        beta = rs2/rs1
        if np.sqrt(rs2) < delta * norm_LR :  # if the threshold (weighted on the norm) 
            break                            # is reached the algorithm is over
        P = Rnew + beta*P  # new direction based on the residual
        R = Rnew  # matrix for residual computation is updated 
    img_SR = X 
    
    return it,img_SR

