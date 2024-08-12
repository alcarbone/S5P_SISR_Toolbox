# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )
"""
"""
 Description: 
           Reduced resolution quality indexes. 
 
 Interface:
           [Q2n_index, Q_index, ERGAS_index, SAM_index] = indexes_evaluation(I_F,I_GT,ratio,L,Q_blocks_size,flag_cut_bounds,dim_cut,th_values)

 Inputs:
           I_F:                Super-resolved Image;
           I_LR:               Low-resolution image;
           I_GT:               Ground-Truth image;
           ratio:              Scale ratio between low- and high-resolution. Pre-condition: Integer value;
           data_range:         The range of the data;
           GNyq_x:             Nyquist gain on x-axis;
           GNyq_y:             Nyquist gain on y-axis;
           max_img:            Maximum value for each channel;
           Q_blocks_size:      Block size of the Q-index locally applied;
           flag_cut_bounds:    Cut the boundaries of the viewed Panchromatic image;
           dim_cut:            Define the dimension of the boundary cut;
           th_values:          Flag. If th_values == 1, apply an hard threshold to the dynamic range;
           K1:                 regularitazion value in SSIM;
           K2:                 regularization value in SSIM.

 Outputs:
           Q2n_index:          Q2n index;
           Dlambda:            spectral distortion index; 
           Q_index:            Q index;
           ERGAS_index:        Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS) index;
           SAM_index:          Spectral Angle Mapper (SAM) index;
           sCC_index:          spatial Correlation Coefficient;
           RMSE_index:         Root Mean Square Error;
           PSNR_index:         Averaged Peak Signal-to-Noise Ratio.
 
 References:
         G. Vivone, M. Dalla Mura, A. Garzelli, and F. Pacifici, "A Benchmarking Protocol for Pansharpening: Dataset, Pre-processing, and Quality Assessment", IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 6102-6118, 2021.
         G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting Pansharpening With Classical and Emerging Pansharpening Methods", IEEE Geoscience and Remote Sensing Magazine, vol. 9, no. 1, pp. 53 - 81, March 2021.
"""

import numpy as np

from scripts.IQA.ERGAS import ERGAS 
from scripts.IQA.SAM import SAM 
from scripts.IQA.Q import Q
from scripts.IQA.q2n import q2n
from scripts.IQA.RMSE import RMSE
from scripts.IQA.sCC import sCC

from scripts.resize_image import resize_image

def indexes_evaluation_SR(I_F,I_LR,I_GT,ratio,data_range,GNyq_x,GNyq_y,max_img,Qblocks_size=32,flag_cut_bounds=0,dim_cut=0,th_values=0,K1=0,K2=0):
    I_F = I_F.astype("float32")
    I_LR = I_LR.astype("float32")
    I_GT = I_GT.astype("float32")
    
    if(len(I_F.shape) == 2):
        I_F1 = np.zeros([I_F.shape[0],I_F.shape[1],1])
        I_F1[:,:,0] = I_F
        I_F = I_F1
        
    if(len(I_GT.shape) == 2):
        I_GT1 = np.zeros([I_GT.shape[0],I_GT.shape[1],1])
        I_GT1[:,:,0] = I_GT
        I_GT = I_GT1
        
    Dlambda = 1
    if (I_GT.shape[2] > 2 and I_GT.shape[2] < 300 and I_LR is not None):
        I_F_LR = np.zeros([I_F.shape[0]//ratio,I_F.shape[1]//ratio,I_F.shape[2]])
        for ii in range(0,I_F.shape[2]):
            a = resize_image(I_F[:,:,ii],ratio,GNyq_x[ii],GNyq_y[ii])
            I_F_LR[:,:,ii] = a[:,:,0]
        q2, q2n_map = q2n(I_LR, I_F_LR, Qblocks_size, Qblocks_size)
        Dlambda = 1 - q2
    elif (I_GT.shape[2] >= 300):
        I_F_LR = np.zeros([I_F.shape[0]//ratio,I_F.shape[1]//ratio,I_F.shape[2]])
        for ii in range(0,I_F.shape[2]):
            a = resize_image(I_F[:,:,ii],ratio,GNyq_x[ii],GNyq_y[ii])
            I_F_LR[:,:,ii] = a[:,:,0]
        q2, q2n_map = q2n(I_LR[:,:,::(I_GT.shape[2]//100)], I_F_LR[:,:,::(I_GT.shape[2]//100)], Qblocks_size, Qblocks_size)
        Dlambda = 1 - q2
        
    
    """ cut bounds """
    if (flag_cut_bounds == 1):
        I_GT = I_GT[dim_cut-1:I_GT.shape[0]-dim_cut,dim_cut-1:I_GT.shape[1]-dim_cut,:]
        I_F = I_F[dim_cut-1:I_F.shape[0]-dim_cut,dim_cut-1:I_F.shape[1]-dim_cut,:]
    
    if th_values == 1:
        I_F[I_F > data_range[1]] = data_range[1]
        I_F[I_F < data_range[0]] = data_range[0]
        
    if np.size(I_GT.shape) == 2 :
        
        SAM_index = 0
        SAM_map = np.zeros(I_GT.shape)
        Q2n_index = 0 
        Q2n_index_map = np.zeros(I_GT.shape)
        
    elif (I_GT.shape[2]) == 1 :
                       
            SAM_index = 0
            SAM_map = np.zeros(I_GT.shape)
            Q2n_index = 0 
            Q2n_index_map = np.zeros(I_GT.shape)
            
    else :
        
        SAM_index, SAM_map = SAM(I_GT,I_F)
        
        Q2n_index = 0
        if I_GT.shape[2] < 300:
            Q2n_index, Q2n_index_map = q2n(I_GT, I_F, Qblocks_size, Qblocks_size)
        else:
            Q2n_index, Q2n_index_map = q2n(I_GT[:,:,::(I_GT.shape[2]//100)], I_F[:,:,::(I_GT.shape[2]//100)], Qblocks_size, Qblocks_size)
     
    ERGAS_index = ERGAS(I_GT,I_F,ratio)
    
    if np.remainder(Qblocks_size,2) == 0:
        Q_index = Q(I_GT,I_F,Qblocks_size + 1, data_range[1]-data_range[0], K1, K2)
    else:
        Q_index = Q(I_GT,I_F,Qblocks_size, data_range[1]-data_range[0], K1, K2)
    
    sCC_index = 0
    if I_GT.shape[2] < 300:
        sCC_index = sCC(I_GT,I_F)
    else:
        sCC_index = sCC(I_GT[:,:,::(I_GT.shape[2]//100)],I_F[:,:,::(I_GT.shape[2]//100)])
    
    RMSE_index = RMSE(I_GT,I_F)
    
    if (len(I_GT.shape) == 2 or (len(I_GT.shape) == 3 and I_GT.shape[2] == 1)):
        PSNR_index = 20*np.log10(data_range[1]/RMSE(I_GT,I_F))
    else:
        PSNRs = list()
        for ii in range(I_GT.shape[2]):
            RMSE_idx = np.sqrt(np.mean((I_F[:,:,ii]-I_GT[:,:,ii])**2))
            PSNRs.append(20*np.log10(max_img[ii]/RMSE_idx))
        PSNR_index = np.mean(PSNRs)
    
    return Q2n_index, Dlambda, Q_index, ERGAS_index, SAM_index, sCC_index, RMSE_index, PSNR_index