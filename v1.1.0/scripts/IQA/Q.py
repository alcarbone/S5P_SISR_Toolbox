# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 Vivone Gemine.
All rights reserved. This work should only be used for nonprofit purposes.

@author: Vivone Gemine (website: https://sites.google.com/site/vivonegemine/home )
"""

"""
 Description: 
           Q/SSIM averaged over all the spectral bands.
 
 Interface:
           Q_avg = Q(I1,I2,L)

 Inputs:
           I1:         First multispectral image;
           I2:         Second multispectral image;
           L:          Radiometric resolution;
           data_range: The range of the data;
           K1, K2:     Regularization parameters for SSIM.

 Outputs:
           Q_avg:      Q index averaged on all bands.
 
 Notes:
     Results very close to the MATLAB toolbox's ones. In particular, when S is odd the results are identical.
     Instead, when S is even ssim cannot work and S is modified to the closest odd number. In this way, the results are very very similar
     with respect to the MATLAB toolbox's ones when S is even. 

 References:
         G. Vivone, M. Dalla Mura, A. Garzelli, and F. Pacifici, "A Benchmarking Protocol for Pansharpening: Dataset, Pre-processing, and Quality Assessment", IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, vol. 14, pp. 6102-6118, 2021.
         G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M. O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting Pansharpening With Classical and Emerging Pansharpening Methods", IEEE Geoscience and Remote Sensing Magazine, vol. 9, no. 1, pp. 53 - 81, March 2021.           
"""

from skimage.metrics import structural_similarity as ssim
import numpy as np

def Q(I1,I2,S,data_range,K1,K2):
    
    I1

    if np.size(I2.shape) == 2 :
         Q_orig = ssim(I1,I2, win_size=S, data_range=data_range)
             
    else:
            
        Q_orig = np.zeros((I1.shape[2],1))
    
        for idim in range(I1.shape[2]):
            Q_orig[idim] = ssim(I1[:,:,idim],I2[:,:,idim], win_size=S, data_range=data_range, K1=K1, K2=K2)

    return np.mean(Q_orig)