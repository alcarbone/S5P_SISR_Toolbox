# -*- coding: utf-8 -*-
"""
Copyright (c) 
All rights reserved. This work should only be used for nonprofit purposes.

@author: 
    Alessia Carbone (alcarbone@unisa.it)
"""

"""
 Description:
           Resize tool. In resize_image the undersampling matrices are created 
           for the two directions (C1 and C2) and the psf is generated by 
           method genPSF depending on the gains.
           With check_gains it is possible to check if the gains of the 
           generated psf are equal to the initial gains. With plot_psf it is
           possible to plot the retrieved psf.
           Then, for each band in the image, the 1D filter is retrieved in both 
           the dimensions, it is normalized, and the MTF is obtained by method
           psf_BCCB by shifting the 1D filter in the two directions.
           The degradation matrix in the two directions is obtained by 
           multiplying the MTF and the downsampling matrix.
           The image is degraded by multiplying it with the degradation 
           matrices. 
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

def resize_image(img, ratio, GNyq_x, GNyq_y):
      
    dim = img.shape 
    
    if (len(dim) == 3):
        new_dim = [dim[0]//ratio,dim[1]//ratio,dim[2]]
        nbands = dim[2]
    else:
        new_dim = [dim[0]//ratio,dim[1]//ratio,1]
        nbands = 1
        
        img1 = np.zeros([dim[0],dim[1],1])
        img1[:,:,0] = img
        img = img1
        
    C1 = np.zeros([new_dim[0],dim[0]])  
    for jj in range(C1.shape[0]):
        C1[jj,jj*ratio+ratio//2] = 1
    
    C2 = np.zeros([new_dim[1],dim[1]])  
    for jj in range(C2.shape[0]):
        C2[jj,jj*ratio+ratio//2] = 1
        
    N = 41
    
    h = genPSF(ratio, N, GNyq_x, GNyq_y, nbands)
    
    # check_gains(h[:,:,0], N, 1/ratio)    
    # plot_psf(h[:,:,0], N)
          
    h_1D_x = np.zeros((1,h.shape[0]))
    h_1D_y = np.zeros((1,h.shape[0]))
    
    img_LR = np.zeros(new_dim)
    
    for i in range(nbands):
        h_1D_x[0,:] = h[(h.shape[0]-1)//2,:,i]/np.sqrt(h[(h.shape[0]-1)//2,(h.shape[0]-1)//2,i])
        h_1D_y[0,:] = h[:,(h.shape[1]-1)//2,i]/np.sqrt(h[(h.shape[1]-1)//2,(h.shape[1]-1)//2,i])
        
        h_1D_x = h_1D_x/sum(h_1D_x[0,:])
        h_1D_y = h_1D_y/sum(h_1D_y[0,:])
        
        H_MTF_x = psf_BCCB(dim[0],h_1D_x,1)
        H_MTF_y = psf_BCCB(dim[1],h_1D_y,1)
        
        S1 = np.dot(C1,H_MTF_x)  
        S2 = np.dot(C2,H_MTF_y)  
           
        img_LR[:,:,i] = np.dot(np.dot(S1,img[:,:,i]),np.transpose(S2))
    
    return img_LR

def genPSF(ratio, N, GNyq_x, GNyq_y, nbands):
    
    fcut = 1/ratio
    
    h = np.zeros((N,N,nbands))
    
    if(nbands == 1):
        alpha_x = np.sqrt(((N-1)*(fcut/2))**2/(-2*np.log(GNyq_x)))
        alpha_y = np.sqrt(((N-1)*(fcut/2))**2/(-2*np.log(GNyq_y)))
        
        H = gaussian2d_asymmetric(N,alpha_x,alpha_y) 
        Hd = H/np.max(H)
        
        w = kaiser2d(N,0.5)
        h = np.real(fir_filter_wind(Hd,w))
        
        h1 = np.zeros([h.shape[0],h.shape[1],1])
        h1[:,:,0] = h
        h = h1
        
    else:
        
        for ii in range(nbands):
            alpha_x = np.sqrt(((N-1)*(fcut/2))**2/(-2*np.log(GNyq_x[ii])))
            alpha_y = np.sqrt(((N-1)*(fcut/2))**2/(-2*np.log(GNyq_y[ii])))
            
            H = gaussian2d_asymmetric(N,alpha_x,alpha_y) 
            Hd = H/np.max(H)
            
            w = kaiser2d(N,0.5)
            h[:,:,ii] = np.real(fir_filter_wind(Hd,w))
            
    return h
    
def fir_filter_wind(Hd,w):
    
    hd = np.rot90(np.fft.fftshift(np.rot90(Hd,2)),2)
    h = np.fft.fftshift(np.fft.ifft2(hd))
    h = np.rot90(h,2)
    h = h*w
    
    return h

def gaussian2d_asymmetric(N, std_x, std_y):
    
    t = np.arange(-(N-1)/2,(N+1)/2)
    
    t1,t2 = np.meshgrid(t,t)
    
    std_x = np.double(std_x)
    std_y = np.double(std_y)
    
    w = np.exp(-0.5*(t1/std_x)**2)*np.exp(-0.5*(t2/std_y)**2) 
    
    return w
       
def kaiser2d(N, beta):
    
    t = np.arange(-(N-1)/2,(N+1)/2)/np.double(N-1)
    
    t1,t2 = np.meshgrid(t,t)
    t12 = np.sqrt(t1*t1+t2*t2)
    
    w1 = np.kaiser(N,beta)
    w = np.interp(t12,t,w1)
    w[t12>t[-1]] = 0
    w[t12<t[0]] = 0
    
    return w

def psf_BCCB(dim,h_1D,tc=1,mu=0):
    
    hn = np.size(h_1D) # length of the filter
    s = (hn-1)//2 # support = (-s,s)
    
    H = np.zeros([dim,dim])
    
    for ii in range(H.shape[0]):  # matrix computation
        l = max(ii - s, 0)
        u = min(ii + s, dim)
        l2 = max(l - ii + s, 0)
        u2 = min(u - ii + s, dim)
        H[ii , l : u] = h_1D[0,l2 : u2]
    
    return H

def check_gains(psf, Nfft, fcut):
    
    MTF = abs(np.fft.fft2(psf, s = [Nfft,Nfft]))
    MTF = MTF/np.amax(MTF)
    
    N_Nyq = (Nfft)*fcut/2
    
    MTF_el_PSF = MTF[0,:]
    GNyq_el_PSF = MTF_el_PSF[round(N_Nyq)]
    
    MTF_az_PSF = MTF[:,0]
    GNyq_az_PSF = MTF_az_PSF[round(N_Nyq)]
    
    print('Nyquist frequency gains: [GNyq_el = {:.2f}, GNyq_az = {:.2f}]'.format(GNyq_el_PSF, GNyq_az_PSF))
    
    return

def plot_psf(psf, Nfft):
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    X = np.arange(-((Nfft-1)//2+1), (Nfft-1)//2, 1)
    Y = np.arange(-((Nfft-1)//2+1), (Nfft-1)//2, 1)
    
    X, Y = np.meshgrid(X, Y)
    
    Z = psf
    
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    
    return