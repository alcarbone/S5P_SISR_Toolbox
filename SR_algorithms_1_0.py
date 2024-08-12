# -*- coding: utf-8 -*-
"""
Copyright (c) 
All rights reserved. This work should only be used for nonprofit purposes.

@author: 
    Alessia Carbone (alcarbone@unisa.it)
"""

"""
 Description:
           Super-resolution toolbox. 
           
           GT and LR images are super-resolved via interpolation, 
           deconvolution and neural networks.
           
           For interpolation we have available:
               * nearest-neighbour
               * linear
               * quadratic
               * cubic0.5 (A2 = -1/2), cubic0.75 (A2 = -3/4), cubic1 (A2 = -1)
               * lanczos1, lanczos2, lanczos3
               * 23tap
           For deconvolution we have:
               * CGA solved with the same filter used for degradation
               * CGA solved with a different filter
           For neural networks we have:
               SOTA:
                   * SRCNN (original weights)
                   * FSRCNN (original weights)
                   * VDSR (original weights)
                   * ESPCN (original weights)
                   * EDSR (original weights)
                   * RCAN (original weights)
                   * RankSRGAN (original weights)
                   * PAN (original weights)
                   * BSRN (original weights)
                   * HAT (original weights)
               Original:
                   * S5Net trained with the same filter used for degradation 
                   * S5Net trained with a different filter 
               Ablation study:
                   * S5Net without transposed convolution trained with the same filter used for degradation
               Pixel-shuffle:
                   * S5Net without transposed convolution trained with the same filter used for degradation with ESPCN (original weights) in input
               
           If the protocol is RR (Reduced Resolution) the indices used are
           Q2n, Q, ERGAS, SAM, sCC, RMSE and PSNR. 
           Q2n and SAM are always zero for monocromatic images.
           If the protocol is FR (Full Resolution) the index used is BRISQUE.
"""

import os
import csv
import scipy
import torch

import scipy.io as sio
import numpy as np

import matlab.engine
eng = matlab.engine.start_matlab()

from scripts.interp.interp import interp
from scripts.interp.kernel import kernel
from scripts.deconv.deconv import deconv
from scripts.IQA1.indexes_evaluation_SR import indexes_evaluation_SR

from scripts.SOTA.SRCNN.SRCNN_arch import SRCNN
from scripts.SOTA.FSRCNN.FSRCNN_arch import FSRCNN
from scripts.SOTA.VDSR.VDSR_arch import VDSR
from scripts.SOTA.ESPCN.ESPCN_arch import ESPCN
from scripts.SOTA.EDSR.EDSR_arch import EDSR
from scripts.SOTA.RCAN.RCAN_arch import RCAN
from scripts.SOTA.RankSRGAN.RankSRGAN_arch import SRResNet
from scripts.SOTA.PAN.PAN_arch import PAN
from scripts.SOTA.BSRN.BSRN_arch import BSRN
from scripts.SOTA.HAT.HAT_arch import HAT

from scripts.S5Net.S5Net_arch import S5Net

#%%
'''Interpolation algorithms'''

interp_type = ['nearest','linear','quadratic','cubic0.5','cubic0.75','cubic1','lanczos1',
               'lanczos2','lanczos3','23tap']

if protocol == 'RR':
    Q2n_SR_int = np.zeros(len(interp_type))
    Q_SR_int = np.zeros(len(interp_type))
    ERGAS_SR_int = np.zeros(len(interp_type))
    SAM_SR_int = np.zeros(len(interp_type))
    sCC_SR_int = np.zeros(len(interp_type))
    RMSE_SR_int = np.zeros(len(interp_type))
    PSNR_SR_int = np.zeros(len(interp_type))
else:
    BRISQUE_SR_int = np.zeros(len(interp_type))
    
for i in range(len(interp_type)):

    radiance_SR_int = interp(radiance_LR, ratio, interp_type[i])
              
    if results:
        dirs_res = dirs_res_path + '/{}'.format(interp_type[i])
        if not os.path.isdir(dirs_res):
            os.makedirs(dirs_res)
        
        radiance_SR_int = np.squeeze(radiance_SR_int)
    
    if protocol == 'RR':
        Q2n_SR_int[i], Q_SR_int[i], ERGAS_SR_int[i], SAM_SR_int[i], sCC_SR, RMSE_SR_int[i], PSNR_SR_int[i] = \
            indexes_evaluation_SR(radiance_SR_int,radiance_GT,ratio,radiance_range,Qblocks_size,flag_cut_bounds,dim_cut,th_values)
        
        sCC_SR_int[i] = sCC_SR[0]
    else:
        min_radiance = np.amin(radiance_SR_int) 
        max_radiance = np.amax(radiance_SR_int) 
        scaled_SR = (radiance_SR_int - min_radiance) / max_radiance
        
        scaled_SR = scaled_SR.squeeze()
        BRISQUE_SR_int[i] = eng.brisque(scaled_SR)
    
    if results:
        sio.savemat(dirs_res + '/' + im_tag + '_SR.mat', {'radiance': radiance_SR_int})   

#%% 
"""Deconvolution algorithms"""
l = 0.1
m = 0.00005
    
delta = 10**(-4)
iters = 200

#%%
'''CGA same filter'''

if protocol == 'RR':
    radiance_SR_CGA_match, it, S1, S2 = deconv(radiance_LR, radiance_GT, ratio, GNyq_x, GNyq_y, 'cga', iters=iters, delta=delta, m=m, l=l) 
else:
    radiance_SR_CGA_match, it, S1, S2 = deconv(radiance_LR, np.zeros([radiance_LR.shape[0]*ratio,radiance_LR.shape[1]*ratio]), ratio, GNyq_x, GNyq_y, 'cga', iters=iters, delta=delta, m=m, l=l) 

if results:
    dirs_res = dirs_res_path + '/cga_match'
    if not os.path.isdir(dirs_res):
        os.makedirs(dirs_res)
    
    radiance_SR_CGA_match = np.squeeze(radiance_SR_CGA_match)
  
    sio.savemat(dirs_res + '/' + im_tag + '_SR.mat', {'radiance': radiance_SR_CGA_match})

if protocol == 'RR': 
    Q2n_SR_CGA_match, Q_SR_CGA_match, ERGAS_SR_CGA_match, SAM_SR_CGA_match, sCC_SR, RMSE_SR_CGA_match, PSNR_SR_CGA_match = \
        indexes_evaluation_SR(radiance_SR_CGA_match,radiance_GT,ratio,radiance_range,Qblocks_size,flag_cut_bounds,dim_cut,th_values) 
        
    sCC_SR_CGA_match = sCC_SR[0]
    
else:
    min_radiance = np.amin(radiance_SR_CGA_match) 
    max_radiance = np.amax(radiance_SR_CGA_match) 
    scaled_SR = (radiance_SR_CGA_match - min_radiance) / max_radiance
    
    scaled_SR = scaled_SR.squeeze()
    BRISQUE_SR_CGA_match = eng.brisque(scaled_SR)

#%%
'''CGA different filter'''

G_x = 0.3
G_y = 0.3

if(len(radiance_GT.shape) == 2):
    GNyq_x = np.asarray(G_x * np.ones(1))
    GNyq_y = np.asarray(G_y * np.ones(1))
else:
    GNyq_x = np.asarray(G_x * np.ones(radiance_GT.shape[2]))
    GNyq_y = np.asarray(G_y * np.ones(radiance_GT.shape[2]))

if protocol == 'RR':
    radiance_SR_CGA_nomatch, it, S1, S2 = deconv(radiance_LR, radiance_GT, ratio, GNyq_x, GNyq_y, 'cga', iters=iters, delta=delta, m=m, l=l) 
else:
    radiance_SR_CGA_nomatch, it, S1, S2 = deconv(radiance_LR, np.zeros([radiance_LR.shape[0]*ratio,radiance_LR.shape[1]*ratio]), ratio, GNyq_x, GNyq_y, 'cga', iters=iters, delta=delta, m=m, l=l) 

if results:    
    dirs_res = dirs_res_path + '/cga_nomatch'
    if not os.path.isdir(dirs_res):
        os.makedirs(dirs_res)
    
    radiance_SR_CGA_nomatch = np.squeeze(radiance_SR_CGA_nomatch)

    sio.savemat(dirs_res + '/' + im_tag + '_SR.mat', {'radiance': radiance_SR_CGA_nomatch})

if protocol == 'RR':    
    Q2n_SR_CGA_nomatch, Q_SR_CGA_nomatch, ERGAS_SR_CGA_nomatch, SAM_SR_CGA_nomatch, sCC_SR, RMSE_SR_CGA_nomatch, PSNR_SR_CGA_nomatch = \
        indexes_evaluation_SR(radiance_SR_CGA_nomatch,radiance_GT,ratio,radiance_range,Qblocks_size,flag_cut_bounds,dim_cut,th_values)

    sCC_SR_CGA_nomatch = sCC_SR[0]
    
else:
    min_radiance = np.amin(radiance_SR_CGA_nomatch) 
    max_radiance = np.amax(radiance_SR_CGA_nomatch) 
    scaled_SR = (radiance_SR_CGA_nomatch - min_radiance) / max_radiance
    
    scaled_SR = scaled_SR.squeeze()
    BRISQUE_SR_CGA_nomatch = eng.brisque(scaled_SR)

#%%
'''Neural Networks'''

import torch

if protocol == 'RR':
    torch.set_default_tensor_type(torch.DoubleTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

radiance_LR = (radiance_LR / max_val) * 0.5

#%%
'''SRCNN (2014)'''

f1 = 9;f2 = 5;f3 = 5;c = 1 # best model in the original paper

I_LR = interp(radiance_LR, ratio, 'cubic')
I_LR = np.transpose(I_LR, (2, 0, 1))

lr_test = []
lr_test.append(I_LR[:,:,:])
lr_test = np.array(lr_test)

preds = np.zeros([1,lr_test.shape[1],lr_test.shape[2],lr_test.shape[3]])

epochs_path = './trained_models/SOTA/SRCNN/'

if(os.path.isfile(epochs_path + 'x{}.mat'.format(ratio))):
    
    mat_file = scipy.io.loadmat(epochs_path + 'x{}.mat'.format(ratio))
    
    n1 = 64;n2 = 32;n3 = n2;
    weights_conv1 = torch.tensor(mat_file['weights_conv1'], dtype=torch.float64)
    weights_conv1 = np.transpose(weights_conv1,(1,0))
    weights_conv1 = np.reshape(weights_conv1, (n1,c,f1*f1))
    weights_conv1 = np.reshape(weights_conv1, (n1,c,f1,f1), order = 'F')
    weights_conv2 = torch.tensor(mat_file['weights_conv2'], dtype=torch.float64)
    weights_conv2 = np.transpose(weights_conv2,(2,1,0))
    weights_conv2 = np.transpose(weights_conv2,(0,2,1))
    weights_conv2 = np.reshape(weights_conv2, (n2,n1,f2,f2), order = 'F')
    weights_conv3 = torch.tensor(mat_file['weights_conv3'], dtype=torch.float64)
    weights_conv3 = np.reshape(weights_conv3,(c,n3,f3*f3))
    weights_conv3 = np.reshape(weights_conv3,(c,n3,f3,f3), order = 'F')
    biases_conv1 = torch.tensor(mat_file['biases_conv1'], dtype=torch.float64)
    biases_conv1 = np.reshape(biases_conv1, (n1), order = 'F')
    biases_conv2 = torch.tensor(mat_file['biases_conv2'], dtype=torch.float64)
    biases_conv2 = np.reshape(biases_conv2, (n2), order = 'F')
    biases_conv3 = torch.tensor(mat_file['biases_conv3'], dtype=torch.float64)
    biases_conv3 = np.reshape(biases_conv3, (c), order = 'F')
    
    for i in range(lr_test.shape[1]):            
        lr_test1 = np.zeros([1,1,lr_test.shape[2],lr_test.shape[3]])
        lr_test1[0,0,:,:] = lr_test[0,i,:,:]
        if protocol == 'RR':
            lr_test1 = torch.tensor(lr_test1, dtype=torch.float64)
        else:
            lr_test1 = torch.tensor(lr_test1)
            lr_test1 = lr_test1.float()
            
        if torch.cuda.is_available():
            lr_test1 = lr_test1.to(device)
    
        model = SRCNN(n1,n2,n3,f1,f2,f3,c,weights_conv1,weights_conv2,weights_conv3,biases_conv1,biases_conv2,biases_conv3).to(device)
        model.eval()
        
        preds1 = model(lr_test1)
        
        preds1 = preds1.cpu()
        preds1 = preds1.detach().numpy()
        preds[0,i,:,:] = preds1
        del preds1
        del lr_test1
        torch.cuda.empty_cache()
    
    I_SR_test = np.transpose(preds, (0, 2, 3, 1))
    I_SR1_test = np.array([I_SR_test.shape[1],I_SR_test.shape[2],1])
    I_SR1_test = I_SR_test[0,:,:,:]
    I_SR_test = I_SR1_test            
    I_SR_SRCNN0 = I_SR_test.astype('float64')
    
    I_SR_SRCNN0 = (I_SR_SRCNN0/0.5) * max_val
    
    if protocol == 'RR':
        Q2n_SR_SRCNN0, Q_SR_SRCNN0, ERGAS_SR_SRCNN0, SAM_SR_SRCNN0, sCC_SR, RMSE_SR_SRCNN0, PSNR_SR_SRCNN0 = \
            indexes_evaluation_SR(I_SR_SRCNN0,radiance_GT,ratio,radiance_range,Qblocks_size,flag_cut_bounds,dim_cut,th_values)      
    
        sCC_SR_SRCNN0 = sCC_SR[0]
        
    else:
        min_radiance = np.amin(I_SR_SRCNN0) 
        max_radiance = np.amax(I_SR_SRCNN0) 
        scaled_SR = (I_SR_SRCNN0 - min_radiance) / max_radiance
        
        scaled_SR = scaled_SR.squeeze()
        BRISQUE_SR_SRCNN0 = eng.brisque(scaled_SR)

    if results:  
        dirs_res = dirs_res_path + '/SRCNN'
        if not os.path.isdir(dirs_res):
            os.makedirs(dirs_res)
        
        I_SR_SRCNN0 = np.squeeze(I_SR_SRCNN0)
        
        sio.savemat(dirs_res + '/' + im_tag + '_SR.mat', {'radiance': I_SR_SRCNN0})
    
#%%
"""FSRCNN (2016)"""

model = FSRCNN(scale_factor=ratio).to(device)

I_LR = radiance_LR
I_LR1 = np.zeros([I_LR.shape[0],I_LR.shape[1],1])
I_LR1[:,:,0] = I_LR
I_LR = I_LR1
I_LR = np.transpose(I_LR, (2, 0, 1))

lr_test = []
lr_test.append(I_LR[:,:,:])
lr_test = np.array(lr_test)

trained_model_path = './trained_models/SOTA/FSRCNN/x{}.pth'.format(ratio) 

if(os.path.isfile(trained_model_path)):          
    if protocol == 'RR':
        lr_test = torch.tensor(lr_test, dtype=torch.float64)
    else:
        lr_test = torch.tensor(lr_test)
        lr_test = lr_test.float()
            
    if torch.cuda.is_available():
        lr_test = lr_test.to(device)

    pretrained_model = torch.load(trained_model_path)
    
    with torch.no_grad():
        model.load_state_dict(pretrained_model)
        model.eval()
        
        preds = model(lr_test)
    
    preds = preds.cpu()
    preds = preds.detach().numpy()
    I_SR_test = np.transpose(preds, (0, 2, 3, 1))
    del preds 
    del lr_test
    torch.cuda.empty_cache()
    I_SR1_test = np.zeros([I_SR_test.shape[1],I_SR_test.shape[2],1])
    I_SR1_test[:,:,0] = I_SR_test[0,:,:,0]
    I_SR_test = I_SR1_test            
    I_SR_FSRCNN = I_SR_test.astype('float64')
    
    I_SR_FSRCNN = (I_SR_FSRCNN/0.5) * max_val
    
    if protocol == 'RR':
        Q2n_SR_FSRCNN, Q_SR_FSRCNN, ERGAS_SR_FSRCNN, SAM_SR_FSRCNN, sCC_SR, RMSE_SR_FSRCNN, PSNR_SR_FSRCNN = \
            indexes_evaluation_SR(I_SR_FSRCNN,radiance_GT,ratio,radiance_range,Qblocks_size,flag_cut_bounds,dim_cut,th_values)      
        
        sCC_SR_FSRCNN = sCC_SR[0]
        
    else:
        min_radiance = np.amin(I_SR_FSRCNN) 
        max_radiance = np.amax(I_SR_FSRCNN) 
        scaled_SR = (I_SR_FSRCNN - min_radiance) / max_radiance
        
        scaled_SR = scaled_SR.squeeze()
        BRISQUE_SR_FSRCNN = eng.brisque(scaled_SR)

    if results:  
        dirs_res = dirs_res_path + '/FSRCNN'
        if not os.path.isdir(dirs_res):
            os.makedirs(dirs_res)
        
        I_SR_FSRCNN = np.squeeze(I_SR_FSRCNN)
        
        sio.savemat(dirs_res + '/' + im_tag + '_SR.mat', {'radiance': I_SR_FSRCNN})
    
#%%
"""VDSR (2016)"""

model = VDSR().to(device)

I_LR = interp(radiance_LR, ratio, 'cubic')
I_LR = np.transpose(I_LR, (2, 0, 1))

lr_test = []
lr_test.append(I_LR[:,:,:])
lr_test = np.array(lr_test)

trained_model_path = './trained_models/SOTA/VDSR/VDSR.pth' 

if(os.path.isfile(trained_model_path)):          
    if protocol == 'RR':
        lr_test = torch.tensor(lr_test, dtype=torch.float64)
    else:
        lr_test = torch.tensor(lr_test)
        lr_test = lr_test.float()
            
    if torch.cuda.is_available():
        lr_test = lr_test.to(device)

    pretrained_model = torch.load(trained_model_path, map_location=device)['state_dict']
    
    with torch.no_grad():
        model.load_state_dict(pretrained_model)
        model.eval()
        
        preds = model(lr_test)
    
    preds = preds.cpu()
    preds = preds.detach().numpy()
    I_SR_test = np.transpose(preds, (0, 2, 3, 1))
    del preds 
    del lr_test
    torch.cuda.empty_cache()
    I_SR1_test = np.zeros([I_SR_test.shape[1],I_SR_test.shape[2],1])
    I_SR1_test[:,:,0] = I_SR_test[0,:,:,0]
    I_SR_test = I_SR1_test            
    I_SR_VDSR = I_SR_test.astype('float64')
    
    I_SR_VDSR = (I_SR_VDSR/0.5) * max_val
    
    if protocol == 'RR':
        Q2n_SR_VDSR, Q_SR_VDSR, ERGAS_SR_VDSR, SAM_SR_VDSR, sCC_SR, RMSE_SR_VDSR, PSNR_SR_VDSR = \
            indexes_evaluation_SR(I_SR_VDSR,radiance_GT,ratio,radiance_range,Qblocks_size,flag_cut_bounds,dim_cut,th_values)      
        
        sCC_SR_VDSR = sCC_SR[0]
        
    else:
        min_radiance = np.amin(I_SR_VDSR) 
        max_radiance = np.amax(I_SR_VDSR) 
        scaled_SR = (I_SR_VDSR - min_radiance) / max_radiance
        
        scaled_SR = scaled_SR.squeeze()
        BRISQUE_SR_VDSR = eng.brisque(scaled_SR)

    if results:  
        dirs_res = dirs_res_path + '/VDSR'
        if not os.path.isdir(dirs_res):
            os.makedirs(dirs_res)
        
        I_SR_VDSR = np.squeeze(I_SR_VDSR)
        
        sio.savemat(dirs_res + '/' + im_tag + '_SR.mat', {'radiance': I_SR_VDSR})
    
#%%
"""ESPCN (2016)"""

model = ESPCN(in_channels = 1, out_channels = 1, channels = 64, upscale_factor = ratio).to(device)

I_LR = radiance_LR
I_LR1 = np.zeros([I_LR.shape[0],I_LR.shape[1],1])
I_LR1[:,:,0] = I_LR
I_LR = I_LR1

I_LR = np.transpose(I_LR, (2, 0, 1))

lr_test = []
lr_test.append(I_LR[:,:,:])
lr_test = np.array(lr_test)

trained_model_path = './trained_models/SOTA/ESPCN/x{}.pth'.format(ratio) 

if(os.path.isfile(trained_model_path)):          
    if protocol == 'RR':
        lr_test = torch.tensor(lr_test, dtype=torch.float64)
    else:
        lr_test = torch.tensor(lr_test)
        lr_test = lr_test.float()
            
    if torch.cuda.is_available():
        lr_test = lr_test.to(device)

    pretrained_model = torch.load(trained_model_path, map_location=device)['state_dict']
    
    with torch.no_grad():
        model.load_state_dict(pretrained_model)
        model.eval()
        
        preds = model(lr_test)
    
    preds = preds.cpu()
    preds = preds.detach().numpy()
    I_SR_test = np.transpose(preds, (0, 2, 3, 1))
    del preds 
    del lr_test
    torch.cuda.empty_cache()
    I_SR1_test = np.zeros([I_SR_test.shape[1],I_SR_test.shape[2],1])
    I_SR1_test[:,:,0] = I_SR_test[0,:,:,0]
    I_SR_test = I_SR1_test            
    I_SR_ESPCN = I_SR_test.astype('float64')
    
    I_SR_ESPCN = (I_SR_ESPCN/0.5) * max_val
    
    if protocol == 'RR':
        Q2n_SR_ESPCN, Q_SR_ESPCN, ERGAS_SR_ESPCN, SAM_SR_ESPCN, sCC_SR, RMSE_SR_ESPCN, PSNR_SR_ESPCN = \
            indexes_evaluation_SR(I_SR_ESPCN,radiance_GT,ratio,radiance_range,Qblocks_size,flag_cut_bounds,dim_cut,th_values)      
        
        sCC_SR_ESPCN = sCC_SR[0]
        
    else:
        min_radiance = np.amin(I_SR_ESPCN) 
        max_radiance = np.amax(I_SR_ESPCN) 
        scaled_SR = (I_SR_ESPCN - min_radiance) / max_radiance
        
        scaled_SR = scaled_SR.squeeze()
        BRISQUE_SR_ESPCN = eng.brisque(scaled_SR)

    if results:  
        dirs_res = dirs_res_path + '/ESPCN'
        if not os.path.isdir(dirs_res):
            os.makedirs(dirs_res)
        
        I_SR_ESPCN = np.squeeze(I_SR_ESPCN)
        
        sio.savemat(dirs_res + '/' + im_tag + '_SR.mat', {'radiance': I_SR_ESPCN})

#%%
'''EDSR (2017)'''

I_LR = radiance_LR
I_LR1 = np.zeros([I_LR.shape[0],I_LR.shape[1],3])
I_LR1[:,:,0] = I_LR
I_LR1[:,:,1] = I_LR
I_LR1[:,:,2] = I_LR
I_LR = I_LR1

maximum = np.amax(I_LR)
I_LR = (I_LR/maximum)*255
I_LR = np.transpose(I_LR, (2, 0, 1))

lr_test = []
lr_test.append(I_LR[:,:,:])
lr_test = np.array(lr_test)

model = EDSR(n_colors=3,n_feats=256,n_resblocks=32,scale=ratio,res_scale=0.1,rgb_range=np.max(lr_test),rgb_mean=(np.mean(radiance_LR),np.mean(radiance_LR),np.mean(radiance_LR))).to(device)

trained_model_path = './trained_models/SOTA/EDSR/x{}.pt'.format(ratio) 

if(os.path.isfile(trained_model_path)):          
    if protocol == 'RR':
        lr_test = torch.tensor(lr_test, dtype=torch.float64)
    else:
        lr_test = torch.tensor(lr_test)
        lr_test = lr_test.float()
            
    if torch.cuda.is_available():
        lr_test = lr_test.to(device)
        
    with torch.no_grad():
        model.load_state_dict(torch.load(trained_model_path))
        model.eval()
        
        preds = model(lr_test)
    
    preds = preds.cpu()
    preds = preds.detach().numpy()
    I_SR_test = np.transpose(preds, (0, 2, 3, 1))
    del preds 
    del lr_test
    torch.cuda.empty_cache()
    I_SR1_test0 = np.zeros([I_SR_test.shape[1],I_SR_test.shape[2],1])
    I_SR1_test1 = np.zeros([I_SR_test.shape[1],I_SR_test.shape[2],1])
    I_SR1_test2 = np.zeros([I_SR_test.shape[1],I_SR_test.shape[2],1])
    I_SR1_test0[:,:,0] = I_SR_test[0,:,:,0] 
    I_SR1_test1[:,:,0] = I_SR_test[0,:,:,1] 
    I_SR1_test2[:,:,0] = I_SR_test[0,:,:,2]
    I_SR_test0 = I_SR1_test0 
    I_SR_test1 = I_SR1_test1 
    I_SR_test2 = I_SR1_test2         
    I_SR_EDSR0 = I_SR_test0.astype('float64')
    I_SR_EDSR1 = I_SR_test1.astype('float64')
    I_SR_EDSR2 = I_SR_test2.astype('float64')
    I_SR_EDSR0 = (I_SR_EDSR0/255) * maximum
    I_SR_EDSR1 = (I_SR_EDSR1/255) * maximum
    I_SR_EDSR2 = (I_SR_EDSR2/255) * maximum
    I_SR_EDSR0 = (I_SR_EDSR0/0.5) * max_val
    I_SR_EDSR1 = (I_SR_EDSR1/0.5) * max_val
    I_SR_EDSR2 = (I_SR_EDSR2/0.5) * max_val
    if protocol == 'RR':
        Q2n_SR_EDSR0, Q_SR_EDSR0, ERGAS_SR_EDSR0, SAM_SR_EDSR0, sCC_SR, RMSE_SR_EDSR0, PSNR_SR_EDSR0 = \
            indexes_evaluation_SR(I_SR_EDSR0,radiance_GT,ratio,radiance_range,Qblocks_size,flag_cut_bounds,dim_cut,th_values)      
        
        sCC_SR_EDSR0 = sCC_SR[0]
        
        Q2n_SR_EDSR1, Q_SR_EDSR1, ERGAS_SR_EDSR1, SAM_SR_EDSR1, sCC_SR, RMSE_SR_EDSR1, PSNR_SR_EDSR1 = \
            indexes_evaluation_SR(I_SR_EDSR1,radiance_GT,ratio,radiance_range,Qblocks_size,flag_cut_bounds,dim_cut,th_values)      
        
        sCC_SR_EDSR1 = sCC_SR[0]
        
        Q2n_SR_EDSR2, Q_SR_EDSR2, ERGAS_SR_EDSR2, SAM_SR_EDSR2, sCC_SR, RMSE_SR_EDSR2, PSNR_SR_EDSR2 = \
            indexes_evaluation_SR(I_SR_EDSR2,radiance_GT,ratio,radiance_range,Qblocks_size,flag_cut_bounds,dim_cut,th_values)      
        
        sCC_SR_EDSR2 = sCC_SR[0]
        
    else:
        min_radiance = np.amin(I_SR_EDSR0) 
        max_radiance = np.amax(I_SR_EDSR0) 
        scaled_SR = (I_SR_EDSR0 - min_radiance) / max_radiance
        
        scaled_SR = scaled_SR.squeeze()
        BRISQUE_SR_EDSR0 = eng.brisque(scaled_SR)
        
        min_radiance = np.amin(I_SR_EDSR1) 
        max_radiance = np.amax(I_SR_EDSR1) 
        scaled_SR = (I_SR_EDSR1 - min_radiance) / max_radiance
        
        scaled_SR = scaled_SR.squeeze()
        BRISQUE_SR_EDSR1 = eng.brisque(scaled_SR)
        
        min_radiance = np.amin(I_SR_EDSR2) 
        max_radiance = np.amax(I_SR_EDSR2) 
        scaled_SR = (I_SR_EDSR2 - min_radiance) / max_radiance
        
        scaled_SR = scaled_SR.squeeze()
        BRISQUE_SR_EDSR2 = eng.brisque(scaled_SR)

    if results:  
        dirs_res = dirs_res_path + '/EDSR0'
        if not os.path.isdir(dirs_res):
            os.makedirs(dirs_res)
        
        I_SR_EDSR0 = np.squeeze(I_SR_EDSR0)
        
        sio.savemat(dirs_res + '/' + im_tag + '_SR.mat', {'radiance': I_SR_EDSR0})
        
        dirs_res = dirs_res_path + '/EDSR1'
        if not os.path.isdir(dirs_res):
            os.makedirs(dirs_res)
        
        I_SR_EDSR1 = np.squeeze(I_SR_EDSR1)
        
        sio.savemat(dirs_res + '/' + im_tag + '_SR.mat', {'radiance': I_SR_EDSR1})
        
        dirs_res = dirs_res_path + '/EDSR2'
        if not os.path.isdir(dirs_res):
            os.makedirs(dirs_res)
        
        I_SR_EDSR2 = np.squeeze(I_SR_EDSR2)
        
        sio.savemat(dirs_res + '/' + im_tag + '_SR.mat', {'radiance': I_SR_EDSR2})

#%%
'''RCAN (2018)'''

I_LR = radiance_LR
I_LR1 = np.zeros([I_LR.shape[0],I_LR.shape[1],3])
I_LR1[:,:,0] = I_LR
I_LR1[:,:,1] = I_LR
I_LR1[:,:,2] = I_LR
I_LR = I_LR1

maximum = np.amax(I_LR)
I_LR = (I_LR/maximum)*255
I_LR = np.transpose(I_LR, (2, 0, 1))

lr_test = []
lr_test.append(I_LR[:,:,:])
lr_test = np.array(lr_test)

model = RCAN(n_colors=3,n_resblocks=16,n_feats=64,n_resgroups=10,reduction=16,scale=ratio,res_scale=0.1,rgb_range=np.max(lr_test),rgb_mean=(np.mean(radiance_LR),np.mean(radiance_LR),np.mean(radiance_LR))).to(device)

trained_model_path = './trained_models/SOTA/RCAN/x{}.pt'.format(ratio) 

if(os.path.isfile(trained_model_path)):          
    if protocol == 'RR':
        lr_test = torch.tensor(lr_test, dtype=torch.float64)
    else:
        lr_test = torch.tensor(lr_test)
        lr_test = lr_test.float()
            
    if torch.cuda.is_available():
        lr_test = lr_test.to(device)
        
    with torch.no_grad():
        model.load_state_dict(torch.load(trained_model_path))
        model.eval()
        
        preds = model(lr_test)
    
    preds = preds.cpu()
    preds = preds.detach().numpy()
    I_SR_test = np.transpose(preds, (0, 2, 3, 1))
    del preds 
    del lr_test
    torch.cuda.empty_cache()
    I_SR1_test0 = np.zeros([I_SR_test.shape[1],I_SR_test.shape[2],1])
    I_SR1_test1 = np.zeros([I_SR_test.shape[1],I_SR_test.shape[2],1])
    I_SR1_test2 = np.zeros([I_SR_test.shape[1],I_SR_test.shape[2],1])
    I_SR1_test0[:,:,0] = I_SR_test[0,:,:,0] 
    I_SR1_test1[:,:,0] = I_SR_test[0,:,:,1] 
    I_SR1_test2[:,:,0] = I_SR_test[0,:,:,2]
    I_SR_test0 = I_SR1_test0 
    I_SR_test1 = I_SR1_test1 
    I_SR_test2 = I_SR1_test2         
    I_SR_RCAN0 = I_SR_test0.astype('float64')
    I_SR_RCAN1 = I_SR_test1.astype('float64')
    I_SR_RCAN2 = I_SR_test2.astype('float64')
    I_SR_RCAN0 = (I_SR_RCAN0/255) * maximum
    I_SR_RCAN1 = (I_SR_RCAN1/255) * maximum
    I_SR_RCAN2 = (I_SR_RCAN2/255) * maximum
    I_SR_RCAN0 = (I_SR_RCAN0/0.5) * max_val
    I_SR_RCAN1 = (I_SR_RCAN1/0.5) * max_val
    I_SR_RCAN2 = (I_SR_RCAN2/0.5) * max_val
    
    if protocol == 'RR':
        Q2n_SR_RCAN0, Q_SR_RCAN0, ERGAS_SR_RCAN0, SAM_SR_RCAN0, sCC_SR, RMSE_SR_RCAN0, PSNR_SR_RCAN0 = \
            indexes_evaluation_SR(I_SR_RCAN0,radiance_GT,ratio,radiance_range,Qblocks_size,flag_cut_bounds,dim_cut,th_values)      
        
        sCC_SR_RCAN0 = sCC_SR[0]
        
        Q2n_SR_RCAN1, Q_SR_RCAN1, ERGAS_SR_RCAN1, SAM_SR_RCAN1, sCC_SR, RMSE_SR_RCAN1, PSNR_SR_RCAN1 = \
            indexes_evaluation_SR(I_SR_RCAN1,radiance_GT,ratio,radiance_range,Qblocks_size,flag_cut_bounds,dim_cut,th_values)      
        
        sCC_SR_RCAN1 = sCC_SR[0]
        
        Q2n_SR_RCAN2, Q_SR_RCAN2, ERGAS_SR_RCAN2, SAM_SR_RCAN2, sCC_SR, RMSE_SR_RCAN2, PSNR_SR_RCAN2 = \
            indexes_evaluation_SR(I_SR_RCAN2,radiance_GT,ratio,radiance_range,Qblocks_size,flag_cut_bounds,dim_cut,th_values)      
        
        sCC_SR_RCAN2 = sCC_SR[0]
        
    else:
        min_radiance = np.amin(I_SR_RCAN0) 
        max_radiance = np.amax(I_SR_RCAN0) 
        scaled_SR = (I_SR_RCAN0 - min_radiance) / max_radiance
        
        scaled_SR = scaled_SR.squeeze()
        BRISQUE_SR_RCAN0 = eng.brisque(scaled_SR)
        
        min_radiance = np.amin(I_SR_RCAN1) 
        max_radiance = np.amax(I_SR_RCAN1) 
        scaled_SR = (I_SR_RCAN1 - min_radiance) / max_radiance
        
        scaled_SR = scaled_SR.squeeze()
        BRISQUE_SR_RCAN1 = eng.brisque(scaled_SR)
        
        min_radiance = np.amin(I_SR_RCAN2) 
        max_radiance = np.amax(I_SR_RCAN2) 
        scaled_SR = (I_SR_RCAN2 - min_radiance) / max_radiance
        
        scaled_SR = scaled_SR.squeeze()
        BRISQUE_SR_RCAN2 = eng.brisque(scaled_SR)

    if results:  
        dirs_res = dirs_res_path + '/RCAN0'
        if not os.path.isdir(dirs_res):
            os.makedirs(dirs_res)
        
        I_SR_RCAN0 = np.squeeze(I_SR_RCAN0)
        
        sio.savemat(dirs_res + '/' + im_tag + '_SR.mat', {'radiance': I_SR_RCAN0})
        
        dirs_res = dirs_res_path + '/RCAN1'
        if not os.path.isdir(dirs_res):
            os.makedirs(dirs_res)
        
        I_SR_RCAN1 = np.squeeze(I_SR_RCAN1)
        
        sio.savemat(dirs_res + '/' + im_tag + '_SR.mat', {'radiance': I_SR_RCAN1})
        
        dirs_res = dirs_res_path + '/RCAN2'
        if not os.path.isdir(dirs_res):
            os.makedirs(dirs_res)
        
        I_SR_RCAN2 = np.squeeze(I_SR_RCAN2)
        
        sio.savemat(dirs_res + '/' + im_tag + '_SR.mat', {'radiance': I_SR_RCAN2})

#%%
'''RankSRGAN (2019)'''

model = SRResNet(upscale=ratio).to(device)

I_LR = radiance_LR
I_LR1 = np.zeros([I_LR.shape[0],I_LR.shape[1],3])
I_LR1[:,:,0] = I_LR
I_LR1[:,:,1] = I_LR
I_LR1[:,:,2] = I_LR
I_LR = I_LR1
I_LR = np.transpose(I_LR, (2, 0, 1))

lr_test = []
lr_test.append(I_LR[:,:,:])
lr_test = np.array(lr_test)

trained_model_path = './trained_models/SOTA/RankSRGAN/PI.pth' 
#trained_model_path = './trained_models/SOTA/RankSRGAN/Ma.pth'
#trained_model_path = './trained_models/SOTA/RankSRGAN/NIQE.pth'

if(os.path.isfile(trained_model_path)):          
    if protocol == 'RR':
        lr_test = torch.tensor(lr_test, dtype=torch.float64)
    else:
        lr_test = torch.tensor(lr_test)
        lr_test = lr_test.float()
            
    if torch.cuda.is_available():
        lr_test = lr_test.to(device)
        
    with torch.no_grad():
        if (ratio == 2 or ratio == 3):
            pretrained_models = torch.load(trained_model_path)
            for key in list(pretrained_models.keys()):
                if 'upconv2.weight' in key:
                    del pretrained_models[key]
                if 'upconv2.bias' in key:
                    del pretrained_models[key]
            
        model.load_state_dict(pretrained_models)
        model.eval()
        
        preds = model(lr_test)
    
    preds = preds.cpu()
    preds = preds.detach().numpy()
    I_SR_test = np.transpose(preds, (0, 2, 3, 1))
    del preds 
    del lr_test
    torch.cuda.empty_cache()
    I_SR1_test0 = np.zeros([I_SR_test.shape[1],I_SR_test.shape[2],1])
    I_SR1_test1 = np.zeros([I_SR_test.shape[1],I_SR_test.shape[2],1])
    I_SR1_test2 = np.zeros([I_SR_test.shape[1],I_SR_test.shape[2],1])
    I_SR1_test0[:,:,0] = I_SR_test[0,:,:,0]
    I_SR1_test1[:,:,0] = I_SR_test[0,:,:,1]
    I_SR1_test2[:,:,0] = I_SR_test[0,:,:,2]
    I_SR_test0 = I_SR1_test0 
    I_SR_test1 = I_SR1_test1
    I_SR_test2 = I_SR1_test2           
    I_SR_RankSRGAN0 = I_SR_test0.astype('float64')
    I_SR_RankSRGAN1 = I_SR_test1.astype('float64')
    I_SR_RankSRGAN2 = I_SR_test2.astype('float64')
    
    I_SR_RankSRGAN0 = (I_SR_RankSRGAN0/0.5) * max_val
    I_SR_RankSRGAN1 = (I_SR_RankSRGAN1/0.5) * max_val
    I_SR_RankSRGAN2 = (I_SR_RankSRGAN2/0.5) * max_val
    
    if protocol == 'RR':
        Q2n_SR_RankSRGAN0, Q_SR_RankSRGAN0, ERGAS_SR_RankSRGAN0, SAM_SR_RankSRGAN0, sCC_SR, RMSE_SR_RankSRGAN0, PSNR_SR_RankSRGAN0 = \
            indexes_evaluation_SR(I_SR_RankSRGAN0,radiance_GT,ratio,radiance_range,Qblocks_size,flag_cut_bounds,dim_cut,th_values)      
        
        sCC_SR_RankSRGAN0 = sCC_SR[0]
        
        Q2n_SR_RankSRGAN1, Q_SR_RankSRGAN1, ERGAS_SR_RankSRGAN1, SAM_SR_RankSRGAN1, sCC_SR, RMSE_SR_RankSRGAN1, PSNR_SR_RankSRGAN1 = \
            indexes_evaluation_SR(I_SR_RankSRGAN1,radiance_GT,ratio,radiance_range,Qblocks_size,flag_cut_bounds,dim_cut,th_values)      
        
        sCC_SR_RankSRGAN1 = sCC_SR[0]
        
        Q2n_SR_RankSRGAN2, Q_SR_RankSRGAN2, ERGAS_SR_RankSRGAN2, SAM_SR_RankSRGAN2, sCC_SR, RMSE_SR_RankSRGAN2, PSNR_SR_RankSRGAN2 = \
            indexes_evaluation_SR(I_SR_RankSRGAN2,radiance_GT,ratio,radiance_range,Qblocks_size,flag_cut_bounds,dim_cut,th_values)      
        
        sCC_SR_RankSRGAN2 = sCC_SR[0]
        
    else:
        min_radiance = np.amin(I_SR_RankSRGAN0) 
        max_radiance = np.amax(I_SR_RankSRGAN0) 
        scaled_SR = (I_SR_RankSRGAN0 - min_radiance) / max_radiance
        
        scaled_SR = scaled_SR.squeeze()
        BRISQUE_SR_RankSRGAN0 = eng.brisque(scaled_SR)
        
        min_radiance = np.amin(I_SR_RankSRGAN1) 
        max_radiance = np.amax(I_SR_RankSRGAN1) 
        scaled_SR = (I_SR_RankSRGAN1 - min_radiance) / max_radiance
        
        scaled_SR = scaled_SR.squeeze()
        BRISQUE_SR_RankSRGAN1 = eng.brisque(scaled_SR)
        
        min_radiance = np.amin(I_SR_RankSRGAN2) 
        max_radiance = np.amax(I_SR_RankSRGAN2) 
        scaled_SR = (I_SR_RankSRGAN2 - min_radiance) / max_radiance
        
        scaled_SR = scaled_SR.squeeze()
        BRISQUE_SR_RankSRGAN2 = eng.brisque(scaled_SR)

    if results:  
        dirs_res = dirs_res_path + '/RankSRGAN0'
        if not os.path.isdir(dirs_res):
            os.makedirs(dirs_res)
        
        I_SR_RankSRGAN0 = np.squeeze(I_SR_RankSRGAN0)
        
        sio.savemat(dirs_res + '/' + im_tag + '_SR.mat', {'radiance': I_SR_RankSRGAN0})
        
        dirs_res = dirs_res_path + '/RankSRGAN1'
        if not os.path.isdir(dirs_res):
            os.makedirs(dirs_res)
        
        I_SR_RankSRGAN1 = np.squeeze(I_SR_RankSRGAN1)
        
        sio.savemat(dirs_res + '/' + im_tag + '_SR.mat', {'radiance': I_SR_RankSRGAN1})
        
        dirs_res = dirs_res_path + '/RankSRGAN2'
        if not os.path.isdir(dirs_res):
            os.makedirs(dirs_res)
        
        I_SR_RankSRGAN2 = np.squeeze(I_SR_RankSRGAN2)
        
        sio.savemat(dirs_res + '/' + im_tag + '_SR.mat', {'radiance': I_SR_RankSRGAN2})

#%%
'''PAN (2020)'''

model = PAN(in_nc=3, out_nc=3, nf=40, unf=24, nb=16, scale=ratio).to(device)

I_LR = radiance_LR
I_LR1 = np.zeros([I_LR.shape[0],I_LR.shape[1],3])
I_LR1[:,:,0] = I_LR
I_LR1[:,:,1] = I_LR
I_LR1[:,:,2] = I_LR
I_LR = I_LR1
I_LR = np.transpose(I_LR, (2, 0, 1))

lr_test = []
lr_test.append(I_LR[:,:,:])
lr_test = np.array(lr_test)

trained_model_path = './trained_models/SOTA/PAN/x{}.pth'.format(ratio) 

if(os.path.isfile(trained_model_path)):          
    if protocol == 'RR':
        lr_test = torch.tensor(lr_test, dtype=torch.float64)
    else:
        lr_test = torch.tensor(lr_test)
        lr_test = lr_test.float()
            
    if torch.cuda.is_available():
        lr_test = lr_test.to(device)
        
    with torch.no_grad():
        model.load_state_dict(torch.load(trained_model_path))
        model.eval()
        
        preds = model(lr_test)
    
    preds = preds.cpu()
    preds = preds.detach().numpy()
    I_SR_test = np.transpose(preds, (0, 2, 3, 1))
    del preds 
    del lr_test
    torch.cuda.empty_cache()
    I_SR1_test0 = np.zeros([I_SR_test.shape[1],I_SR_test.shape[2],1])
    I_SR1_test1 = np.zeros([I_SR_test.shape[1],I_SR_test.shape[2],1])
    I_SR1_test2 = np.zeros([I_SR_test.shape[1],I_SR_test.shape[2],1])
    I_SR1_test0[:,:,0] = I_SR_test[0,:,:,0] 
    I_SR1_test1[:,:,0] = I_SR_test[0,:,:,1] 
    I_SR1_test2[:,:,0] = I_SR_test[0,:,:,2]
    I_SR_test0 = I_SR1_test0 
    I_SR_test1 = I_SR1_test1 
    I_SR_test2 = I_SR1_test2         
    I_SR_PAN0 = I_SR_test0.astype('float64')
    I_SR_PAN1 = I_SR_test1.astype('float64')
    I_SR_PAN2 = I_SR_test2.astype('float64')
    
    I_SR_PAN0 = (I_SR_PAN0/0.5) * max_val
    I_SR_PAN1 = (I_SR_PAN1/0.5) * max_val
    I_SR_PAN2 = (I_SR_PAN2/0.5) * max_val
    
    if protocol == 'RR':
        Q2n_SR_PAN0, Q_SR_PAN0, ERGAS_SR_PAN0, SAM_SR_PAN0, sCC_SR, RMSE_SR_PAN0, PSNR_SR_PAN0 = \
            indexes_evaluation_SR(I_SR_PAN0,radiance_GT,ratio,radiance_range,Qblocks_size,flag_cut_bounds,dim_cut,th_values)      
        
        sCC_SR_PAN0 = sCC_SR[0]
        
        Q2n_SR_PAN1, Q_SR_PAN1, ERGAS_SR_PAN1, SAM_SR_PAN1, sCC_SR, RMSE_SR_PAN1, PSNR_SR_PAN1 = \
            indexes_evaluation_SR(I_SR_PAN1,radiance_GT,ratio,radiance_range,Qblocks_size,flag_cut_bounds,dim_cut,th_values)      
        
        sCC_SR_PAN1 = sCC_SR[0]
        
        Q2n_SR_PAN2, Q_SR_PAN2, ERGAS_SR_PAN2, SAM_SR_PAN2, sCC_SR, RMSE_SR_PAN2, PSNR_SR_PAN2 = \
            indexes_evaluation_SR(I_SR_PAN2,radiance_GT,ratio,radiance_range,Qblocks_size,flag_cut_bounds,dim_cut,th_values)      
        
        sCC_SR_PAN2 = sCC_SR[0]
        
    else:
        min_radiance = np.amin(I_SR_PAN0) 
        max_radiance = np.amax(I_SR_PAN0) 
        scaled_SR = (I_SR_PAN0 - min_radiance) / max_radiance
        
        scaled_SR = scaled_SR.squeeze()
        BRISQUE_SR_PAN0 = eng.brisque(scaled_SR)
        
        min_radiance = np.amin(I_SR_PAN1) 
        max_radiance = np.amax(I_SR_PAN1) 
        scaled_SR = (I_SR_PAN1 - min_radiance) / max_radiance
        
        scaled_SR = scaled_SR.squeeze()
        BRISQUE_SR_PAN1 = eng.brisque(scaled_SR)
        
        min_radiance = np.amin(I_SR_PAN2) 
        max_radiance = np.amax(I_SR_PAN2) 
        scaled_SR = (I_SR_PAN2 - min_radiance) / max_radiance
        
        scaled_SR = scaled_SR.squeeze()
        BRISQUE_SR_PAN2 = eng.brisque(scaled_SR)

    if results:  
        dirs_res = dirs_res_path + '/PAN0'
        if not os.path.isdir(dirs_res):
            os.makedirs(dirs_res)
        
        I_SR_PAN0 = np.squeeze(I_SR_PAN0)
        
        sio.savemat(dirs_res + '/' + im_tag + '_SR.mat', {'radiance': I_SR_PAN0})
        
        dirs_res = dirs_res_path + '/PAN1'
        if not os.path.isdir(dirs_res):
            os.makedirs(dirs_res)
        
        I_SR_PAN1 = np.squeeze(I_SR_PAN1)
        
        sio.savemat(dirs_res + '/' + im_tag + '_SR.mat', {'radiance': I_SR_PAN1})
        
        dirs_res = dirs_res_path + '/PAN2'
        if not os.path.isdir(dirs_res):
            os.makedirs(dirs_res)
        
        I_SR_PAN2 = np.squeeze(I_SR_PAN2)
        
        sio.savemat(dirs_res + '/' + im_tag + '_SR.mat', {'radiance': I_SR_PAN2})
 
#%%
'''BSRN (2022)'''

model = BSRN(upscale=ratio).to(device)

I_LR = radiance_LR
I_LR1 = np.zeros([I_LR.shape[0],I_LR.shape[1],3])
I_LR1[:,:,0] = I_LR
I_LR1[:,:,1] = I_LR
I_LR1[:,:,2] = I_LR
I_LR = I_LR1
I_LR = np.transpose(I_LR, (2, 0, 1))

lr_test = []
lr_test.append(I_LR[:,:,:])
lr_test = np.array(lr_test)

trained_model_path = './trained_models/SOTA/BSRN/x{}.pth'.format(ratio) 

if(os.path.isfile(trained_model_path)):          
    if protocol == 'RR':
        lr_test = torch.tensor(lr_test, dtype=torch.float64)
    else:
        lr_test = torch.tensor(lr_test)
        lr_test = lr_test.float()
            
    if torch.cuda.is_available():
        lr_test = lr_test.to(device)
        
    with torch.no_grad():
        pretrained_models = torch.load(trained_model_path)['params']
        
        model.load_state_dict(pretrained_models)
        model.eval()
        
        preds = model(lr_test)
    
    preds = preds.cpu()
    preds = preds.detach().numpy()
    I_SR_test = np.transpose(preds, (0, 2, 3, 1))
    del preds 
    del lr_test
    torch.cuda.empty_cache()
    I_SR1_test0 = np.zeros([I_SR_test.shape[1],I_SR_test.shape[2],1])
    I_SR1_test1 = np.zeros([I_SR_test.shape[1],I_SR_test.shape[2],1])
    I_SR1_test2 = np.zeros([I_SR_test.shape[1],I_SR_test.shape[2],1])
    I_SR1_test0[:,:,0] = I_SR_test[0,:,:,0] 
    I_SR1_test1[:,:,0] = I_SR_test[0,:,:,1] 
    I_SR1_test2[:,:,0] = I_SR_test[0,:,:,2]
    I_SR_test0 = I_SR1_test0 
    I_SR_test1 = I_SR1_test1 
    I_SR_test2 = I_SR1_test2         
    I_SR_BSRN0 = I_SR_test0.astype('float64')
    I_SR_BSRN1 = I_SR_test1.astype('float64')
    I_SR_BSRN2 = I_SR_test2.astype('float64')
    
    I_SR_BSRN0 = (I_SR_BSRN0/0.5) * max_val
    I_SR_BSRN1 = (I_SR_BSRN1/0.5) * max_val
    I_SR_BSRN2 = (I_SR_BSRN2/0.5) * max_val
    
    if protocol == 'RR':
        Q2n_SR_BSRN0, Q_SR_BSRN0, ERGAS_SR_BSRN0, SAM_SR_BSRN0, sCC_SR, RMSE_SR_BSRN0, PSNR_SR_BSRN0 = \
            indexes_evaluation_SR(I_SR_BSRN0,radiance_GT,ratio,radiance_range,Qblocks_size,flag_cut_bounds,dim_cut,th_values)      
        
        sCC_SR_BSRN0 = sCC_SR[0]
        
        Q2n_SR_BSRN1, Q_SR_BSRN1, ERGAS_SR_BSRN1, SAM_SR_BSRN1, sCC_SR, RMSE_SR_BSRN1, PSNR_SR_BSRN1 = \
            indexes_evaluation_SR(I_SR_BSRN1,radiance_GT,ratio,radiance_range,Qblocks_size,flag_cut_bounds,dim_cut,th_values)      
        
        sCC_SR_BSRN1 = sCC_SR[0]
        
        Q2n_SR_BSRN2, Q_SR_BSRN2, ERGAS_SR_BSRN2, SAM_SR_BSRN2, sCC_SR, RMSE_SR_BSRN2, PSNR_SR_BSRN2 = \
            indexes_evaluation_SR(I_SR_BSRN2,radiance_GT,ratio,radiance_range,Qblocks_size,flag_cut_bounds,dim_cut,th_values)      
        
        sCC_SR_BSRN2 = sCC_SR[0]
        
    else:
        min_radiance = np.amin(I_SR_BSRN0) 
        max_radiance = np.amax(I_SR_BSRN0) 
        scaled_SR = (I_SR_BSRN0 - min_radiance) / max_radiance
        
        scaled_SR = scaled_SR.squeeze()
        BRISQUE_SR_BSRN0 = eng.brisque(scaled_SR)
        
        min_radiance = np.amin(I_SR_BSRN1) 
        max_radiance = np.amax(I_SR_BSRN1) 
        scaled_SR = (I_SR_BSRN1 - min_radiance) / max_radiance
        
        scaled_SR = scaled_SR.squeeze()
        BRISQUE_SR_BSRN1 = eng.brisque(scaled_SR)
        
        min_radiance = np.amin(I_SR_BSRN2) 
        max_radiance = np.amax(I_SR_BSRN2) 
        scaled_SR = (I_SR_BSRN2 - min_radiance) / max_radiance
        
        scaled_SR = scaled_SR.squeeze()
        BRISQUE_SR_BSRN2 = eng.brisque(scaled_SR)

    if results:  
        dirs_res = dirs_res_path + '/BSRN0'
        if not os.path.isdir(dirs_res):
            os.makedirs(dirs_res)
        
        I_SR_BSRN0 = np.squeeze(I_SR_BSRN0)
        
        sio.savemat(dirs_res + '/' + im_tag + '_SR.mat', {'radiance': I_SR_BSRN0})
        
        dirs_res = dirs_res_path + '/BSRN1'
        if not os.path.isdir(dirs_res):
            os.makedirs(dirs_res)
        
        I_SR_BSRN1 = np.squeeze(I_SR_BSRN1)
        
        sio.savemat(dirs_res + '/' + im_tag + '_SR.mat', {'radiance': I_SR_BSRN1})
        
        dirs_res = dirs_res_path + '/BSRN2'
        if not os.path.isdir(dirs_res):
            os.makedirs(dirs_res)
        
        I_SR_BSRN2 = np.squeeze(I_SR_BSRN2)
        
        sio.savemat(dirs_res + '/' + im_tag + '_SR.mat', {'radiance': I_SR_BSRN2})

#%%
'''HAT (2023)'''

if band == 'SWIR':
    shape = radiance_LR.shape[1]
    new_shape = 2**(shape - 1).bit_length()
    diff = (new_shape-shape)
    if (diff%2 == 1):
        pad = (diff+1)//2
    else:
        pad = diff//2
    I_LR = np.pad(radiance_LR,((0,0),(pad,pad)),mode='symmetric')
    I_LR = I_LR[:,0:new_shape]
else:
    I_LR = radiance_LR
    
I_LR1 = np.zeros([I_LR.shape[0],I_LR.shape[1],3])
I_LR1[:,:,0] = I_LR
I_LR1[:,:,1] = I_LR
I_LR1[:,:,2] = I_LR
I_LR = I_LR1

maximum = np.amax(I_LR)
I_LR = (I_LR/maximum)*255
I_LR = np.transpose(I_LR, (2, 0, 1))

lr_test = []
lr_test.append(I_LR[:,:,:])
lr_test = np.array(lr_test)

model = HAT(embed_dim=180,depths=(6, 6, 6, 6, 6, 6),num_heads=(6, 6, 6, 6, 6, 6),window_size=16,mlp_ratio=2.,upscale=ratio,img_range=np.max(lr_test),upsampler='pixelshuffle',resi_connection='1conv',rgb_mean=(np.mean(radiance_LR),np.mean(radiance_LR),np.mean(radiance_LR))).to(device)

trained_model_path = './trained_models/SOTA/HAT/x{}.pth'.format(ratio) 

if(os.path.isfile(trained_model_path)):          
    if protocol == 'RR':
        lr_test = torch.tensor(lr_test, dtype=torch.float64)
    else:
        lr_test = torch.tensor(lr_test)
        lr_test = lr_test.float()
            
    if torch.cuda.is_available():
        lr_test = lr_test.to(device)
        
    with torch.no_grad():
        pretrained_models = torch.load(trained_model_path)['params_ema']
        
        model.load_state_dict(pretrained_models)
        model.eval()
        
        preds = model(lr_test)
    
    preds = preds.cpu()
    preds = preds.detach().numpy()
    I_SR_test = np.transpose(preds, (0, 2, 3, 1))
    del preds 
    del lr_test
    torch.cuda.empty_cache()
    shape = I_SR_test.shape[2]
    new_shape = radiance_GT.shape[1]
    diff = (shape-new_shape)
    if (diff%2 == 1):
        pad_sx = (diff//2)
        pad_dx = pad_sx + 1
    else:
        pad_sx = (diff//2)
        pad_dx = pad_sx 
    I_SR_test = I_SR_test[:,:,pad_sx:(shape-pad_dx),:]
    I_SR1_test0 = np.zeros([I_SR_test.shape[1],I_SR_test.shape[2],1])
    I_SR1_test1 = np.zeros([I_SR_test.shape[1],I_SR_test.shape[2],1])
    I_SR1_test2 = np.zeros([I_SR_test.shape[1],I_SR_test.shape[2],1])
    I_SR1_test0[:,:,0] = I_SR_test[0,:,:,0] 
    I_SR1_test1[:,:,0] = I_SR_test[0,:,:,1] 
    I_SR1_test2[:,:,0] = I_SR_test[0,:,:,2]
    I_SR_test0 = I_SR1_test0 
    I_SR_test1 = I_SR1_test1 
    I_SR_test2 = I_SR1_test2         
    I_SR_HAT0 = I_SR_test0.astype('float64')
    I_SR_HAT1 = I_SR_test1.astype('float64')
    I_SR_HAT2 = I_SR_test2.astype('float64')
    I_SR_HAT0 = (I_SR_HAT0/255) * maximum
    I_SR_HAT1 = (I_SR_HAT1/255) * maximum
    I_SR_HAT2 = (I_SR_HAT2/255) * maximum
    I_SR_HAT0 = (I_SR_HAT0/0.5) * max_val
    I_SR_HAT1 = (I_SR_HAT1/0.5) * max_val
    I_SR_HAT2 = (I_SR_HAT2/0.5) * max_val
    
    if protocol == 'RR':
        Q2n_SR_HAT0, Q_SR_HAT0, ERGAS_SR_HAT0, SAM_SR_HAT0, sCC_SR, RMSE_SR_HAT0, PSNR_SR_HAT0 = \
            indexes_evaluation_SR(I_SR_HAT0,radiance_GT,ratio,radiance_range,Qblocks_size,flag_cut_bounds,dim_cut,th_values)      
        
        sCC_SR_HAT0 = sCC_SR[0]
        
        Q2n_SR_HAT1, Q_SR_HAT1, ERGAS_SR_HAT1, SAM_SR_HAT1, sCC_SR, RMSE_SR_HAT1, PSNR_SR_HAT1 = \
            indexes_evaluation_SR(I_SR_HAT1,radiance_GT,ratio,radiance_range,Qblocks_size,flag_cut_bounds,dim_cut,th_values)      
        
        sCC_SR_HAT1 = sCC_SR[0]
        
        Q2n_SR_HAT2, Q_SR_HAT2, ERGAS_SR_HAT2, SAM_SR_HAT2, sCC_SR, RMSE_SR_HAT2, PSNR_SR_HAT2 = \
            indexes_evaluation_SR(I_SR_HAT2,radiance_GT,ratio,radiance_range,Qblocks_size,flag_cut_bounds,dim_cut,th_values)      
        
        sCC_SR_HAT2 = sCC_SR[0]
        
    else:
        min_radiance = np.amin(I_SR_HAT0) 
        max_radiance = np.amax(I_SR_HAT0) 
        scaled_SR = (I_SR_HAT0 - min_radiance) / max_radiance
        
        scaled_SR = scaled_SR.squeeze()
        BRISQUE_SR_HAT0 = eng.brisque(scaled_SR)
        
        min_radiance = np.amin(I_SR_HAT1) 
        max_radiance = np.amax(I_SR_HAT1) 
        scaled_SR = (I_SR_HAT1 - min_radiance) / max_radiance
        
        scaled_SR = scaled_SR.squeeze()
        BRISQUE_SR_HAT1 = eng.brisque(scaled_SR)
        
        min_radiance = np.amin(I_SR_HAT2) 
        max_radiance = np.amax(I_SR_HAT2) 
        scaled_SR = (I_SR_HAT2 - min_radiance) / max_radiance
        
        scaled_SR = scaled_SR.squeeze()
        BRISQUE_SR_HAT2 = eng.brisque(scaled_SR)

    if results:  
        dirs_res = dirs_res_path + '/HAT0'
        if not os.path.isdir(dirs_res):
            os.makedirs(dirs_res)
        
        I_SR_HAT0 = np.squeeze(I_SR_HAT0)
        
        sio.savemat(dirs_res + '/' + im_tag + '_SR.mat', {'radiance': I_SR_HAT0})
        
        dirs_res = dirs_res_path + '/HAT1'
        if not os.path.isdir(dirs_res):
            os.makedirs(dirs_res)
        
        I_SR_HAT1 = np.squeeze(I_SR_HAT1)
        
        sio.savemat(dirs_res + '/' + im_tag + '_SR.mat', {'radiance': I_SR_HAT1})
        
        dirs_res = dirs_res_path + '/HAT2'
        if not os.path.isdir(dirs_res):
            os.makedirs(dirs_res)
        
        I_SR_HAT2 = np.squeeze(I_SR_HAT2)
        
        sio.savemat(dirs_res + '/' + im_tag + '_SR.mat', {'radiance': I_SR_HAT2})

#%%
'''S5Net same filter'''

inter = 'cubic'

I_LR = radiance_LR
I_LR1 = np.zeros([I_LR.shape[0],I_LR.shape[1],1])
I_LR1[:,:,0] = I_LR
I_LR = I_LR1
I_LR = np.transpose(I_LR, (2, 0, 1))

lr_test = []
lr_test.append(I_LR[:,:,:])
lr_test = np.array(lr_test)

preds = np.zeros([1,lr_test.shape[1],lr_test.shape[2]*ratio,lr_test.shape[3]*ratio])

tc = 1    
ker = kernel(tc,ratio,inter)
ker1 = np.zeros([ker.shape[0],1])
ker1[:,0] = ker
ker = ker1
kernel_dec = np.matmul(ker,np.transpose(ker))    
kernel_dec = np.transpose(kernel_dec)

if protocol == 'RR':
    kernel_dec = torch.tensor(np.reshape(kernel_dec, (1,1,kernel_dec.shape[0],kernel_dec.shape[1])), dtype=torch.float64)
else:
    kernel_dec = torch.tensor(np.reshape(kernel_dec, (1,1,kernel_dec.shape[0],kernel_dec.shape[1])))

model = S5Net(64,32,32,9,5,5,c,kernel_dec.shape[2],ratio).to(device)

epochs_path = './trained_models/S5Net/{}.{}/{}/'.format(num,band,protocol) 

if(os.path.isfile(epochs_path + 'x{}_{}_match.pth'.format(ratio,im_tag))): 
    for i in range(lr_test.shape[1]):            
        lr_test1 = np.zeros([1,1,lr_test.shape[2],lr_test.shape[3]])
        lr_test1[0,0,:,:] = lr_test[0,i,:,:]
        if protocol == 'RR':
            lr_test1 = torch.tensor(lr_test1, dtype=torch.float64)
        else:
            lr_test1 = torch.tensor(lr_test1)
            lr_test1 = lr_test1.float()
            
        if torch.cuda.is_available():
            lr_test1 = lr_test1.to(device)
    
        with torch.no_grad():
            model.load_state_dict(torch.load(epochs_path + 'x{}_{}_match.pth'.format(ratio,im_tag)))
            model.eval()
            
            preds1 = model(lr_test1,ratio,device)
        
        preds1 = preds1.cpu()
        preds1 = preds1.detach().numpy()
        preds[0,i,:,:] = preds1
    
    del preds1 
    del lr_test1
    torch.cuda.empty_cache()
    I_SR_test = np.transpose(preds, (0, 2, 3, 1))
    I_SR1_test = np.array([I_SR_test.shape[1],I_SR_test.shape[2],1])
    I_SR1_test = I_SR_test[0,:,:,:]
    I_SR_test = I_SR1_test            
    I_SR_S5Net_match = I_SR_test.astype('float64')
    
    I_SR_S5Net_match = (I_SR_S5Net_match/0.5) * max_val
    
    if protocol == 'RR':
        Q2n_SR_S5Net_match, Q_SR_S5Net_match, ERGAS_SR_S5Net_match, SAM_SR_S5Net_match, sCC_SR, RMSE_SR_S5Net_match, PSNR_SR_S5Net_match = \
            indexes_evaluation_SR(I_SR_S5Net_match,radiance_GT,ratio,radiance_range,Qblocks_size,flag_cut_bounds,dim_cut,th_values)      
        
        sCC_SR_S5Net_match = sCC_SR[0]
        
    else:
        min_radiance = np.amin(I_SR_S5Net_match) 
        max_radiance = np.amax(I_SR_S5Net_match) 
        scaled_SR = (I_SR_S5Net_match - min_radiance) / max_radiance
        
        scaled_SR = scaled_SR.squeeze()
        BRISQUE_SR_S5Net_match = eng.brisque(scaled_SR)

    if results:  
        dirs_res = dirs_res_path + '/S5Net_match'
        if not os.path.isdir(dirs_res):
            os.makedirs(dirs_res)
        
        I_SR_S5Net_match = np.squeeze(I_SR_S5Net_match)
        
        sio.savemat(dirs_res + '/' + im_tag + '_SR.mat', {'radiance': I_SR_S5Net_match})
    
#%%
'''S5Net different filter'''

preds = np.zeros([1,lr_test.shape[1],lr_test.shape[2]*ratio,lr_test.shape[3]*ratio])

tc = 1    
ker = kernel(tc,ratio,inter)
ker1 = np.zeros([ker.shape[0],1])
ker1[:,0] = ker
ker = ker1
kernel_dec = np.matmul(ker,np.transpose(ker))    
kernel_dec = np.transpose(kernel_dec)
if protocol == 'RR':
    kernel_dec = torch.tensor(np.reshape(kernel_dec, (1,1,kernel_dec.shape[0],kernel_dec.shape[1])), dtype=torch.float64)
else:
    kernel_dec = torch.tensor(np.reshape(kernel_dec, (1,1,kernel_dec.shape[0],kernel_dec.shape[1])))

model = S5Net(64,32,32,9,5,5,c,kernel_dec.shape[2],ratio).to(device)

epochs_path = './trained_models/S5Net/{}.{}/{}/'.format(num,band,protocol)

if(os.path.isfile(epochs_path + 'x{}_{}_nomatch.pth'.format(ratio,im_tag))): 
    for i in range(lr_test.shape[1]):            
        lr_test1 = np.zeros([1,1,lr_test.shape[2],lr_test.shape[3]])
        lr_test1[0,0,:,:] = lr_test[0,i,:,:]
        if protocol == 'RR':
            lr_test1 = torch.tensor(lr_test1, dtype=torch.float64)
        else:
            lr_test1 = torch.tensor(lr_test1)
            lr_test1 = lr_test1.float()
            
        if torch.cuda.is_available():
            lr_test1 = lr_test1.to(device)
    
        with torch.no_grad():
            model.load_state_dict(torch.load(epochs_path + 'x{}_{}_nomatch.pth'.format(ratio,im_tag)))
            model.eval()
            
            preds1 = model(lr_test1,ratio,device)
        
        preds1 = preds1.cpu()
        preds1 = preds1.detach().numpy()
        preds[0,i,:,:] = preds1
    
    del preds1 
    del lr_test1
    torch.cuda.empty_cache()
    I_SR_test = np.transpose(preds, (0, 2, 3, 1))
    I_SR1_test = np.array([I_SR_test.shape[1],I_SR_test.shape[2],1])
    I_SR1_test = I_SR_test[0,:,:,:]
    I_SR_test = I_SR1_test            
    I_SR_S5Net_nomatch = I_SR_test.astype('float64')
    
    I_SR_S5Net_nomatch = (I_SR_S5Net_nomatch/0.5) * max_val
    
    if protocol == 'RR':
        Q2n_SR_S5Net_nomatch, Q_SR_S5Net_nomatch, ERGAS_SR_S5Net_nomatch, SAM_SR_S5Net_nomatch, sCC_SR, RMSE_SR_S5Net_nomatch, PSNR_SR_S5Net_nomatch = \
            indexes_evaluation_SR(I_SR_S5Net_nomatch,radiance_GT,ratio,radiance_range,Qblocks_size,flag_cut_bounds,dim_cut,th_values)      
        
        sCC_SR_S5Net_nomatch = sCC_SR[0]
        
    else:
        min_radiance = np.amin(I_SR_S5Net_nomatch) 
        max_radiance = np.amax(I_SR_S5Net_nomatch) 
        scaled_SR = (I_SR_S5Net_nomatch - min_radiance) / max_radiance
        
        scaled_SR = scaled_SR.squeeze()
        BRISQUE_SR_S5Net_nomatch = eng.brisque(scaled_SR)

    if results:  
        dirs_res = dirs_res_path + '/S5Net_nomatch'
        if not os.path.isdir(dirs_res):
            os.makedirs(dirs_res)
        
        I_SR_S5Net_nomatch = np.squeeze(I_SR_S5Net_nomatch)
        
        sio.savemat(dirs_res + '/' + im_tag + '_SR.mat', {'radiance': I_SR_S5Net_nomatch})

#%%
'''S5Net (no deconvolution)'''

f1 = 9;f2 = 5;f3 = 5;c = 1
interp_methods = ['nearest','linear','quadratic','cubic0.5','cubic0.75','cubic1','lanczos1',
                  'lanczos2','lanczos3','23tap']

if protocol == 'RR':
    Q2n_SR_S5Net_nodec = np.zeros(len(interp_methods))
    Q_SR_S5Net_nodec = np.zeros(len(interp_methods))
    ERGAS_SR_S5Net_nodec = np.zeros(len(interp_methods))
    SAM_SR_S5Net_nodec = np.zeros(len(interp_methods))
    sCC_SR_S5Net_nodec = np.zeros(len(interp_methods))
    RMSE_SR_S5Net_nodec = np.zeros(len(interp_methods))
    PSNR_SR_S5Net_nodec = np.zeros(len(interp_methods))
else:
    BRISQUE_SR_S5Net_nodec = np.zeros(len(interp_methods))

method_i = 0
for method in interp_methods:
    I_LR = interp(radiance_LR, ratio, method)
    I_LR = np.transpose(I_LR, (2, 0, 1))

    lr_test = []
    lr_test.append(I_LR[:,:,:])
    lr_test = np.array(lr_test)

    preds = np.zeros([1,lr_test.shape[1],lr_test.shape[2],lr_test.shape[3]])

    model = SRCNN(64,32,32,f1,f2,f3,c).to(device)

    epochs_path = './trained_models/S5Net_no_transpose/{}.{}/{}/'.format(num,band,protocol) 

    if(os.path.isfile(epochs_path + 'x{}_{}_match.pth'.format(ratio,im_tag))): 
        for i in range(lr_test.shape[1]):            
            lr_test1 = np.zeros([1,1,lr_test.shape[2],lr_test.shape[3]])
            lr_test1[0,0,:,:] = lr_test[0,i,:,:]
            if protocol == 'RR':
                lr_test1 = torch.tensor(lr_test1, dtype=torch.float64)
            else:
                lr_test1 = torch.tensor(lr_test1)
                lr_test1 = lr_test1.float()
                
            if torch.cuda.is_available():
                lr_test1 = lr_test1.to(device)
        
            model.load_state_dict(torch.load(epochs_path + 'x{}_{}_match.pth'.format(ratio,im_tag)))
            model.eval()
            
            preds1 = model(lr_test1)
            
            preds1 = preds1.cpu()
            preds1 = preds1.detach().numpy()
            preds[0,i,:,:] = preds1
        
        del preds1 
        del lr_test1
        torch.cuda.empty_cache()
        I_SR_test = np.transpose(preds, (0, 2, 3, 1))
        I_SR1_test = np.array([I_SR_test.shape[1],I_SR_test.shape[2],1])
        I_SR1_test = I_SR_test[0,:,:,:]
        I_SR_test = I_SR1_test            
        I_SR_S5Net_nodec = I_SR_test.astype('float64')
        
        I_SR_S5Net_nodec = (I_SR_S5Net_nodec/0.5) * max_val
        
        if protocol == 'RR':
            Q2n_SR_S5Net_nodec[method_i], Q_SR_S5Net_nodec[method_i], ERGAS_SR_S5Net_nodec[method_i], SAM_SR_S5Net_nodec[method_i], sCC_SR, RMSE_SR_S5Net_nodec[method_i], PSNR_SR_S5Net_nodec[method_i] = \
                indexes_evaluation_SR(I_SR_S5Net_nodec,radiance_GT,ratio,radiance_range,Qblocks_size,flag_cut_bounds,dim_cut,th_values)      
            
            sCC_SR_S5Net_nodec[method_i] = sCC_SR[0]
            
        else:
            min_radiance = np.amin(I_SR_S5Net_nodec) 
            max_radiance = np.amax(I_SR_S5Net_nodec) 
            scaled_SR = (I_SR_S5Net_nodec - min_radiance) / max_radiance
            
            scaled_SR = scaled_SR.squeeze()
            BRISQUE_SR_S5Net_nodec[method_i] = eng.brisque(scaled_SR)
        
        method_i += 1
        
        if results:  
            dirs_res = dirs_res_path + '/S5Net_nodec_{}'.format(method)
            if not os.path.isdir(dirs_res):
                os.makedirs(dirs_res)
            
            I_SR_S5Net_nodec = np.squeeze(I_SR_S5Net_nodec)
            
            sio.savemat(dirs_res + '/' + im_tag + '_SR.mat', {'radiance': I_SR_S5Net_nodec})
 
#%%
"""ESPCN+S5Net_nodec same filter"""

model = ESPCN(in_channels = 1, out_channels = 1, channels = 64, upscale_factor = ratio).to(device)

I_LR = radiance_LR
I_LR1 = np.zeros([I_LR.shape[0],I_LR.shape[1],1])
I_LR1[:,:,0] = I_LR
I_LR = I_LR1

I_LR = np.transpose(I_LR, (2, 0, 1))

lr_test = []
lr_test.append(I_LR[:,:,:])
lr_test = np.array(lr_test)

trained_model_path = './trained_models/SOTA/ESPCN/x{}.pth'.format(ratio) 

if(os.path.isfile(trained_model_path)):          
    if protocol == 'RR':
        lr_test = torch.tensor(lr_test, dtype=torch.float64)
    else:
        lr_test = torch.tensor(lr_test)
        lr_test = lr_test.float()
            
    if torch.cuda.is_available():
        lr_test = lr_test.to(device)

    pretrained_model = torch.load(trained_model_path, map_location=device)['state_dict']
    
    with torch.no_grad():
        model.load_state_dict(pretrained_model)
        model.eval()
        
        preds = model(lr_test)
    
    preds = preds.cpu()
    preds = preds.detach().numpy()
    I_SR_test = np.transpose(preds, (0, 2, 3, 1))
    del preds 
    del lr_test
    torch.cuda.empty_cache()
    I_SR1_test = np.zeros([I_SR_test.shape[1],I_SR_test.shape[2],1])
    I_SR1_test[:,:,0] = I_SR_test[0,:,:,0]
    I_SR_test = I_SR1_test            
    I_SR_ESPCN = I_SR_test.astype('float64')

f1 = 9;f2 = 5;f3 = 5;c = 1
inter = 'cubic0.5'

I_LR = I_SR_ESPCN
I_LR = np.transpose(I_LR, (2, 0, 1))

lr_test = []
lr_test.append(I_LR[:,:,:])
lr_test = np.array(lr_test)

preds = np.zeros([1,lr_test.shape[1],lr_test.shape[2],lr_test.shape[3]])

epochs_path = './trained_models/S5Net/{}.{}/{}/'.format(num,band,protocol)

if(os.path.isfile(epochs_path + 'x{}_{}_match.pth'.format(ratio,im_tag))): 
    for i in range(lr_test.shape[1]):            
        lr_test1 = np.zeros([1,1,lr_test.shape[2],lr_test.shape[3]])
        lr_test1[0,0,:,:] = lr_test[0,i,:,:]
        if protocol == 'RR':
            lr_test1 = torch.tensor(lr_test1, dtype=torch.float64)
        else:
            lr_test1 = torch.tensor(lr_test1)
            lr_test1 = lr_test1.float()
            
        if torch.cuda.is_available():
            lr_test1 = lr_test1.to(device)
            
        with torch.no_grad():
            state_dict = torch.load(epochs_path + 'x{}_{}_match.pth'.format(ratio,im_tag))
            weights_conv1 = state_dict['conv1.weight'].clone().detach().requires_grad_(True).type(torch.DoubleTensor)
            weights_conv2 = state_dict['conv2.weight'].clone().detach().requires_grad_(True).type(torch.DoubleTensor)
            weights_conv3 = state_dict['conv3.weight'].clone().detach().requires_grad_(True).type(torch.DoubleTensor)
            
            biases_conv1 = state_dict['conv1.bias'].clone().detach().requires_grad_(True).type(torch.DoubleTensor)
            biases_conv2 = state_dict['conv2.bias'].clone().detach().requires_grad_(True).type(torch.DoubleTensor)
            biases_conv3 = state_dict['conv3.bias'].clone().detach().requires_grad_(True).type(torch.DoubleTensor)
            
            model = SRCNN(64,32,32,f1,f2,f3,c,weights_conv1=weights_conv1,weights_conv2=weights_conv2,weights_conv3=weights_conv3,biases_conv1=biases_conv1,biases_conv2=biases_conv2,biases_conv3=biases_conv3).to(device)
            model.eval()
            
            preds1 = model(lr_test1)
    
        preds1 = preds1.cpu()
        preds1 = preds1.detach().numpy()
        preds[0,i,:,:] = preds1
    
    del preds1 
    del lr_test1
    torch.cuda.empty_cache()
    I_SR_test = np.transpose(preds, (0, 2, 3, 1))
    I_SR1_test = np.array([I_SR_test.shape[1],I_SR_test.shape[2],1])
    I_SR1_test = I_SR_test[0,:,:,:]
    I_SR_test = I_SR1_test            
    I_SR_ESPCN_S5Net_nodec = I_SR_test.astype('float64')
    
    I_SR_ESPCN_S5Net_nodec = (I_SR_ESPCN_S5Net_nodec/0.5) * max_val
    
    if protocol == 'RR':
        Q2n_SR_ESPCN_S5Net_nodec, Q_SR_ESPCN_S5Net_nodec, ERGAS_SR_ESPCN_S5Net_nodec, SAM_SR_ESPCN_S5Net_nodec, sCC_SR, RMSE_SR_ESPCN_S5Net_nodec, PSNR_SR_ESPCN_S5Net_nodec = \
            indexes_evaluation_SR(I_SR_ESPCN_S5Net_nodec,radiance_GT,ratio,radiance_range,Qblocks_size,flag_cut_bounds,dim_cut,th_values)      
        
        sCC_SR_ESPCN_S5Net_nodec = sCC_SR[0]
        
    else:
        min_radiance = np.amin(I_SR_ESPCN_S5Net_nodec) 
        max_radiance = np.amax(I_SR_ESPCN_S5Net_nodec) 
        scaled_SR = (I_SR_ESPCN_S5Net_nodec - min_radiance) / max_radiance
        
        scaled_SR = scaled_SR.squeeze()
        BRISQUE_SR_ESPCN_S5Net_nodec = eng.brisque(scaled_SR)
    
    if results:  
        dirs_res = dirs_res_path + '/ESPCN_S5Net_nodec_{}'.format(method)
        if not os.path.isdir(dirs_res):
            os.makedirs(dirs_res)
        
        I_SR_ESPCN_S5Net_nodec = np.squeeze(I_SR_ESPCN_S5Net_nodec)
        
        sio.savemat(dirs_res + '/' + im_tag + '_SR.mat', {'radiance': I_SR_ESPCN_S5Net_nodec})
        
#%%
'''Save .csv files with results'''
if results:
    if protocol == 'RR':        
        if not (os.path.isfile(dirs_res_path  + '/' + im_tag + '_quality_indexes.csv')):  
            names = ['Method','Q2n','Q','ERGAS','SAM','sCC','RMSE','PSNR']
            f = open(dirs_res_path  + '/'+ im_tag + '_quality_indexes.csv', 'w') 
            wr = csv.writer(f, delimiter = ',', lineterminator='\n')
            wr.writerow(names)  
        l = [] 
        l.append(['GT',0,1,0,0,1,0,float('inf')])
        for i in range(len(interp_type)):
            l.append([interp_type[i],Q2n_SR_int[i],Q_SR_int[i],ERGAS_SR_int[i],SAM_SR_int[i],sCC_SR_int[i],RMSE_SR_int[i],PSNR_SR_int[i]]) 
        l.append(['SRCNN',Q2n_SR_SRCNN0,Q_SR_SRCNN0,ERGAS_SR_SRCNN0,SAM_SR_SRCNN0,sCC_SR_SRCNN0,RMSE_SR_SRCNN0,PSNR_SR_SRCNN0]) 
        l.append(['FSRCNN',Q2n_SR_FSRCNN,Q_SR_FSRCNN,ERGAS_SR_FSRCNN,SAM_SR_FSRCNN,sCC_SR_FSRCNN,RMSE_SR_FSRCNN,PSNR_SR_FSRCNN])
        l.append(['VDSR',Q2n_SR_VDSR,Q_SR_VDSR,ERGAS_SR_VDSR,SAM_SR_VDSR,sCC_SR_VDSR,RMSE_SR_VDSR,PSNR_SR_VDSR])
        l.append(['ESPCN',Q2n_SR_ESPCN,Q_SR_ESPCN,ERGAS_SR_ESPCN,SAM_SR_ESPCN,sCC_SR_ESPCN,RMSE_SR_ESPCN,PSNR_SR_ESPCN])
        l.append(['EDSR0',Q2n_SR_EDSR0,Q_SR_EDSR0,ERGAS_SR_EDSR0,SAM_SR_EDSR0,sCC_SR_EDSR0,RMSE_SR_EDSR0,PSNR_SR_EDSR0])
        l.append(['EDSR1',Q2n_SR_EDSR1,Q_SR_EDSR1,ERGAS_SR_EDSR1,SAM_SR_EDSR1,sCC_SR_EDSR1,RMSE_SR_EDSR1,PSNR_SR_EDSR1])
        l.append(['EDSR2',Q2n_SR_EDSR2,Q_SR_EDSR2,ERGAS_SR_EDSR2,SAM_SR_EDSR2,sCC_SR_EDSR2,RMSE_SR_EDSR2,PSNR_SR_EDSR2])
        l.append(['RCAN0',Q2n_SR_RCAN0,Q_SR_RCAN0,ERGAS_SR_RCAN0,SAM_SR_RCAN0,sCC_SR_RCAN0,RMSE_SR_RCAN0,PSNR_SR_RCAN0])
        l.append(['RCAN1',Q2n_SR_RCAN1,Q_SR_RCAN1,ERGAS_SR_RCAN1,SAM_SR_RCAN1,sCC_SR_RCAN1,RMSE_SR_RCAN1,PSNR_SR_RCAN1])
        l.append(['RCAN2',Q2n_SR_RCAN2,Q_SR_RCAN2,ERGAS_SR_RCAN2,SAM_SR_RCAN2,sCC_SR_RCAN2,RMSE_SR_RCAN2,PSNR_SR_RCAN2])
        l.append(['RankSRGAN0',Q2n_SR_RankSRGAN0,Q_SR_RankSRGAN0,ERGAS_SR_RankSRGAN0,SAM_SR_RankSRGAN0,sCC_SR_RankSRGAN0,RMSE_SR_RankSRGAN0,PSNR_SR_RankSRGAN0])
        l.append(['RankSRGAN1',Q2n_SR_RankSRGAN1,Q_SR_RankSRGAN1,ERGAS_SR_RankSRGAN1,SAM_SR_RankSRGAN1,sCC_SR_RankSRGAN1,RMSE_SR_RankSRGAN1,PSNR_SR_RankSRGAN1])
        l.append(['RankSRGAN2',Q2n_SR_RankSRGAN2,Q_SR_RankSRGAN2,ERGAS_SR_RankSRGAN2,SAM_SR_RankSRGAN2,sCC_SR_RankSRGAN2,RMSE_SR_RankSRGAN2,PSNR_SR_RankSRGAN2])
        l.append(['PAN0',Q2n_SR_PAN0,Q_SR_PAN0,ERGAS_SR_PAN0,SAM_SR_PAN0,sCC_SR_PAN0,RMSE_SR_PAN0,PSNR_SR_PAN0])
        l.append(['PAN1',Q2n_SR_PAN1,Q_SR_PAN1,ERGAS_SR_PAN1,SAM_SR_PAN1,sCC_SR_PAN1,RMSE_SR_PAN1,PSNR_SR_PAN1])
        l.append(['PAN2',Q2n_SR_PAN2,Q_SR_PAN2,ERGAS_SR_PAN2,SAM_SR_PAN2,sCC_SR_PAN2,RMSE_SR_PAN2,PSNR_SR_PAN2])
        l.append(['BSRN0',Q2n_SR_BSRN0,Q_SR_BSRN0,ERGAS_SR_BSRN0,SAM_SR_BSRN0,sCC_SR_BSRN0,RMSE_SR_BSRN0,PSNR_SR_BSRN0])
        l.append(['BSRN1',Q2n_SR_BSRN1,Q_SR_BSRN1,ERGAS_SR_BSRN1,SAM_SR_BSRN1,sCC_SR_BSRN1,RMSE_SR_BSRN1,PSNR_SR_BSRN1])
        l.append(['BSRN2',Q2n_SR_BSRN2,Q_SR_BSRN2,ERGAS_SR_BSRN2,SAM_SR_BSRN2,sCC_SR_BSRN2,RMSE_SR_BSRN2,PSNR_SR_BSRN2])
        l.append(['HAT0',Q2n_SR_HAT0,Q_SR_HAT0,ERGAS_SR_HAT0,SAM_SR_HAT0,sCC_SR_HAT0,RMSE_SR_HAT0,PSNR_SR_HAT0])
        l.append(['HAT1',Q2n_SR_HAT1,Q_SR_HAT1,ERGAS_SR_HAT1,SAM_SR_HAT1,sCC_SR_HAT1,RMSE_SR_HAT1,PSNR_SR_HAT1])
        l.append(['HAT2',Q2n_SR_HAT2,Q_SR_HAT2,ERGAS_SR_HAT2,SAM_SR_HAT2,sCC_SR_HAT2,RMSE_SR_HAT2,PSNR_SR_HAT2])
        l.append(['CGA_nomatch',Q2n_SR_CGA_nomatch,Q_SR_CGA_nomatch,ERGAS_SR_CGA_nomatch,SAM_SR_CGA_nomatch,sCC_SR_CGA_nomatch,RMSE_SR_CGA_nomatch,PSNR_SR_CGA_nomatch])
        l.append(['CGA_match',Q2n_SR_CGA_match,Q_SR_CGA_match,ERGAS_SR_CGA_match,SAM_SR_CGA_match,sCC_SR_CGA_match,RMSE_SR_CGA_match,PSNR_SR_CGA_match])
        l.append(['S5Net_nomatch',Q2n_SR_S5Net_nomatch,Q_SR_S5Net_nomatch,ERGAS_SR_S5Net_nomatch,SAM_SR_S5Net_nomatch,sCC_SR_S5Net_nomatch,RMSE_SR_S5Net_nomatch,PSNR_SR_S5Net_nomatch])
        l.append(['S5Net_match',Q2n_SR_S5Net_match,Q_SR_S5Net_match,ERGAS_SR_S5Net_match,SAM_SR_S5Net_match,sCC_SR_S5Net_match,RMSE_SR_S5Net_match,PSNR_SR_S5Net_match])
        for i in range(len(interp_methods)):
            l.append(['S5Net_nodec_{}'.format(interp_methods[i]),Q2n_SR_S5Net_nodec[i],Q_SR_S5Net_nodec[i],ERGAS_SR_S5Net_nodec[i],SAM_SR_S5Net_nodec[i],sCC_SR_S5Net_nodec[i],RMSE_SR_S5Net_nodec[i],PSNR_SR_S5Net_nodec[i]])
        l.append(['ESPCN+S5Net_nodec',Q2n_SR_ESPCN_S5Net_nodec,Q_SR_ESPCN_S5Net_nodec,ERGAS_SR_ESPCN_S5Net_nodec,SAM_SR_ESPCN_S5Net_nodec,sCC_SR_ESPCN_S5Net_nodec,RMSE_SR_ESPCN_S5Net_nodec,PSNR_SR_ESPCN_S5Net_nodec])
    else:
        if not (os.path.isfile(dirs_res_path  + '/'+ im_tag + '_quality_indexes.csv')):
            names = ['Method','BRISQUE']   
            f = open(dirs_res_path  + '/'+ im_tag + '_quality_indexes.csv', 'w') 
            wr = csv.writer(f, delimiter = ',', lineterminator='\n')
            wr.writerow(names) 
        l = []
        l.append(['GT',0])
        for i in range(len(interp_type)):
            l.append([interp_type[i],BRISQUE_SR_int[i]])  
        l.append(['SRCNN',BRISQUE_SR_SRCNN0])
        l.append(['FSRCNN',BRISQUE_SR_FSRCNN])
        l.append(['VDSR',BRISQUE_SR_VDSR])
        l.append(['ESPCN',BRISQUE_SR_ESPCN])
        l.append(['EDSR0',BRISQUE_SR_EDSR0])
        l.append(['EDSR1',BRISQUE_SR_EDSR1])
        l.append(['EDSR2',BRISQUE_SR_EDSR2])
        l.append(['RCAN0',BRISQUE_SR_RCAN0])
        l.append(['RCAN1',BRISQUE_SR_RCAN1])
        l.append(['RCAN2',BRISQUE_SR_RCAN2])
        l.append(['RankSRGAN0',BRISQUE_SR_RankSRGAN0])
        l.append(['RankSRGAN1',BRISQUE_SR_RankSRGAN1])
        l.append(['RankSRGAN2',BRISQUE_SR_RankSRGAN2])
        l.append(['PAN0',BRISQUE_SR_PAN0])
        l.append(['PAN1',BRISQUE_SR_PAN1])
        l.append(['PAN2',BRISQUE_SR_PAN2])
        l.append(['BSRN0',BRISQUE_SR_BSRN0])
        l.append(['BSRN1',BRISQUE_SR_BSRN1])
        l.append(['BSRN2',BRISQUE_SR_BSRN2])
        l.append(['HAT0',BRISQUE_SR_HAT0])
        l.append(['HAT1',BRISQUE_SR_HAT1])
        l.append(['HAT2',BRISQUE_SR_HAT2])
        l.append(['CGA_nomatch',BRISQUE_SR_CGA_nomatch])
        l.append(['CGA_match',BRISQUE_SR_CGA_match])
        l.append(['S5Net_nomatch',BRISQUE_SR_S5Net_nomatch])
        l.append(['S5Net_match',BRISQUE_SR_S5Net_match])   
        for i in range(len(interp_methods)):
            l.append(['S5Net_nodec_{}'.format(interp_methods[i]),BRISQUE_SR_S5Net_nodec[i]])
        l.append(['ESPCN+S5Net_nodec',BRISQUE_SR_ESPCN_S5Net_nodec])

    wr.writerows(l)
    f.close()       
        