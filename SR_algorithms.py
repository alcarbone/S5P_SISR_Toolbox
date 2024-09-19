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
               * cubic (A2 = -1/2)
           For deconvolution we have:
               * CGA 
           For neural networks we have:
               SOTA:
                   * SRCNN 
                   * VDSR
                   * EDSR
                   * PAN 
                   * HAT 
               Original:
                   * S5Net 
               Computationally-efficient:
                   * DSR-S5Net-st
                   * DSR-S5Net-dyn
                   * DSR-S5Net-st
                   * DSR-S5Net-dyn
                   
           If the protocol is RR (Reduced Resolution) the indices used are
           Q2n, Dlambda, Q, ERGAS, SAM, sCC, RMSE and PSNR. 
           If the protocol is FR (Full Resolution) the index used is 
           BRISQUE.
"""

import os
import csv
import h5py
import time
import scipy
import torch

import numpy as np

import matlab.engine
eng = matlab.engine.start_matlab()

from scripts.interp.interp import interp
from scripts.interp.kernel import kernel
from scripts.deconv.deconv import deconv
from scripts.S5Net.S5Net_arch import S5Net
from scripts.resize_image import resize_image

from scripts.SOTA.SRCNN.SRCNN_arch import SRCNN
from scripts.SOTA.VDSR.VDSR_arch import VDSR
from scripts.SOTA.EDSR.EDSR_arch import EDSR
from scripts.SOTA.PAN.PAN_arch import PAN
from scripts.SOTA.HAT.HAT_arch import HAT

from scripts.IQA.indexes_evaluation_SR import indexes_evaluation_SR

Qblocks_size = 32 
th_values = 0
K1 = K2 = 1e-8

if not (os.path.isfile(dirs_res_path + '/all_idxs.csv')):  
    names = ['Method','Q2n','Dlambda','Q','ERGAS','SAM','sCC','RMSE','PSNR','Prediction time (s)']
    f = open(dirs_res_path + 'all_idxs.csv', 'w') 
    wr = csv.writer(f, delimiter = ',', lineterminator='\n',)
    wr.writerow(names)
    l = [] 
    l.append(['GT',1,0,1,0,0,1,0,float('inf'),0])

#%%
"""Prepare image"""

s5p_path = f"data/{im_tag}.nc"

s5p_img_total = np.asarray(h5py.File(s5p_path)['radiance'])
s5p_lat = np.asarray(h5py.File(s5p_path)['latitude'])
s5p_lon = np.asarray(h5py.File(s5p_path)['longitude'])

bands = [2,3,4,5,6,7,8]

if protocol == 'RR':
    img_GTs = list()
    GNyq_xs = list()
    GNyq_ys = list()
img_LRs = list()

img_ids = list()
dets = list()
channels = list()

first_idx = 0

for band in bands:
    
    if (band == 7 or band == 8):
        total_channels = 480
    else:
        total_channels = 497
    
    s5p_img = s5p_img_total[:,:,first_idx:(first_idx+total_channels)]
    first_idx += total_channels
    
    if protocol == 'RR':
        if (band == 1 or band == 2):
            GNyq_x = 0.36
            GNyq_y = 0.37
        elif (band == 7 or band == 8):
            GNyq_x = 0.20
            GNyq_y = 0.15
        else:
            GNyq_x = 0.74
            GNyq_y = 0.44
    
    ch = 0
    
    for bd in range(0,s5p_img.shape[2]):
        if protocol == 'RR':
            img_GT = s5p_img[:,:,bd]
            img_LR = resize_image(img_GT,ratio,GNyq_x,GNyq_y)
            img_GTs.append(img_GT)
            GNyq_xs.append(GNyq_x)
            GNyq_ys.append(GNyq_y)
        else:
            img_LR = s5p_img[:,:,bd]
        
        img_LRs.append(img_LR)
            
        ch += 1
 
if protocol == 'RR':
    img_GT = np.zeros([img_GT.shape[0],img_GT.shape[1],len(img_GTs)])
    GNyq_xs = np.asarray(GNyq_xs)
    GNyq_ys = np.asarray(GNyq_ys)
    
img_LR = np.zeros([img_LR.shape[0],img_LR.shape[1],len(img_LRs)])

for ii in range(len(img_LRs)):
    if protocol == 'RR':
        img_GT[:,:,ii] = img_GTs[ii][:,:]
    img_LR[:,:,ii] = img_LRs[ii][:,:,0]
    
if protocol == 'RR':
    img_GT = img_GT[0:img_LR.shape[0]*ratio,0:img_LR.shape[1]*ratio,:]
    
radiance_range = [np.amin(img_GT), np.amax(img_GT)]
max_img = np.amax(np.reshape(img_GT,[img_GT.shape[0]*img_GT.shape[1],img_GT.shape[2]]),axis=0)

#%%
"""Interpolation algorithms"""

#%%
"""Cubic"""

start = time.time()

img_SR_cubic = interp(img_LR, ratio, 'cubic')

stop = time.time()

time_SR_cubic = stop-start

if protocol == 'RR':
    
    Q2n_SR_cubic, Dlambda_SR_cubic, Q_SR_cubic, ERGAS_SR_cubic, SAM_SR_cubic, sCC_SR, RMSE_SR_cubic, PSNR_SR_cubic = \
        indexes_evaluation_SR(img_SR_cubic,img_LR,img_GT,ratio,radiance_range,GNyq_xs,GNyq_ys,max_img,Qblocks_size,flag_cut_bounds,dim_cut,th_values,K1,K2)
    
    sCC_SR_cubic = sCC_SR[0]
    
else:
    min_radiance = np.amin(img_SR_cubic) 
    max_radiance = np.amax(img_SR_cubic) 
    scaled_SR = (img_SR_cubic - min_radiance) / (max_radiance - min_radiance)
    
    scaled_SR = scaled_SR.squeeze()
    BRISQUE_SR_cubic = eng.brisque(scaled_SR)
    
if results:
    l.append(['Cubic',Q2n_SR_cubic,Dlambda_SR_cubic,Q_SR_cubic,ERGAS_SR_cubic,SAM_SR_cubic,sCC_SR_cubic,RMSE_SR_cubic,PSNR_SR_cubic,time_SR_cubic]) 
    
    new = h5py.File(os.path.join(dirs_res_path + '/int_cubic.nc'), mode='w')
    new.create_dataset('radiance', data=img_SR_cubic, dtype=img_SR_cubic.dtype, shape=img_SR_cubic.shape)
    new.create_dataset('latitude', data=s5p_lat, dtype=s5p_lat.dtype, shape=s5p_lat.shape)
    new.create_dataset('longitude', data=s5p_lon, dtype=s5p_lon.dtype, shape=s5p_lon.shape)
    new.close()
    
#%%
"""Deconvolution algorithms"""

l = 0.1
m = 0.00005 
delta = 10**(-4)
iters = 200

#%%
"""CGA"""

img_SR_cga = np.zeros([img_GT.shape[0],img_GT.shape[1],img_GT.shape[2]])

start = time.time()

for bd in range(0,img_GT.shape[2]):
    GNyq_x = GNyq_xs[bd]
    GNyq_y = GNyq_xs[bd]
            
    SR_cga, it, S1, S2 = deconv(img_LR[:,:,bd],np.zeros([img_LR.shape[0]*ratio,img_LR.shape[1]*ratio]),ratio,GNyq_x,GNyq_y,'cga',iters,delta,m,l)
    img_SR_cga[:,:,bd] = SR_cga[:,:,0]
    
stop = time.time() 

time_SR_cga = stop-start

if protocol == 'RR':
    Q2n_SR_cga, Dlambda_SR_cga, Q_SR_cga, ERGAS_SR_cga, SAM_SR_cga, sCC_SR, RMSE_SR_cga, PSNR_SR_cga = \
        indexes_evaluation_SR(img_SR_cga,img_LR,img_GT,ratio,radiance_range,GNyq_xs,GNyq_ys,max_img,Qblocks_size,flag_cut_bounds,dim_cut,th_values,K1,K2)
    
    sCC_SR_cga = sCC_SR[0]
    
else:
    min_radiance = np.amin(img_SR_cga) 
    max_radiance = np.amax(img_SR_cga) 
    scaled_SR = (img_SR_cga - min_radiance) / max_radiance
    
    scaled_SR = scaled_SR.squeeze()
    BRISQUE_SR_cga = eng.brisque(scaled_SR)
    
if results:
    l.append(['CGA',Q2n_SR_cga,Dlambda_SR_cga,Q_SR_cga,ERGAS_SR_cga,SAM_SR_cga,sCC_SR_cga,RMSE_SR_cga,PSNR_SR_cga,time_SR_cga]) 
    
    new = h5py.File(os.path.join(dirs_res_path + '/cga.nc'), mode='w')
    new.create_dataset('radiance', data=img_SR_cga, dtype=img_SR_cga.dtype, shape=img_SR_cga.shape)
    new.create_dataset('latitude', data=s5p_lat, dtype=s5p_lat.dtype, shape=s5p_lat.shape)
    new.create_dataset('longitude', data=s5p_lon, dtype=s5p_lon.dtype, shape=s5p_lon.shape)
    new.close()

#%%
"""Neural Networks"""
    
import torch

if protocol == 'RR':
    torch.set_default_tensor_type(torch.DoubleTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
        
#%%
"""SRCNN"""

epochs_path = './trained_models/SOTA/SRCNN/'

mat_file = scipy.io.loadmat(epochs_path + 'x{}.mat'.format(ratio))

n1 = 64;n2 = 32;n3 = n2;f1 = 9;f2 = 5;f3 = 5;c = 1

if protocol == 'RR':
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
else:
    weights_conv1 = torch.tensor(mat_file['weights_conv1'], dtype=torch.float32)
    weights_conv1 = np.transpose(weights_conv1,(1,0))
    weights_conv1 = np.reshape(weights_conv1, (n1,c,f1*f1))
    weights_conv1 = np.reshape(weights_conv1, (n1,c,f1,f1), order = 'F')
    weights_conv2 = torch.tensor(mat_file['weights_conv2'], dtype=torch.float32)
    weights_conv2 = np.transpose(weights_conv2,(2,1,0))
    weights_conv2 = np.transpose(weights_conv2,(0,2,1))
    weights_conv2 = np.reshape(weights_conv2, (n2,n1,f2,f2), order = 'F')
    weights_conv3 = torch.tensor(mat_file['weights_conv3'], dtype=torch.float32)
    weights_conv3 = np.reshape(weights_conv3,(c,n3,f3*f3))
    weights_conv3 = np.reshape(weights_conv3,(c,n3,f3,f3), order = 'F')
    biases_conv1 = torch.tensor(mat_file['biases_conv1'], dtype=torch.float32)
    biases_conv1 = np.reshape(biases_conv1, (n1), order = 'F')
    biases_conv2 = torch.tensor(mat_file['biases_conv2'], dtype=torch.float32)
    biases_conv2 = np.reshape(biases_conv2, (n2), order = 'F')
    biases_conv3 = torch.tensor(mat_file['biases_conv3'], dtype=torch.float32)
    biases_conv3 = np.reshape(biases_conv3, (c), order = 'F')

model = SRCNN(n1,n2,n3,f1,f2,f3,c,weights_conv1,weights_conv2,weights_conv3,biases_conv1,biases_conv2,biases_conv3).to(device)

img_SR_SRCNN = np.zeros([img_GT.shape[0],img_GT.shape[1],img_GT.shape[2]])

if protocol == 'RR':
    torch.set_default_tensor_type(torch.DoubleTensor)
else:
    torch.set_default_tensor_type(torch.FloatTensor)

start = time.time()

for bd in range(0,img_GT.shape[2]):
    LR = img_LR[:,:,bd]
    
    LR = interp(LR, ratio, 'cubic')
    
    maxi_val = np.amax(LR) 
    mini_val = np.amin(LR) 

    LR = ((LR - mini_val) / (maxi_val - mini_val)) * 0.5
    
    LR = np.transpose(LR, (2, 0, 1))

    lr_test = []
    lr_test.append(LR[:,:,:])
    lr_test = np.array(lr_test)

    preds = np.zeros([1,lr_test.shape[1],lr_test.shape[2],lr_test.shape[3]])
    
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
    I_SR = I_SR_test.astype('float64')
    
    I_SR = ((I_SR/0.5) * (maxi_val - mini_val)) + mini_val
    
    img_SR_SRCNN[:,:,bd] = I_SR[:,:,0]    
    
stop = time.time()

time_SR_SRCNN = stop-start

if protocol == 'RR':
    Q2n_SR_SRCNN, Dlambda_SR_SRCNN, Q_SR_SRCNN, ERGAS_SR_SRCNN, SAM_SR_SRCNN, sCC_SR, RMSE_SR_SRCNN, PSNR_SR_SRCNN = \
        indexes_evaluation_SR(img_SR_SRCNN,img_LR,img_GT,ratio,radiance_range,GNyq_xs,GNyq_ys,max_img,Qblocks_size,flag_cut_bounds,dim_cut,th_values,K1,K2)
    
    sCC_SR_SRCNN = sCC_SR[0]
    
else:
    min_radiance = np.amin(img_SR_SRCNN) 
    max_radiance = np.amax(img_SR_SRCNN) 
    scaled_SR = (img_SR_SRCNN - min_radiance) / max_radiance
    
    scaled_SR = scaled_SR.squeeze()
    BRISQUE_SR_SRCNN = eng.brisque(scaled_SR)
    
if results:
    l.append(['SRCNN',Q2n_SR_SRCNN,Dlambda_SR_SRCNN,Q_SR_SRCNN,ERGAS_SR_SRCNN,SAM_SR_SRCNN,sCC_SR_SRCNN,RMSE_SR_SRCNN,PSNR_SR_SRCNN,time_SR_SRCNN]) 
    
    new = h5py.File(os.path.join(dirs_res_path + '/SRCNN.nc'), mode='w')
    new.create_dataset('radiance', data=img_SR_SRCNN, dtype=img_SR_SRCNN.dtype, shape=img_SR_SRCNN.shape)
    new.create_dataset('latitude', data=s5p_lat, dtype=s5p_lat.dtype, shape=s5p_lat.shape)
    new.create_dataset('longitude', data=s5p_lon, dtype=s5p_lon.dtype, shape=s5p_lon.shape)
    new.close()
    
#%%
"""VDSR"""

model = VDSR().to(device)

img_SR_VDSR = np.zeros([img_GT.shape[0],img_GT.shape[1],img_GT.shape[2]])

start = time.time()

for bd in range(img_GT.shape[2]):

    I_LR = interp(img_LR[:,:,bd], ratio, 'cubic')
    
    maxi_val = np.amax(I_LR) 
    mini_val = np.amin(I_LR) 

    I_LR = ((I_LR - mini_val) / (maxi_val - mini_val)) * 0.5
    
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
        I_SR = I_SR_test.astype('float64')
        
        I_SR = ((I_SR/0.5) * (maxi_val - mini_val)) + mini_val
        
        img_SR_VDSR[:,:,bd] = I_SR[:,:,0]    
        
        
stop = time.time()
time_SR_VDSR = stop-start

if protocol == 'RR':
    Q2n_SR_VDSR, Dlambda_SR_VDSR, Q_SR_VDSR, ERGAS_SR_VDSR, SAM_SR_VDSR, sCC_SR, RMSE_SR_VDSR, PSNR_SR_VDSR = \
        indexes_evaluation_SR(img_SR_VDSR,img_LR,img_GT,ratio,radiance_range,GNyq_xs,GNyq_ys,max_img,Qblocks_size,flag_cut_bounds,dim_cut,th_values,K1,K2)
    
    sCC_SR_VDSR = sCC_SR[0]
    
else:
    min_radiance = np.amin(img_SR_VDSR) 
    max_radiance = np.amax(img_SR_VDSR) 
    scaled_SR = (img_SR_VDSR - min_radiance) / max_radiance
    
    scaled_SR = scaled_SR.squeeze()
    BRISQUE_SR_VDSR = eng.brisque(scaled_SR) 
    
if results:
    l.append(['VDSR',Q2n_SR_VDSR,Dlambda_SR_VDSR,Q_SR_VDSR,ERGAS_SR_VDSR,SAM_SR_VDSR,sCC_SR_VDSR,RMSE_SR_VDSR,PSNR_SR_VDSR,time_SR_VDSR]) 
    
    new = h5py.File(os.path.join(dirs_res_path + '/VDSR.nc'), mode='w')
    new.create_dataset('radiance', data=img_SR_VDSR, dtype=img_SR_VDSR.dtype, shape=img_SR_VDSR.shape)
    new.create_dataset('latitude', data=s5p_lat, dtype=s5p_lat.dtype, shape=s5p_lat.shape)
    new.create_dataset('longitude', data=s5p_lon, dtype=s5p_lon.dtype, shape=s5p_lon.shape)
    new.close()

#%%
"""EDSR"""

img_SR_EDSR = np.zeros([img_GT.shape[0],img_GT.shape[1],img_GT.shape[2]])

start = time.time()

for bd in range(img_GT.shape[2]):
    I_LR = img_LR[:,:,bd]
    
    maxi_val = np.amax(I_LR) 
    mini_val = np.amin(I_LR) 

    I_LR = ((I_LR - mini_val) / (maxi_val - mini_val)) * 0.5
    
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
    
    model = EDSR(n_colors=3,n_feats=256,n_resblocks=32,scale=ratio,res_scale=0.1,rgb_range=np.max(img_LR[:,:,bd]),rgb_mean=(np.mean(img_LR[:,:,bd]),np.mean(img_LR[:,:,bd]),np.mean(img_LR[:,:,bd]))).to(device)
    
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
        I_SR1_test = np.zeros([I_SR_test.shape[1],I_SR_test.shape[2],1])
        if ((dets[bd] == 'NIR') or (dets[bd] == 'SWIR')):
            I_SR1_test[:,:,0] = I_SR_test[0,:,:,0]
        elif ((dets[bd] == 'UVIS')):
            I_SR1_test[:,:,0] = I_SR_test[0,:,:,1]
        else:
            I_SR1_test[:,:,0] = I_SR_test[0,:,:,2]
        I_SR_test = I_SR1_test
        I_SR = I_SR_test.astype('float64')
        I_SR = (I_SR/255) * maximum
        I_SR = ((I_SR/0.5) * (maxi_val - mini_val)) + mini_val
        
        img_SR_EDSR[:,:,bd] = I_SR[:,:,0]    

stop = time.time()
time_SR_EDSR = stop-start

if protocol == 'RR':
    Q2n_SR_EDSR, Dlambda_SR_EDSR, Q_SR_EDSR, ERGAS_SR_EDSR, SAM_SR_EDSR, sCC_SR, RMSE_SR_EDSR, PSNR_SR_EDSR = \
        indexes_evaluation_SR(img_SR_EDSR,img_LR,img_GT,ratio,radiance_range,GNyq_xs,GNyq_ys,max_img,Qblocks_size,flag_cut_bounds,dim_cut,th_values,K1,K2)
    
    sCC_SR_EDSR = sCC_SR[0]
    
else:
    min_radiance = np.amin(img_SR_EDSR) 
    max_radiance = np.amax(img_SR_EDSR) 
    scaled_SR = (img_SR_EDSR - min_radiance) / max_radiance
    
    scaled_SR = scaled_SR.squeeze()
    BRISQUE_SR_EDSR = eng.brisque(scaled_SR)
    
if results:
    l.append(['EDSR',Q2n_SR_EDSR,Dlambda_SR_EDSR,Q_SR_EDSR,ERGAS_SR_EDSR,SAM_SR_EDSR,sCC_SR_EDSR,RMSE_SR_EDSR,PSNR_SR_EDSR,time_SR_EDSR]) 
    
    new = h5py.File(os.path.join(dirs_res_path + '/EDSR.nc'), mode='w')
    new.create_dataset('radiance', data=img_SR_EDSR, dtype=img_SR_EDSR.dtype, shape=img_SR_EDSR.shape)
    new.create_dataset('latitude', data=s5p_lat, dtype=s5p_lat.dtype, shape=s5p_lat.shape)
    new.create_dataset('longitude', data=s5p_lon, dtype=s5p_lon.dtype, shape=s5p_lon.shape)
    new.close()
    
#%%
"""PAN"""

model = PAN(in_nc=3, out_nc=3, nf=40, unf=24, nb=16, scale=ratio).to(device)

img_SR_PAN = np.zeros([img_GT.shape[0],img_GT.shape[1],img_GT.shape[2]])


start = time.time()

for bd in range(img_LR.shape[2]):
    I_LR = img_LR[:,:,bd]
    
    maxi_val = np.amax(I_LR) 
    mini_val = np.amin(I_LR) 
    
    I_LR = ((I_LR - mini_val) / (maxi_val - mini_val)) * 0.5
    
    I_LR1 = np.zeros([I_LR.shape[0],I_LR.shape[1],3])
    I_LR1[:,:,0] = I_LR
    I_LR1[:,:,1] = I_LR
    I_LR1[:,:,2] = I_LR
    I_LR = I_LR1
    I_LR = np.transpose(I_LR, (2,0,1))
    
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
        I_SR_test = np.transpose(preds, (0,2,3,1))
        del preds 
        del lr_test
        torch.cuda.empty_cache()
        I_SR1_test = np.zeros([I_SR_test.shape[1],I_SR_test.shape[2],1])
        if ((dets[bd] == 'NIR') or (dets[bd] == 'SWIR')):
            I_SR1_test[:,:,0] = I_SR_test[0,:,:,0]
        elif ((dets[bd] == 'UVIS')):
            I_SR1_test[:,:,0] = I_SR_test[0,:,:,1]
        else:
            I_SR1_test[:,:,0] = I_SR_test[0,:,:,2]
        I_SR_test = I_SR1_test
        I_SR_PAN = I_SR_test.astype('float64')
        I_SR_PAN = ((I_SR_PAN/0.5) * (maxi_val - mini_val)) + mini_val
        
        img_SR_PAN[:,:,bd] = I_SR_PAN[:,:,0]
        
stop = time.time()
time_SR_PAN = stop-start

if protocol == 'RR':
    Q2n_SR_PAN, Dlambda_SR_PAN, Q_SR_PAN, ERGAS_SR_PAN, SAM_SR_PAN, sCC_SR, RMSE_SR_PAN, PSNR_SR_PAN = \
        indexes_evaluation_SR(img_SR_PAN,img_LR,img_GT,ratio,radiance_range,GNyq_xs,GNyq_ys,max_img,Qblocks_size,flag_cut_bounds,dim_cut,th_values,K1,K2)
    
    sCC_SR_PAN = sCC_SR[0]
    
else:
    min_radiance = np.amin(img_SR_PAN) 
    max_radiance = np.amax(img_SR_PAN) 
    scaled_SR = (img_SR_PAN - min_radiance) / max_radiance
    
    scaled_SR = scaled_SR.squeeze()
    BRISQUE_SR_PAN = eng.brisque(scaled_SR)
    
if results:
    l.append(['PAN',Q2n_SR_PAN,Dlambda_SR_PAN,Q_SR_PAN,ERGAS_SR_PAN,SAM_SR_PAN,sCC_SR_PAN,RMSE_SR_PAN,PSNR_SR_PAN,time_SR_PAN]) 
    
    new = h5py.File(os.path.join(dirs_res_path + '/PAN.nc'), mode='w')
    new.create_dataset('radiance', data=img_SR_PAN, dtype=img_SR_PAN.dtype, shape=img_SR_PAN.shape)
    new.create_dataset('latitude', data=s5p_lat, dtype=s5p_lat.dtype, shape=s5p_lat.shape)
    new.create_dataset('longitude', data=s5p_lon, dtype=s5p_lon.dtype, shape=s5p_lon.shape)
    new.close()
    
#%%
"""HAT"""

img_SR_HAT = np.zeros([img_GT.shape[0],img_GT.shape[1],img_GT.shape[2]])

start = time.time()

for bd in range(img_GT.shape[2]):
    I_LR = img_LR[:,:,bd]
    
    shape = [I_LR.shape[0],I_LR.shape[1]]
    new_shape = [2**(shape[0] - 1).bit_length(),2**(shape[1] - 1).bit_length()]
    diff = np.asarray(new_shape)-np.asarray(shape)
    if (diff[0]%2 == 1):
        pad0 = (diff[0]+1)//2
    else:
        pad0 = diff[0]//2
    if (diff[1]%2 == 1):
        pad1 = (diff[1]+1)//2
    else:
        pad1 = diff[1]//2
        
    I_LR = np.pad(I_LR,((pad0,pad0),(pad1,pad1)),mode='symmetric')
    I_LR = I_LR[0:new_shape[0],0:new_shape[1]]
    
    maxi_val = np.amax(I_LR) 
    mini_val = np.amin(I_LR) 
    
    I_LR = ((I_LR - mini_val) / (maxi_val - mini_val)) * 0.5
    
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

    model = HAT(embed_dim=180,depths=(6,6,6,6,6,6),num_heads=(6,6,6,6,6,6),window_size=16,mlp_ratio=2.,upscale=ratio,img_range=np.max(lr_test),upsampler='pixelshuffle',resi_connection='1conv',rgb_mean=(np.mean(I_LR),np.mean(I_LR),np.mean(I_LR))).to(device)

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
        shape = [I_SR_test.shape[1],I_SR_test.shape[2]]
        new_shape = [img_GT.shape[0],img_GT.shape[1]]
        diff = (np.asarray(shape)-np.asarray(new_shape))
        if (diff[0]%2 == 1):
            pad_sx0 = (diff[0]//2)
            pad_dx0 = pad_sx0 + 1
        else:
            pad_sx0 = (diff[0]//2)
            pad_dx0 = pad_sx0 
        if (diff[1]%2 == 1):
            pad_sx1 = (diff[1]//2)
            pad_dx1 = pad_sx1 + 1
        else:
            pad_sx1 = (diff[1]//2)
            pad_dx1 = pad_sx1 
        I_SR_test = I_SR_test[:,pad_sx0:(shape[0]-pad_dx0),pad_sx1:(shape[1]-pad_dx1),:]
        I_SR1_test = np.zeros([I_SR_test.shape[1],I_SR_test.shape[2],1])
        if ((dets[bd] == 'NIR') or (dets[bd] == 'SWIR')):
            I_SR1_test[:,:,0] = I_SR_test[0,:,:,0]
        elif ((dets[bd] == 'UVIS')):
            I_SR1_test[:,:,0] = I_SR_test[0,:,:,1]
        else:
            I_SR1_test[:,:,0] = I_SR_test[0,:,:,2]
        I_SR_test = I_SR1_test
        I_SR_HAT = I_SR_test.astype('float64')
        I_SR_HAT = (I_SR_HAT/255) * maximum
        I_SR_HAT = ((I_SR_HAT/0.5) * (maxi_val - mini_val)) + mini_val
        
        img_SR_HAT[:,:,bd] = I_SR_HAT[:,:,0]
        
stop = time.time()
time_SR_HAT = stop-start

if protocol == 'RR':
    Q2n_SR_HAT, Dlambda_SR_HAT, Q_SR_HAT, ERGAS_SR_HAT, SAM_SR_HAT, sCC_SR, RMSE_SR_HAT, PSNR_SR_HAT = \
        indexes_evaluation_SR(img_SR_HAT,img_LR,img_GT,ratio,radiance_range,GNyq_xs,GNyq_ys,max_img,Qblocks_size,flag_cut_bounds,dim_cut,th_values,K1,K2)
    
    sCC_SR_HAT = sCC_SR[0]
    
else:
    min_radiance = np.amin(img_SR_HAT) 
    max_radiance = np.amax(img_SR_HAT) 
    scaled_SR = (img_SR_HAT - min_radiance) / max_radiance
    
    scaled_SR = scaled_SR.squeeze()
    BRISQUE_SR_HAT = eng.brisque(scaled_SR)
    
if results:
    l.append(['HAT',Q2n_SR_HAT,Dlambda_SR_HAT,Q_SR_HAT,ERGAS_SR_HAT,SAM_SR_HAT,sCC_SR_HAT,RMSE_SR_HAT,PSNR_SR_HAT,time_SR_HAT]) 
    
    new = h5py.File(os.path.join(dirs_res_path + '/HAT.nc'), mode='w')
    new.create_dataset('radiance', data=img_SR_HAT, dtype=img_SR_HAT.dtype, shape=img_SR_HAT.shape)
    new.create_dataset('latitude', data=s5p_lat, dtype=s5p_lat.dtype, shape=s5p_lat.shape)
    new.create_dataset('longitude', data=s5p_lon, dtype=s5p_lon.dtype, shape=s5p_lon.shape)
    new.close()

#%%
"""S5Net"""

n1 = 64;n2 = 32;n3 = n2;f1 = 9;f2 = 5;f3 = 5;c = 1

epochs_path = f'./trained_models/S5Net/S5Net/{im_tag}/'

img_SR_S5Net = np.zeros([img_GT.shape[0],img_GT.shape[1],img_GT.shape[2]])
    
tc = 1    
ker = kernel(tc,ratio,'cubic')
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

start = time.time()

for bd in range(0,img_GT.shape[2]):
    
    band_epochs_path = epochs_path + f'band_{bd}/x{ratio}.pth'
    
    LR = img_LR[:,:,bd]
    LR1 = np.zeros([LR.shape[0],LR.shape[1],1])
    LR1[:,:,0] = LR
    LR = LR1
    
    maxi_val = np.amax(LR) 
    mini_val = np.amin(LR) 

    LR = ((LR - mini_val) / (maxi_val - mini_val)) * 0.5
    
    LR = np.transpose(LR, (2, 0, 1))

    lr_test = []
    lr_test.append(LR[:,:,:])
    lr_test = np.array(lr_test)

    preds = np.zeros([1,lr_test.shape[1],lr_test.shape[2]*ratio,lr_test.shape[3]*ratio])
    
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
        
        model.load_state_dict(torch.load(band_epochs_path))
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
    I_SR = I_SR_test.astype('float64')
    
    I_SR = ((I_SR/0.5) * (maxi_val - mini_val)) + mini_val
    
    img_SR_S5Net[:,:,bd] = I_SR[:,:,0] 
    
stop = time.time()

time_SR_S5Net = stop-start

if protocol == 'RR':
    Q2n_SR_S5Net, Dlambda_SR_S5Net, Q_SR_S5Net, ERGAS_SR_S5Net, SAM_SR_S5Net, sCC_SR, RMSE_SR_S5Net, PSNR_SR_S5Net = \
        indexes_evaluation_SR(img_SR_S5Net,img_LR,img_GT,ratio,radiance_range,GNyq_xs,GNyq_ys,max_img,Qblocks_size,flag_cut_bounds,dim_cut,th_values,K1,K2)
    
    sCC_SR_S5Net = sCC_SR[0]

else:
    min_radiance = np.amin(img_SR_S5Net) 
    max_radiance = np.amax(img_SR_S5Net) 
    scaled_SR = (img_SR_S5Net - min_radiance) / max_radiance
    
    scaled_SR = scaled_SR.squeeze()
    BRISQUE_SR_S5Net = eng.brisque(scaled_SR)
    
if results:
    l.append(['S5Net',Q2n_SR_S5Net,Dlambda_SR_S5Net,Q_SR_S5Net,ERGAS_SR_S5Net,SAM_SR_S5Net,sCC_SR_S5Net,RMSE_SR_S5Net,PSNR_SR_S5Net,time_SR_S5Net]) 
    
    new = h5py.File(os.path.join(dirs_res_path + '/S5Net.nc'), mode='w')
    new.create_dataset('radiance', data=img_SR_S5Net, dtype=img_SR_S5Net.dtype, shape=img_SR_S5Net.shape)
    new.create_dataset('latitude', data=s5p_lat, dtype=s5p_lat.dtype, shape=s5p_lat.shape)
    new.create_dataset('longitude', data=s5p_lon, dtype=s5p_lon.dtype, shape=s5p_lon.shape)
    new.close()

#%%
"""GSR-S5Net-st"""

n1 = 64;n2 = 32;n3 = n2;f1 = 9;f2 = 5;f3 = 5;c = 1

epochs_path = f'./trained_models/S5Net/GSR-S5Net-st/{im_tag}/'

img_SR_S5Net = np.zeros([img_GT.shape[0],img_GT.shape[1],img_GT.shape[2]])
    
tc = 1    
ker = kernel(tc,ratio,'cubic')
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

start = time.time()

for bd in range(0,img_GT.shape[2]):
    
    band_epochs_path = epochs_path + f'band_{bd}/x{ratio}.pth'
    
    LR = img_LR[:,:,bd]
    LR1 = np.zeros([LR.shape[0],LR.shape[1],1])
    LR1[:,:,0] = LR
    LR = LR1
    
    maxi_val = np.amax(LR) 
    mini_val = np.amin(LR) 

    LR = ((LR - mini_val) / (maxi_val - mini_val)) * 0.5
    
    LR = np.transpose(LR, (2, 0, 1))

    lr_test = []
    lr_test.append(LR[:,:,:])
    lr_test = np.array(lr_test)

    preds = np.zeros([1,lr_test.shape[1],lr_test.shape[2]*ratio,lr_test.shape[3]*ratio])
    
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
        
        model.load_state_dict(torch.load(band_epochs_path))
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
    I_SR = I_SR_test.astype('float64')
    
    I_SR = ((I_SR/0.5) * (maxi_val - mini_val)) + mini_val
    
    img_SR_DSR_S5Net_st[:,:,bd] = I_SR[:,:,0] 
    
stop = time.time()

time_SR_DSR_S5Net_st = stop-start

if protocol == 'RR':
    Q2n_SR_DSR_S5Net_st, Dlambda_SR_DSR_S5Net_st, Q_SR_DSR_S5Net_st, ERGAS_SR_DSR_S5Net_st, SAM_SR_DSR_S5Net_st, sCC_SR, RMSE_SR_DSR_S5Net_st, PSNR_SR_DSR_S5Net_st = \
        indexes_evaluation_SR(img_SR_DSR_S5Net_st,img_LR,img_GT,ratio,radiance_range,GNyq_xs,GNyq_ys,max_img,Qblocks_size,flag_cut_bounds,dim_cut,th_values,K1,K2)
    
    sCC_SR_DSR_S5Net_st = sCC_SR[0]

else:
    min_radiance = np.amin(img_SR_DSR_S5Net_st) 
    max_radiance = np.amax(img_SR_DSR_S5Net_st) 
    scaled_SR = (img_SR_DSR_S5Net_st - min_radiance) / max_radiance
    
    scaled_SR = scaled_SR.squeeze()
    BRISQUE_SR_DSR_S5Net_st = eng.brisque(scaled_SR)
    
if results:
    l.append(['GSR_S5Net_st',Q2n_SR_DSR_S5Net_st,Dlambda_SR_DSR_S5Net_st,Q_SR_DSR_S5Net_st,ERGAS_SR_DSR_S5Net_st,SAM_SR_DSR_S5Net_st,sCC_SR_DSR_S5Net_st,RMSE_SR_DSR_S5Net_st,PSNR_SR_DSR_S5Net_st,time_SR_DSR_S5Net_st]) 
    
    new = h5py.File(os.path.join(dirs_res_path + '/S5Net.nc'), mode='w')
    new.create_dataset('radiance', data=img_SR_DSR_S5Net_st, dtype=img_SR_DSR_S5Net_st.dtype, shape=img_SR_DSR_S5Net_st.shape)
    new.create_dataset('latitude', data=s5p_lat, dtype=s5p_lat.dtype, shape=s5p_lat.shape)
    new.create_dataset('longitude', data=s5p_lon, dtype=s5p_lon.dtype, shape=s5p_lon.shape)
    new.close()
    
#%%
"""GSR-S5Net-dyn"""

n1 = 64;n2 = 32;n3 = n2;f1 = 9;f2 = 5;f3 = 5;c = 1

epochs_path = f'./trained_models/S5Net/GSR-S5Net-dyn/{im_tag}/'

img_SR_S5Net = np.zeros([img_GT.shape[0],img_GT.shape[1],img_GT.shape[2]])
    
tc = 1    
ker = kernel(tc,ratio,'cubic')
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

start = time.time()

for bd in range(0,img_GT.shape[2]):
    
    band_epochs_path = epochs_path + f'band_{bd}/x{ratio}.pth'
    
    LR = img_LR[:,:,bd]
    LR1 = np.zeros([LR.shape[0],LR.shape[1],1])
    LR1[:,:,0] = LR
    LR = LR1
    
    maxi_val = np.amax(LR) 
    mini_val = np.amin(LR) 

    LR = ((LR - mini_val) / (maxi_val - mini_val)) * 0.5
    
    LR = np.transpose(LR, (2, 0, 1))

    lr_test = []
    lr_test.append(LR[:,:,:])
    lr_test = np.array(lr_test)

    preds = np.zeros([1,lr_test.shape[1],lr_test.shape[2]*ratio,lr_test.shape[3]*ratio])
    
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
        
        model.load_state_dict(torch.load(band_epochs_path))
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
    I_SR = I_SR_test.astype('float64')
    
    I_SR = ((I_SR/0.5) * (maxi_val - mini_val)) + mini_val
    
    img_SR_DSR_S5Net_dyn[:,:,bd] = I_SR[:,:,0] 
    
stop = time.time()

time_SR_DSR_S5Net_dyn = stop-start

if protocol == 'RR':
    Q2n_SR_DSR_S5Net_dyn, Dlambda_SR_DSR_S5Net_dyn, Q_SR_DSR_S5Net_dyn, ERGAS_SR_DSR_S5Net_dyn, SAM_SR_DSR_S5Net_dyn, sCC_SR, RMSE_SR_DSR_S5Net_dyn, PSNR_SR_DSR_S5Net_dyn = \
        indexes_evaluation_SR(img_SR_DSR_S5Net_dyn,img_LR,img_GT,ratio,radiance_range,GNyq_xs,GNyq_ys,max_img,Qblocks_size,flag_cut_bounds,dim_cut,th_values,K1,K2)
    
    sCC_SR_DSR_S5Net_dyn = sCC_SR[0]

else:
    min_radiance = np.amin(img_SR_DSR_S5Net_dyn) 
    max_radiance = np.amax(img_SR_DSR_S5Net_dyn) 
    scaled_SR = (img_SR_DSR_S5Net_dyn - min_radiance) / max_radiance
    
    scaled_SR = scaled_SR.squeeze()
    BRISQUE_SR_DSR_S5Net_dyn = eng.brisque(scaled_SR)
    
if results:
    l.append(['GSR_S5Net_dyn',Q2n_SR_DSR_S5Net_dyn,Dlambda_SR_DSR_S5Net_dyn,Q_SR_DSR_S5Net_dyn,ERGAS_SR_DSR_S5Net_dyn,SAM_SR_DSR_S5Net_dyn,sCC_SR_DSR_S5Net_dyn,RMSE_SR_DSR_S5Net_dyn,PSNR_SR_DSR_S5Net_dyn,time_SR_DSR_S5Net_dyn]) 
    
    new = h5py.File(os.path.join(dirs_res_path + '/S5Net.nc'), mode='w')
    new.create_dataset('radiance', data=img_SR_DSR_S5Net_dyn, dtype=img_SR_DSR_S5Net_dyn.dtype, shape=img_SR_DSR_S5Net_dyn.shape)
    new.create_dataset('latitude', data=s5p_lat, dtype=s5p_lat.dtype, shape=s5p_lat.shape)
    new.create_dataset('longitude', data=s5p_lon, dtype=s5p_lon.dtype, shape=s5p_lon.shape)
    new.close()

#%%
"""DSR-S5Net-st"""

n1 = 64;n2 = 32;n3 = n2;f1 = 9;f2 = 5;f3 = 5;c = 1

epochs_path = f'./trained_models/S5Net/DSR-S5Net-st/{im_tag}/'

img_SR_S5Net = np.zeros([img_GT.shape[0],img_GT.shape[1],img_GT.shape[2]])
    
tc = 1    
ker = kernel(tc,ratio,'cubic')
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

start = time.time()

for bd in range(0,img_GT.shape[2]):
    
    band_epochs_path = epochs_path + f'band_{bd}/x{ratio}.pth'
    
    LR = img_LR[:,:,bd]
    LR1 = np.zeros([LR.shape[0],LR.shape[1],1])
    LR1[:,:,0] = LR
    LR = LR1
    
    maxi_val = np.amax(LR) 
    mini_val = np.amin(LR) 

    LR = ((LR - mini_val) / (maxi_val - mini_val)) * 0.5
    
    LR = np.transpose(LR, (2, 0, 1))

    lr_test = []
    lr_test.append(LR[:,:,:])
    lr_test = np.array(lr_test)

    preds = np.zeros([1,lr_test.shape[1],lr_test.shape[2]*ratio,lr_test.shape[3]*ratio])
    
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
        
        model.load_state_dict(torch.load(band_epochs_path))
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
    I_SR = I_SR_test.astype('float64')
    
    I_SR = ((I_SR/0.5) * (maxi_val - mini_val)) + mini_val
    
    img_SR_DSR_S5Net_st[:,:,bd] = I_SR[:,:,0] 
    
stop = time.time()

time_SR_DSR_S5Net_st = stop-start

if protocol == 'RR':
    Q2n_SR_DSR_S5Net_st, Dlambda_SR_DSR_S5Net_st, Q_SR_DSR_S5Net_st, ERGAS_SR_DSR_S5Net_st, SAM_SR_DSR_S5Net_st, sCC_SR, RMSE_SR_DSR_S5Net_st, PSNR_SR_DSR_S5Net_st = \
        indexes_evaluation_SR(img_SR_DSR_S5Net_st,img_LR,img_GT,ratio,radiance_range,GNyq_xs,GNyq_ys,max_img,Qblocks_size,flag_cut_bounds,dim_cut,th_values,K1,K2)
    
    sCC_SR_DSR_S5Net_st = sCC_SR[0]

else:
    min_radiance = np.amin(img_SR_DSR_S5Net_st) 
    max_radiance = np.amax(img_SR_DSR_S5Net_st) 
    scaled_SR = (img_SR_DSR_S5Net_st - min_radiance) / max_radiance
    
    scaled_SR = scaled_SR.squeeze()
    BRISQUE_SR_DSR_S5Net_st = eng.brisque(scaled_SR)
    
if results:
    l.append(['DSR_S5Net_st',Q2n_SR_DSR_S5Net_st,Dlambda_SR_DSR_S5Net_st,Q_SR_DSR_S5Net_st,ERGAS_SR_DSR_S5Net_st,SAM_SR_DSR_S5Net_st,sCC_SR_DSR_S5Net_st,RMSE_SR_DSR_S5Net_st,PSNR_SR_DSR_S5Net_st,time_SR_DSR_S5Net_st]) 
    
    new = h5py.File(os.path.join(dirs_res_path + '/S5Net.nc'), mode='w')
    new.create_dataset('radiance', data=img_SR_DSR_S5Net_st, dtype=img_SR_DSR_S5Net_st.dtype, shape=img_SR_DSR_S5Net_st.shape)
    new.create_dataset('latitude', data=s5p_lat, dtype=s5p_lat.dtype, shape=s5p_lat.shape)
    new.create_dataset('longitude', data=s5p_lon, dtype=s5p_lon.dtype, shape=s5p_lon.shape)
    new.close()
    
#%%
"""DSR-S5Net-dyn"""

n1 = 64;n2 = 32;n3 = n2;f1 = 9;f2 = 5;f3 = 5;c = 1

epochs_path = f'./trained_models/S5Net/DSR-S5Net-dyn/{im_tag}/'

img_SR_S5Net = np.zeros([img_GT.shape[0],img_GT.shape[1],img_GT.shape[2]])
    
tc = 1    
ker = kernel(tc,ratio,'cubic')
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

start = time.time()

for bd in range(0,img_GT.shape[2]):
    
    band_epochs_path = epochs_path + f'band_{bd}/x{ratio}.pth'
    
    LR = img_LR[:,:,bd]
    LR1 = np.zeros([LR.shape[0],LR.shape[1],1])
    LR1[:,:,0] = LR
    LR = LR1
    
    maxi_val = np.amax(LR) 
    mini_val = np.amin(LR) 

    LR = ((LR - mini_val) / (maxi_val - mini_val)) * 0.5
    
    LR = np.transpose(LR, (2, 0, 1))

    lr_test = []
    lr_test.append(LR[:,:,:])
    lr_test = np.array(lr_test)

    preds = np.zeros([1,lr_test.shape[1],lr_test.shape[2]*ratio,lr_test.shape[3]*ratio])
    
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
        
        model.load_state_dict(torch.load(band_epochs_path))
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
    I_SR = I_SR_test.astype('float64')
    
    I_SR = ((I_SR/0.5) * (maxi_val - mini_val)) + mini_val
    
    img_SR_DSR_S5Net_dyn[:,:,bd] = I_SR[:,:,0] 
    
stop = time.time()

time_SR_DSR_S5Net_dyn = stop-start

if protocol == 'RR':
    Q2n_SR_DSR_S5Net_dyn, Dlambda_SR_DSR_S5Net_dyn, Q_SR_DSR_S5Net_dyn, ERGAS_SR_DSR_S5Net_dyn, SAM_SR_DSR_S5Net_dyn, sCC_SR, RMSE_SR_DSR_S5Net_dyn, PSNR_SR_DSR_S5Net_dyn = \
        indexes_evaluation_SR(img_SR_DSR_S5Net_dyn,img_LR,img_GT,ratio,radiance_range,GNyq_xs,GNyq_ys,max_img,Qblocks_size,flag_cut_bounds,dim_cut,th_values,K1,K2)
    
    sCC_SR_DSR_S5Net_dyn = sCC_SR[0]

else:
    min_radiance = np.amin(img_SR_DSR_S5Net_dyn) 
    max_radiance = np.amax(img_SR_DSR_S5Net_dyn) 
    scaled_SR = (img_SR_DSR_S5Net_dyn - min_radiance) / max_radiance
    
    scaled_SR = scaled_SR.squeeze()
    BRISQUE_SR_DSR_S5Net_dyn = eng.brisque(scaled_SR)
    
if results:
    l.append(['DSR_S5Net_dyn',Q2n_SR_DSR_S5Net_dyn,Dlambda_SR_DSR_S5Net_dyn,Q_SR_DSR_S5Net_dyn,ERGAS_SR_DSR_S5Net_dyn,SAM_SR_DSR_S5Net_dyn,sCC_SR_DSR_S5Net_dyn,RMSE_SR_DSR_S5Net_dyn,PSNR_SR_DSR_S5Net_dyn,time_SR_DSR_S5Net_dyn]) 
    
    new = h5py.File(os.path.join(dirs_res_path + '/S5Net.nc'), mode='w')
    new.create_dataset('radiance', data=img_SR_DSR_S5Net_dyn, dtype=img_SR_DSR_S5Net_dyn.dtype, shape=img_SR_DSR_S5Net_dyn.shape)
    new.create_dataset('latitude', data=s5p_lat, dtype=s5p_lat.dtype, shape=s5p_lat.shape)
    new.create_dataset('longitude', data=s5p_lon, dtype=s5p_lon.dtype, shape=s5p_lon.shape)
    new.close()