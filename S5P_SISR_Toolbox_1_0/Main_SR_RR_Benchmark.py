# -*- coding: utf-8 -*-
"""
Copyright (c) 
All rights reserved. This work should only be used for nonprofit purposes.

@author: 
    Alessia Carbone (alcarbone@unisa.it)
"""

"""
 Description:
           Choose the configuration to test on.
"""

from IPython import get_ipython
get_ipython().run_line_magic('reset','-sf')

import os

#%%
""" Configuration """

ratio = 4

flag_cut_bounds = 1
dim_cut = 15

results = True

Qblocks_size = 32 
th_values = 0
flag_cut_bounds = 1
dim_cut = 15

L = 14

protocols = ['RR']
bands = ['UV','UVIS','NIR','SWIR']
im_tags = ['IN','US']

for protocol in protocols:
    for im_tag in im_tags:
        for band in bands:

            if (band == 'UV'):
                num = 1
            elif (band == 'UVIS'):
                num = 2
            elif (band == 'NIR'):
                num = 3
            else:
                num = 4
                
            dirs = './data{}.{}/'.format(num,band)
            
            if (band == 'UV'):
                GNyq_el = 0.36
                GNyq_az = 0.37
            elif (band == 'UVIS' or band == 'NIR'):
                GNyq_el = 0.74
                GNyq_az = 0.44
            else:
                GNyq_el = 0.20
                GNyq_az = 0.15
                
            file_name = dirs + '/' + im_tag + '.mat'
            ds = sio.loadmat(file_name)
            radiance_GT = np.array(ds['radiance']) 
            
            radiance_LR = resize_image(radiance_GT, ratio, GNyq_el, GNyq_az)
            radiance_GT = radiance_GT[0:(radiance_LR.shape[0]*ratio),0:(radiance_LR.shape[1]*ratio)]
            radiance_LR = np.squeeze(radiance_LR)
            
            GNyq_x = GNyq_el
            GNyq_y = GNyq_az
            
            max_val = np.amax(radiance_LR) 
            radiance_range = [0, 2*max_val]
            
            if results:
                dirs_res_path = './results/x{}'.format(ratio) + '/' + protocol + '/{}.{}'.format(num,band)
                
                if not os.path.isdir(dirs_res_path):
                    os.makedirs(dirs_res_path)
            
            execfile('SR_algorithms_1_0.py')