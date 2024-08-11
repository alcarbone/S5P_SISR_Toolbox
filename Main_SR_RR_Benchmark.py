# -*- coding: utf-8 -*-
"""
Copyright (c) 
All rights reserved. This work should only be used for nonprofit purposes.

@author: 
    Alessia Carbone (alcarbone@unisa.it)
"""

"""
 Description:
           Choose the configuration:
               * ratio: scaling ratio for testing super-resolution approaches;
               * im_tag: name of the chosen image;
               * protocol: RR or FR protocol;
               * flag_cut_bounds: if 0 image borders are not cropped;
               * dim_cut: if flag_cut_bounds == 1, number of pixels cropped from 
                   each border;
               * results: if True all the results and images are saved in the 
                   directory ./results as .csv and .nc files, respectively.
"""

from IPython import get_ipython
get_ipython().run_line_magic('reset','-sf')

import os
import numpy as np
import scipy.io as sio

from scripts.resize_image import resize_image

#%%
"""Configuration"""

ratio = 4

im_tag = 'US'
protocol = 'RR'

flag_cut_bounds = 1
dim_cut = 15

results = True

if results:
    dirs_res_path = f'./results/x{ratio}/{protocol}/{im_tag}'
    
    if not os.path.isdir(dirs_res_path):
        os.makedirs(dirs_res_path)       

execfile('SR_algorithms.py')

if results:
    wr.writerows(l)
    f.close()