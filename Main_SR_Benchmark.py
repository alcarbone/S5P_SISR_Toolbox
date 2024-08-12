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
    
im_tag = 'US'
protocol = 'RR'

if results:
    dirs_res_path = f'./results/x{ratio}/{protocol}/{im_tag}'
    
    if not os.path.isdir(dirs_res_path):
        os.makedirs(dirs_res_path)       

execfile('SR_algorithms.py')

if results:
    wr.writerows(l)
    f.close()