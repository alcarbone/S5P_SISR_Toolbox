# -*- coding: utf-8 -*-
"""
Copyright (c) 
All rights reserved. This work should only be used for nonprofit purposes.

@author: Alessia Carbone (alcarbone@unisa.it)

"""

"""

Description:
    This file reads and pre-processes S5P level-1B radiance data. The orbits
    are resampled on a fixed grid and cropped in the given coordinates.
    All intermediate steps can be saved in a given directory.

"""

import h5py
import os
import numpy as np

from utils import *

home_path = fr"" # insert the path where the dataset is located

date = '20230401';area_of_interest = [29.0, 4.0];im_tag = 'IN'
#date = '20230709';area_of_interest = [48.0, 23.0];im_tag = 'US'
#date = '20240804';area_of_interest = [36.0, 11.0];im_tag = 'EG'

#%%
"""
    Read radiance data (s5p_path can be both a file and a directory). 
    The maximum resolution max_res is the maximum_resolution allowed 
    across_track (if it is sufficiently low, the borders are discarded). 
"""

s5p_path = os.path.join(home_path,fr"Datasets\S5P\level-1b\raw\{date}\RA")

if os.path.isfile(s5p_path):
    s5p_files = [s5p_path]
else:
    s5p_files = os.listdir(s5p_path)
    s5p_files = [os.path.join(s5p_path,x) for x in s5p_files]
           
s5p_lat = [[] for _ in range(len(s5p_files))] 
s5p_lon = [[] for _ in range(len(s5p_files))]
s5p_rad = [[] for _ in range(len(s5p_files))]

max_res = 8   
     
print('Starting reading Sentinel-5p tiles (radiances)')
for ind, s5p_file in enumerate(s5p_files,start=1):
    filename = os.path.split(s5p_file)[-1]
    print('Reading {}'.format(filename))
    print('Tile {}/{}'.format(ind,len(s5p_files)))
    s5_orig = h5py.File(s5p_file)
    s5p_rad[ind-1], s5p_lat[ind-1], s5p_lon[ind-1] = read_s5p(s5_orig, max_res)
    print('{} read'.format(filename))

#%%    
"""
    The images are cropped along-track in the area of interest (first
    latitude to keep from the North, last latitude to keep from the North).
    Longitudes are cropped when max_res is specified, i.e., in the previous cell.
"""
        
    
print('Starting cropping Sentinel-5p tiles along-track')
for ind, s5p_file in enumerate(s5p_files,start=1):
    filename = os.path.split(s5p_file)[-1]
    print('Cropping {}'.format(filename))
    print('Tile {}/{}'.format(ind,len(s5p_files)))
    s5p_rad[ind-1],s5p_lat[ind-1],s5p_lon[ind-1] = crop_s5p(s5p_rad[ind-1],s5p_lat[ind-1],s5p_lon[ind-1],area_of_interest)
    print('{} cropped'.format(filename))
        
#%%  
"""
    Create the biggest straight and uniform grids for both latitudes and longitudes 
    whose points are available in all the images (in other words, we do not allow 
    extrapolation of data).
    The longitude resolution can be chosen because it changes (band 1, 7, 8 have in fact
    different resolutions than other images across-track).
"""

first_lat = +90
last_lat = -90

for ii in range(len(s5p_lat)):
    if(s5p_lat[ii][0,0] < first_lat):
        first_lat = s5p_lat[ii][0,0]
    if(s5p_lat[ii][-1,-1] > last_lat):
        last_lat = s5p_lat[ii][-1,-1]
        
first_lon = -180
last_lon = +180

for ii in range(len(s5p_lon)):
    if(s5p_lon[ii][-1,0] > first_lon):
        first_lon = s5p_lon[ii][-1,0]
    if(s5p_lon[ii][0,-1] < last_lon):
        last_lon = s5p_lon[ii][0,-1]
        
lon_res = 3.5 #in km
    
grid_lat,grid_lon = create_grid(first_lat,first_lon,last_lat,last_lon,lon_res)

#%%   
"""
    Resample the image in the given grid given the methodology to use for
    resampling:
        - pyresample_nearest (pyresample.kd_tree.resample_nearest available at https://buildmedia.readthedocs.org/media/pdf/pyresample/develop/pyresample.pdf ),
        - pyresample_bilinear (pyresample.bilinear.resample_bilinear available at https://buildmedia.readthedocs.org/media/pdf/pyresample/develop/pyresample.pdf ),
        - matlab (scatteredInterpolant available at https://it.mathworks.com/help/matlab/ref/scatteredinterpolant.html ). 
    The last option is the best one.
    Activate MATLAB license to use the last option.
"""

method = 'matlab' 

new_s5p_lat = [[] for _ in range(len(s5p_files))] 
new_s5p_lon = [[] for _ in range(len(s5p_files))]
new_s5p_rad = [[] for _ in range(len(s5p_files))]

print('Starting resampling Sentinel-5p tiles')
for ind, s5p_file in enumerate(s5p_files,start=1):
    filename = os.path.split(s5p_file)[-1]
    print('Resampling {}'.format(filename))
    print('Tile {}/{}'.format(ind,len(s5p_files)))
    new_s5p_rad[ind-1] = resample_img(s5p_rad[ind-1],s5p_lat[ind-1],s5p_lon[ind-1],grid_lat,grid_lon,method)
    new_s5p_lat[ind-1] = grid_lat
    new_s5p_lon[ind-1] = grid_lon
    print('{} resampled'.format(filename))   
    
#%%
"""
    Save the resampled images separetely in a given directory.
"""

output_dir = os.path.join(home_path,fr"Datasets\S5P\level-1b\processed_{method}")
name = os.path.split(s5p_files[0])[-1].split('.')[0]
acquisition = name.split('_')[5].split('T')[0]
output_dir = os.path.join(output_dir, acquisition)    
# output_dir = os.path.join(output_dir, 'SWIR_reduced')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print('Starting saving Sentinel-5p tiles')    
for ind, s5p_file in enumerate(s5p_files, start = 1):
    filename = os.path.split(s5p_file)[-1]
    print('Saving {}'.format(filename))
    print('Tile {}/{}'.format(ind,len(s5p_files)))
    new = h5py.File(os.path.join(output_dir, filename), mode='w')
    new.create_dataset('radiance', data=new_s5p_rad[ind-1], dtype=new_s5p_rad[ind-1].dtype, shape=new_s5p_rad[ind-1].shape)
    new.create_dataset('latitude', data=new_s5p_lat[ind-1], dtype=new_s5p_lat[ind-1].dtype, shape=new_s5p_lat[ind-1].shape)
    new.create_dataset('longitude', data=new_s5p_lon[ind-1], dtype=new_s5p_lon[ind-1].dtype, shape=new_s5p_lon[ind-1].shape)
    new.close()
    print('{} saved'.format(filename))
      
#%%
"""
    Put together all bands in a single image with a certain tag after resampling and save it.
"""

s5p = new_s5p_rad[0]
lat = new_s5p_lat[0]
lon = new_s5p_lon[0]

im = s5p
print('Starting putting together Sentinel-5p tiles')
for ii in range(1,len(s5p_files)):
    s5p_file = s5p_files[ii]
    s5p = new_s5p_rad[ii]
    im = np.append(im, s5p, axis = 2)
print('{} created'.format(im_tag))

#%%
"""
    Save stacked image in a given directory.
"""

output_dir = os.path.join(home_path,fr"Datasets\S5P\level-1b\images")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

new = h5py.File(os.path.join(output_dir, im_tag + '.nc'), mode='w')
new.create_dataset('radiance', data=im, dtype=im.dtype, shape=im.shape)
new.create_dataset('latitude', data=lat, dtype=lat.dtype, shape=lat.shape)
new.create_dataset('longitude', data=lon, dtype=lon.dtype, shape=lon.shape)
new.close()
print('{} saved'.format(im_tag))