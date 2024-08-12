# -*- coding: utf-8 -*-
"""
Copyright (c) 
All rights reserved. This work should only be used for nonprofit purposes.

@author: Alessia Carbone (alcarbone@unisa.it)

"""

"""

Description:
    This file contains all utility functions to read and preprocess S5P 
    level-1B radiance data. 
             
"""

import geopy
import numpy as np
import matplotlib.pyplot as plt

#%%
""" 
    Computes the difference between adjacent single coordinates in a grid for
    a given direction ('across' is along the columns and 'along' is along
    the rows). 
    * Latitudes and longitudes diffence should be constant in both directions.
    * Latitudes difference must be zero across-track.
    * Longitudes difference must be zero along-track.
"""

def coord_interval(coord,direction):
    if direction == 'across':
        distance = np.zeros([coord.shape[0],coord.shape[1]-1])
        for ii in range(coord.shape[0]):
            for jj in range(1,coord.shape[1]):
                distance[ii,jj-1] = abs(coord[ii,jj-1]-coord[ii,jj])
    else:
        distance = np.zeros([coord.shape[0]-1,coord.shape[1]])
        for ii in range(coord.shape[1]):
            for jj in range(1,coord.shape[0]):
                distance[jj-1,ii] = abs(coord[jj-1,ii]-coord[jj,ii])
    return distance

#%%
""" 
    Computes the difference between adjacent single coordinates in a grid 
    for a given direction ('across' is along the columns and 'along' is along 
    the rows). This distance in km is the distance between the center of two 
    adjancent pixels in a given direction (the spatial resolution).
"""

def coord_diff_km(coord1,coord2,direction):
    if direction == 'across':
        distance = np.zeros([coord1.shape[0],coord1.shape[1]-1])
        for ii in range(coord1.shape[0]):
            for jj in range(1,coord1.shape[1]):
                point1 = (coord1[ii,jj-1],coord2[ii,jj-1])
                point2 = (coord1[ii,jj],coord2[ii,jj])
                distance[ii,jj-1] = geopy.distance.geodesic(point1, point2).km
    else:
        distance = np.zeros([coord1.shape[0]-1,coord1.shape[1]])
        for ii in range(coord1.shape[1]):
            for jj in range(1,coord1.shape[0]):
                point1 = (coord1[jj-1,ii],coord2[jj-1,ii])
                point2 = (coord1[jj,ii],coord2[jj,ii])
                distance[jj-1,ii] = geopy.distance.geodesic(point1, point2).km
    return distance

#%%
"""
    This function computes the distance between two points, given their 
    latitudes and longitudes using the Haversine formula which is used to 
    compute distances on a sphere. 
"""

def hav_dist(lat, lon):
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    if (len(lat.shape) == 3):
        lat1 = lat[:, :, 1:]
        lat2 = lat[:, :, 0:-1]
        lon1 = lon[:, :, 1:]
        lon2 = lon[:, :, 0:-1]
    else:
        lat1 = lat[:, 1:]
        lat2 = lat[:, 0:-1]
        lon1 = lon[:, 1:]
        lon2 = lon[:, 0:-1]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    radius_of_earth = 6371.0  # Radius of the Earth in kilometers

    distance = radius_of_earth * c

    return distance

#%%
""" 
    Preprocesses a NetCDF4 file. The images are all cropped by considering
    the binning table and a maximum resolution. The area of the swath taken 
    under consideration is the area in which the binning factor is equal to 
    2 (or 1 for bands 7 and 8) and the distance in km is at most equal to 
    max_res. All values equal to the fill value are set to zero and all 
    values lower than zero are set to zero. The values are flipped along 
    the rows because the satellite travels from the south to the north pole.
"""

def read_s5p(s5p, max_res):
    key = [x for x in s5p.keys() if x.__contains__("BAND")][0]
    rad = s5p[key]['STANDARD_MODE']['OBSERVATIONS']['radiance']
    fill_value = rad.attrs['_FillValue']
    s5p_lat = s5p[key]['STANDARD_MODE']['GEODATA']['latitude']
    s5p_lon = s5p[key]['STANDARD_MODE']['GEODATA']['longitude']
    instrument = s5p[key]['STANDARD_MODE']['INSTRUMENT']
    
    if 'measurement_to_detector_row_table' in instrument.keys():
        measurement_to_detector = instrument['measurement_to_detector_row_table']
        bin_fact = measurement_to_detector['det_end_row'] - measurement_to_detector['det_start_row']
        bin_fact = bin_fact[:,:,1:]

        if not (key.__contains__('BAND7') or key.__contains__('BAND8')):
            dist = hav_dist(s5p_lat, s5p_lon) <= max_res
            crop = (bin_fact == 2) * dist
        else:
            crop = bin_fact==1
        indices = np.where(np.prod(crop > 0, axis = 1)>0)

        min_col = np.min(indices[1])
        max_col = np.max(indices[1])
        rad = rad[:,:,min_col:max_col,:]
        s5p_lon = s5p_lon[:, :, min_col:max_col]
        s5p_lat = s5p_lat[:, :, min_col:max_col]
        
    rad = np.squeeze(rad)
    s5p_lon = np.squeeze(s5p_lon)
    s5p_lat = np.squeeze(s5p_lat)
    
    rad[rad < 0] = 0
    rad[rad == fill_value] = 0
    
    s5p_lat = np.flip(s5p_lat,axis=0)
    s5p_lon = np.flip(s5p_lon,axis=0)
    rad = np.flip(rad,axis=0)
        
    return rad,s5p_lat,s5p_lon

#%%
"""
    Given an area of interest (lat1,lat2), from north to south pole,
    the function crops the image along-track.
"""

def crop_s5p(rad,lat,lon,area_of_interest):
    for rr in range(lat.shape[0]):
        if lat[rr,0] <= area_of_interest[0]:
            start_lat = rr
            break
    for rr in range(lat.shape[0]):
        if lat[rr,-1] <= area_of_interest[1]:
            stop_lat = rr
            break
    rad = rad[start_lat:stop_lat,:,:]
    lat = lat[start_lat:stop_lat,:]
    lon = lon[start_lat:stop_lat,:]
    
    return rad,lat,lon

#%%
"""
    Given the coordinates of the 4 points of the grid this function creates
    a uniformly spaced grid. Keep in mind: the resolution along-track (5.5 km) 
    is almost the same for the whole image (a step between latitudes of 0.0496), 
    but in across-track it is not the same. Here we chose a resolution 
    across-track of 3.5 km (a step of 0.0316).
"""

def create_grid(first_lat,first_lon,last_lat,last_lon,lon_res):
             
    lat_step = 0.0496 #5.5 km
    lon_step = round((lon_res * lat_step) / 5.5, 4)
    
    new_dim_across = int(abs(first_lon-last_lon)/lon_step)
    new_dim_along = int(abs(first_lat-last_lat)/lat_step) 
    
    new_dim = [new_dim_along,new_dim_across]
    
    grid_lat = np.zeros(new_dim)
    grid_lat[0,:] = first_lat
    for rr in range(1,new_dim[0]):
        grid_lat[rr,:] = grid_lat[rr-1,:] - lat_step
        
    grid_lon = np.zeros(new_dim)
    grid_lon[:,0] = first_lon 
    for cc in range(1,new_dim[1]):
        grid_lon[:,cc] = grid_lon[:,cc-1] + lon_step
    
    return grid_lat,grid_lon

#%%
""" 
    Resample image given a fixed grid and a method. 
"""

def resample_img(orig_rad,orig_lat,orig_lon,new_lat,new_lon,method):
    if method == 'pyresample_nearest':
        new_rad = resample_img_pyresample(orig_rad,orig_lat,orig_lon,new_lat,new_lon,'nearest')
    elif method == 'pyresample_bilinear':
        new_rad = resample_img_pyresample(orig_rad,orig_lat,orig_lon,new_lat,new_lon,'bilinear')
    elif method == 'sinc':
        neighbours = 5
        new_rad = resample_img_sinc(orig_rad,orig_lat,orig_lon,new_lat,new_lon,neighbours)
    elif method == 'matlab':
        interpolation = 'natural'
        new_rad = resample_img_matlab(orig_rad,orig_lat,orig_lon,new_lat,new_lon,interpolation)
    else:
        Exception('Resampling method not available!')
    return new_rad

#%%
"""
    Resample image with pyresample library 
    (https://buildmedia.readthedocs.org/media/pdf/pyresample/develop/pyresample.pdf ).
    For the bilinear case the Proj object must be used to convert latitudes and longitudes
    grids in map proxection (x,y) coordinates 
    (available at https://pyproj4.github.io/pyproj/stable/api/proj.html ).
"""

def resample_img_pyresample(orig_rad,orig_lat,orig_lon,new_lat,new_lon,method):
    if method == 'nearest':
        from pyresample.geometry import SwathDefinition, GridDefinition
        from pyresample.kd_tree import resample_nearest
        
        old_grid = SwathDefinition(lons=orig_lon, lats=orig_lat)
        new_grid = GridDefinition(lons=new_lon, lats=new_lat)
        
        new_rad = resample_nearest(old_grid, orig_rad, new_grid, radius_of_influence=20000, fill_value=None)
    elif method == 'bilinear':
        from pyproj import Proj 
        from pyresample.geometry import SwathDefinition, AreaDefinition
        from pyresample import bilinear
    
        old_grid = SwathDefinition(lons=orig_lon, lats=orig_lat)
        p = Proj("EPSG:32667", preserve_units=False)
        x,y = p(new_lon,new_lat)
        new_grid = AreaDefinition('target_area','The area definition is based on target latitudes and longitudes',
                                  'projection_string',p.definition_string(),
                                  x.shape[1],x.shape[0],[x[-1,0],y[-1,0],x[0,-1],y[0,-1]])
        
        new_rad = bilinear.resample_bilinear(orig_rad, old_grid, new_grid)
    else:
        Exception('Resampling method not available!')
    
    new_rad = np.asarray(new_rad)
    
    return new_rad

#%%
""" 
    Resample image with sinc function passing trough a certain number of points. 
"""

def resample_img_sinc(orig_rad,orig_lat,orig_lon,new_lat,new_lon,points): #TODO

    return None

#%%
""" 
    Resample image with matlab scatteredInterpolant function (available at
    https://it.mathworks.com/help/matlab/ref/scatteredinterpolant.html ). 
    If you want to use this function you must activate your MATLAB license.
"""

def resample_img_matlab(orig_rad,orig_lat,orig_lon,new_lat,new_lon,method):
    import matlab.engine
    eng = matlab.engine.start_matlab()
    
    new_dim = [new_lat.shape[0],new_lon.shape[1],orig_rad.shape[2]]
    new_rad = np.zeros(new_dim)
    
    new_lon = eng.transpose(np.float64(new_lon.flatten()).T)
    new_lat = eng.transpose(np.float64(new_lat.flatten()).T)
    orig_lon = eng.transpose(np.float64(orig_lon.flatten()).T)
    orig_lat = eng.transpose(np.float64(orig_lat.flatten()).T)
    
    for bb in range(orig_rad.shape[2]):
        orig_rad_band = eng.transpose(np.float64(orig_rad[:,:,bb].flatten()).T)
        F = eng.scatteredInterpolant(orig_lon,orig_lat,orig_rad_band,method)
        eng.workspace['F'] = F
        eng.workspace['new_lon'] = new_lon
        eng.workspace['new_lat'] = new_lat
        interp_rad = np.asarray(eng.eval('F(new_lon,new_lat)')) 
        interp_rad = np.reshape(np.ravel(interp_rad),(new_dim[0],new_dim[1]))
        new_rad[:,:,bb] = interp_rad
    
    return new_rad

#%%
"""
    This function pads an image given a number of points to add at the borders
    and a padding method.
"""

def padding_image(im, points, padding):
    pad = points-1
    
    if (len(im.shape) != 3):
        im11 = np.zeros([im.shape[0],im.shape[1],1])
        im11[:,:,0] = im[:,:]
        im = im11
        
    im1 = np.zeros([im.shape[0] + pad,im.shape[1] + pad,im.shape[2]])
    
    im1[pad//2:im1.shape[0]-pad//2,pad//2:im1.shape[1]-pad//2,:] = im
    
    if padding == 'zeros':
        return im1.squeeze()    
    elif padding == 'replicate':
        for ii in range(pad//2):
            im1[pad//2:im1.shape[0]-pad//2,ii,:] = im[:,0,:]
        for ii in range(im.shape[1]+1,im1.shape[1]):
            im1[pad//2:im1.shape[0]-pad//2,ii,:] = im[:,-1,:]
        for ii in range(pad//2):
            im1[ii,pad//2:im1.shape[1]-pad//2,:] = im[0,:,:] 
        for ii in range(im.shape[0]+1,im1.shape[0]):
            im1[ii,pad//2:im1.shape[1]-pad//2,:] = im[-1,:,:]
        for ii in range(pad//2):
            for jj in range(pad//2):
                im1[ii,jj,:] = im[0,0,:]
        for ii in range(pad//2):
            for jj in range(im.shape[1]+1,im1.shape[1]):
                im1[ii,jj,:] = im[0,-1,:]
        for ii in range(im.shape[0]+1,im1.shape[0]):
            for jj in range(pad//2):
                im1[ii,jj,:] = im[-1,0,:]                
        for ii in range(im.shape[0]+1,im1.shape[0]):
            for jj in range(im.shape[1]+1,im1.shape[1]):
                im1[ii,jj,:] = im[-1,-1,:]
        return im1.squeeze()  