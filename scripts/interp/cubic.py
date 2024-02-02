# -*- coding: utf-8 -*-
"""
Copyright (c) 
All rights reserved. This work should only be used for nonprofit purposes.

@author: 
    Alessia Carbone (alcarbone@unisa.it)
"""

"""
 Description: 
     cubic kernel definition.
 
     Interface: y = cubic(s,a)
    
     Inputs: s: x coordinate.
             a: A2 in [1].
     
     Outputs: y: y coordinate.
     
  References:
     [1] Keys, Robert «Cubic convolution interpolation for digital image processing». IEEE Transactions on Acoustics, Speech, and Signal Processing 29, fasc. 6 (dicembre 1981): 1153–60. https://doi.org/10.1109/TASSP.1981.1163711.
     [2] Aiazzi, Bruno, Stefano Baronti, Massimo Selva, e Luciano Alparone. «Bi-cubic interpolation for shift-free pan-sharpening». ISPRS Journal of Photogrammetry and Remote Sensing 86 (1 dicembre 2013): 65–76. https://doi.org/10.1016/j.isprsjprs.2013.09.007.     
"""

def cubic(s, a=-0.5):
    if (abs(s) <= 1):
        return (a + 2) * (abs(s) ** 3) - (a + 3) * (abs(s) ** 2) + 1
    elif ((abs(s) > 1) & (abs(s) <= 2)):
        return (a) * (abs(s) ** 3) - (5*a) * (abs(s) ** 2) + (8*a) * abs(s) - (4*a)
    return 0