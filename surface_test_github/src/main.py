#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 14:42:09 2023

@author: phialan
"""
import pims    

from calc import *

## pjt_slope15_prebuilt_062218, high fric baseline parameters
images = pims.ImageSequence('data/*.jpg')
xmin,xmax,ymin,ymax = 95, 5095,300, 1290 # pixel boundaries of image crop
scale=63. #pixels/cm spatial
prefix = 'pjt_slope15_prebuilt_062218'
im_w=images.frame_shape[1]
im_h=images.frame_shape[0]


# surface parameters        
surf_cutoff = 15 # px of change
hdf_storename = f'{prefix}_surfnocv_2023.h5'
from_larger = True

subsel_surface_generator_nocv(images,hdf_storename,xmin,xmax,ymin,ymax,
                              from_larger=True)