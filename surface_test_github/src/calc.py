#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 14:47:35 2023

@author: phialan
"""
from scipy.ndimage import binary_fill_holes
from skimage import color, io, filters, measure
import pandas as pd
import numpy as np
import pims
from scipy.signal import medfilt
# import glob
# import time

def subsel_surface_generator_nocv(images,hdf_storename,
                                  xmin,xmax,ymin,ymax,
                                  surf_cutoff = 15,
                                  plate_y_pos=700, 
                                  from_larger=False, plate=False, 
                                  medwindow=31):
    '''
    Calculate wedge surface location for an entire wedge experiment, without 
        using OpenCV, as it can be hard to install on some/most systems,
        and with all required functionality handled by skimage, scipy 
        and pandas

    Parameters
    ----------
    images : list
        List of images to analize.
    hdf_storename : str
        Hdf5 file to store surface data, with keys of format 'wedgetop_*****.
    plate_y_pos : float or int, optional
        Postion in image coordinates where the top of the plate is located.
        The default is 700.
    from_larger : bool, optional
        WHether or not the surface should be taken from larger image. 
        The default is False.
    plate : bool, optional
        Presence of plate. The default is False.

    Returns
    -------
    None.

    '''
    st = pd.HDFStore(hdf_storename)
    for i,im in enumerate(images):
        # greyscale of original image
        if type(images) == pims.image_sequence.ImageSequence:
            if im.shape[-1] == 4: #RGBA
                basic_grey = color.rgb2gray(np.asarray(im)[:,:,:-1])
            else: #RGB
                basic_grey = color.rgb2gray(np.asarray(im))
        else:
            im = io.imread(im)
            if im.shape[-1] == 4: #RGBA
                basic_grey = color.rgb2gray(im[:,:,:-1])
            else: #RGB
                basic_grey = color.rgb2gray(im)
        # crop from larger image (most are)
        if from_larger:
            orig_h = basic_grey.shape[0]
            basic_grey = basic_grey[ymin:ymax,xmin:xmax]  
        
        # Otsu (1979) thresholding to get histogram threshold value
        thresh = filters.threshold_otsu(basic_grey)
        threshed = basic_grey<thresh
        
        # fill in any holes, no matter how small, via labeling (Jain, 1989)
        labels = measure.label(threshed)
        labelCount = np.bincount(labels.ravel())
        background = np.argmax(labelCount)
        threshed[labels != background] = 1
        sure_bg_filled = binary_fill_holes(threshed).astype(np.uint8)
        
        # fill in wedge foreground so that reflective plate does not
        #   interfere with analysis
        if plate: 
            sure_bg_filled[plate_y_pos:,:] = 1
            
        # sobel edge detection 
        edge = filters.sobel(sure_bg_filled)
        
        # extract surface from detected edge
        surf = []
        for j in range(edge.shape[1]):
            if len(edge[:,j][edge[:,j] > 0]):
                bottom = np.nonzero(edge[:,j])[0][-1]
                surf.append([j,edge.shape[0]-bottom])
        surf = pd.DataFrame(surf,columns=('x','y'))
        # delete points that are within wedge
        surf.y[surf.y.diff().abs() > surf_cutoff] = np.nan
        # adjustable median filter window to remove errant points in surface
        surf.y = medfilt(surf.y,medwindow)
        # correct coordinates to cropped version
        if from_larger:
            surf.x = surf.x + xmin
            surf.y = surf.y + (orig_h-ymax)
        # save surf, as int16 as we're dealing with pixel edges (save space)
        surf_xy = np.vstack((surf.x,surf.y)).astype(np.int16)
        st.put('wedgetop_%05.0f'%i,pd.DataFrame(surf_xy.T,columns=('x','y')))
        print(r'Calculated surface of image # %05.0f'%(i))
    st.close()