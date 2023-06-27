#!/usr/bin/env python3

"""
Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
Published under the GNU GPL (Version 3), check at the LICENSE file
"""

import numpy as np
import os
import tensorflow as tf

def load_tif_data(self,variables):
    """
    Load the input files from tiff file (alternative to load_ncdf_data)
    Select available fields in variables
    you need at least topg or usurf, and thk,
    filed e.g. topg.tif, thk.tif must be present in the working forlder
    you need rasterio to play this function
    """
    
    import rasterio
    
    for var in variables:
        file = os.path.join(self.config.working_dir,var+'.tif')
        if os.path.exists(file):
            self.profile_tif_file = rasterio.open(file, 'r').profile
            with rasterio.open(file) as src:    
                vars()[var] = np.flipud(src.read(1))
                height      = vars()[var].shape[0]
                width       = vars()[var].shape[1]
                cols, rows  = np.meshgrid(np.arange(width), np.arange(height))
                x, y = rasterio.transform.xy(src.transform, rows, cols)
                x = np.array(x)[0,:]
                y = np.flip(np.array(y)[:,0])                  
                vars(self)[var] = tf.Variable(vars()[var].astype("float32"))
            del src
 
    self.x = tf.constant(x.astype("float32"))
    self.y = tf.constant(y.astype("float32"))