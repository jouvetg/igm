#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file 

"""
This IGM module loads input spatial fields from tiff file. You may select
available fields in variables you need at least topg or usurf, and thk,
filed e.g. topg.tif, thk.tif must be present in the working forlder.

==============================================================================

Input: tiff files
Output: variables contained inside as tensorflow objects
"""


import numpy as np
import os
import tensorflow as tf

from igm.modules.utils import complete_data

def params_load_tif_data(parser):
    pass

def init_load_tif_data(params, self):

    import rasterio

    files = glob.glob(os.path.join(self.config.working_dir, "*.tif"))

    for file in files:
        var = os.path.split(file)[-1].split(".")[0]
        if os.path.exists(file):
            self.profile_tif_file = rasterio.open(file, "r").profile
            with rasterio.open(file) as src:
                vars()[var] = np.flipud(src.read(1))
                height = vars()[var].shape[0]
                width = vars()[var].shape[1]
                cols, rows = np.meshgrid(np.arange(width), np.arange(height))
                x, y = rasterio.transform.xy(src.transform, rows, cols)
                x = np.array(x)[0, :]
                y = np.flip(np.array(y)[:, 0])
                vars(self)[var] = tf.Variable(vars()[var].astype("float32"))
            del src

    self.x = tf.constant(x.astype("float32"))
    self.y = tf.constant(y.astype("float32"))

    complete_data(self)

def update_load_tif_data(params, self):
    pass
    
    
def final_load_tif_data(params, self):
    pass
    


