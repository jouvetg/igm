#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file 

import numpy as np
import os, glob
import tensorflow as tf

from igm.modules.utils import *

def params_load_tif_data(parser):
    parser.add_argument(
        "--crop_data",
        type=str2bool,
        default="False",
        help="Crop the data with xmin, xmax, ymin, ymax (default: False)",
    )
    parser.add_argument(
        "--crop_xmin",
        type=float, 
        help="crop_xmin",
    )
    parser.add_argument(
        "--crop_xmax",
        type=float, 
        help="crop_xmax",
    )
    parser.add_argument(
        "--crop_ymin",
        type=float, 
        help="crop_ymin",
    )
    parser.add_argument(
        "--crop_ymax",
        type=float, 
        help="crop_ymax"
    )


def init_load_tif_data(params, state):

    import rasterio

    files = glob.glob(os.path.join(state.config.working_dir, "*.tif"))

    if params.crop_data:
        i0,i1,j0,j1 = crop_field(params, state)

    for file in files:
        var = os.path.split(file)[-1].split(".")[0]
        if os.path.exists(file):
            state.profile_tif_file = rasterio.open(file, "r").profile
            with rasterio.open(file) as src:
                vars()[var] = np.flipud(src.read(1))
                height = vars()[var].shape[0]
                width = vars()[var].shape[1]
                cols, rows = np.meshgrid(np.arange(width), np.arange(height))
                x, y = rasterio.transform.xy(src.transform, rows, cols)
                x = np.array(x)[0, :]
                y = np.flip(np.array(y)[:, 0])
            del src

        if params.crop_data: 
            vars()[var] = vars()[var][j0:j1,i0:i1] 
 
        vars(state)[var] = tf.Variable(vars()[var].astype("float32"))
 
    if params.crop_data: 
        y = y[j0:j1]
        x = x[i0:i1]

    state.x = tf.constant(x.astype("float32"))
    state.y = tf.constant(y.astype("float32"))

    complete_data(state)

def update_load_tif_data(params, state):
    pass
    
    
def final_load_tif_data(params, state):
    pass
    


