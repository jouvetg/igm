#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os 
import tensorflow as tf
from netCDF4 import Dataset
 
from igm.modules.utils import *
 
def params(parser):
    parser.add_argument(
        "--rncd_input_file",
        type=str,
        default="output.nc",
        help="NetCDF input data file",
    )
    parser.add_argument(
        "--rncd_crop",
        type=str2bool,
        default="False",
        help="Crop the data from NetCDF file with given top/down/left/right bounds",
    )
    parser.add_argument(
        "--rncd_xmin",
        type=float,
        help="X left coordinate for cropping the NetCDF data",
        default=-(10**20),
    )
    parser.add_argument(
        "--rncd_xmax",
        type=float,
        help="X right coordinate for cropping the NetCDF data",
        default=10**20,
    )
    parser.add_argument(
        "--rncd_ymin",
        type=float,
        help="Y bottom coordinate fro cropping the NetCDF data",
        default=-(10**20),
    )
    parser.add_argument(
        "--rncd_ymax",
        type=float,
        help="Y top coordinate for cropping the NetCDF data",
        default=10**20,
    )
 
def initialize(params, state):
 
    nc = Dataset(params.rncd_input_file, "r")

    state.x    = np.squeeze(nc.variables["x"]).astype("float32")
    state.y    = np.squeeze(nc.variables["y"]).astype("float32")
    state.time = np.squeeze(nc.variables["time"]).astype("float32")

    state.list_var = [l for l in nc.variables if not l in ["x", "y", "z", "time"]]

    for var in state.list_var:
        vars(state)[var+'_'+'t'] = np.squeeze(nc.variables[var]).astype("float32")
 
    # crop if requested
    if params.rncd_crop:
        i0 = max(0, int((params.rncd_xmin - x[0]) / (x[1] - x[0])))
        i1 = min(int((params.rncd_xmax - x[0]) / (x[1] - x[0])), x.shape[0] - 1)
        i1 = max(i0 + 1, i1)
        j0 = max(0, int((params.rncd_ymin - y[0]) / (y[1] - y[0])))
        j1 = min(int((params.rncd_ymax - y[0]) / (y[1] - y[0])), y.shape[0] - 1)
        j1 = max(j0 + 1, j1) 
        for var in state.list_var:
            vars(state)[var+'_'+'t'] = vars(state)[var+'_'+'t'][j0:j1, i0:i1]
        y = y[j0:j1]
        x = x[i0:i1]
        
    for var in ["x", "y"]:
        vars(state)[var] = tf.constant(vars(state)[var].astype("float32"))

    nc.close()
    
    params.time_start = state.time[0]
    params.time_end   = state.time[-1]
     
    state.t = tf.Variable(float(params.time_start))
    state.it = 0
    state.saveresult = True

def update(params, state):
    
    state.t.assign(state.time[state.it])
     
    for var in state.list_var:
        vars(state)[var] = tf.Variable(vars(state)[var+'_'+'t'][state.it]) 

    state.it += 1

def finalize(params, state):
    pass
