#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file 

import numpy as np
import os
import datetime, time
import tensorflow as tf
from netCDF4 import Dataset
from scipy.interpolate import RectBivariateSpline

from igm.modules.utils import *


def params_load_ncdf_data(parser):
    parser.add_argument(
        "--geology_file",
        type=str,
        default="geology.nc",
        help="Input data file (default: geology.nc)",
    )
    parser.add_argument(
        "--resample",
        type=int,
        default=1,
        help="Resample the data to a coarser resolution (default: 1), e.g. 2 would be twice coarser ignore data each 2 grid points",
    )
    parser.add_argument(
        "--crop_data",
        type=str2bool,
        default="False",
        help="Crop the data with xmin, xmax, ymin, ymax (default: False)",
    )
    parser.add_argument(
        "--crop_xmin",
        type=float, 
        help="X left coordinate for cropping",
    )
    parser.add_argument(
        "--crop_xmax",
        type=float, 
        help="X right coordinate for cropping",
    )
    parser.add_argument(
        "--crop_ymin",
        type=float, 
        help="Y bottom coordinate fro cropping",
    )
    parser.add_argument(
        "--crop_ymax",
        type=float, 
        help="Y top coordinate for cropping"
    )




def init_load_ncdf_data(params, state):

    if hasattr(state,'logger'):
        state.logger.info("LOAD NCDF file")

    nc = Dataset(os.path.join(params.working_dir, params.geology_file), "r")

    x = np.squeeze(nc.variables["x"]).astype("float32")
    y = np.squeeze(nc.variables["y"]).astype("float32")

    # make sure the grid has same cell spacing in x and y
    assert x[1] - x[0] == y[1] - y[0]

    # load any field contained in the ncdf file, replace missing entries by nan
    for var in nc.variables:
        if not var in ["x", "y"]:
            vars()[var] = np.squeeze(nc.variables[var]).astype("float32")
            vars()[var] = np.where(vars()[var] > 10**35, np.nan, vars()[var])

    # resample if requested
    if params.resample > 1:
        xx = x[:: params.resample]
        yy = y[:: params.resample]
        for var in nc.variables:
            if not var in ["x", "y"]:
                vars()[var] = RectBivariateSpline(y, x, vars()[var])(yy, xx)
        x = xx
        y = yy

    # crop if requested
    if params.crop_data:
        i0,i1,j0,j1 = crop_field(params, state)
        for var in nc.variables:
            if not var in ["x", "y"]:
                vars()[var] = vars()[var][j0:j1,i0:i1]
        y = y[j0:j1]
        x = x[i0:i1]

    # transform from numpy to tensorflow
    for var in nc.variables:
        if var in ["x", "y"]:
            vars(state)[var] = tf.constant(vars()[var].astype("float32"))
        else:
            vars(state)[var] = tf.Variable(vars()[var].astype("float32"))
 
    nc.close()

    complete_data(state)


def update_load_ncdf_data(params, state):
    pass


def final_load_ncdf_data(params, state):
    pass 
