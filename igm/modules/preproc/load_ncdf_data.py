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
        "--input_file",
        type=str,
        default="input.nc",
        help="Input data file",
    )
    parser.add_argument(
        "--coarsen_ncdf",
        type=int,
        default=1,
        help="coarsen the data to a coarser resolution (default: 1), e.g. 2 would be twice coarser ignore data each 2 grid points",
    )
    parser.add_argument(
        "--crop_ncdf",
        type=str2bool,
        default="False",
        help="Crop the data with xmin, xmax, ymin, ymax (default: False)",
    )
    parser.add_argument(
        "--crop_ncdf_xmin",
        type=float, 
        help="X left coordinate for cropping",
    )
    parser.add_argument(
        "--crop_ncdf_xmax",
        type=float, 
        help="X right coordinate for cropping",
    )
    parser.add_argument(
        "--crop_ncdf_ymin",
        type=float, 
        help="Y bottom coordinate fro cropping",
    )
    parser.add_argument(
        "--crop_ncdf_ymax",
        type=float, 
        help="Y top coordinate for cropping"
    )




def initialize_load_ncdf_data(params, state):

    if hasattr(state,'logger'):
        state.logger.info("LOAD NCDF file")

    nc = Dataset(os.path.join(params.working_dir, params.input_file), "r")

    x = np.squeeze(nc.variables["x"]).astype("float32")
    y = np.squeeze(nc.variables["y"]).astype("float32")

    # make sure the grid has same cell spacing in x and y
    assert x[1] - x[0] == y[1] - y[0]

    # load any field contained in the ncdf file, replace missing entries by nan
    for var in nc.variables:
        if not var in ["x", "y"]:
            vars()[var] = np.squeeze(nc.variables[var]).astype("float32")
            vars()[var] = np.where(vars()[var] > 10**35, np.nan, vars()[var])

    # coarsen if requested
    if params.coarsen_ncdf> 1:
        xx = x[:: params.coarsen_ncdf]
        yy = y[:: params.coarsen_ncdf]
        for var in nc.variables:
            if (not var in ["x", "y"]) & (vars()[var].ndim==2):
                vars()[var] = vars()[var][:: params.coarsen_ncdf,:: params.coarsen_ncdf]
#                vars()[var] = RectBivariateSpline(y, x, vars()[var])(yy, xx) # does not work
        x = xx
        y = yy

    # crop if requested
    if params.crop_ncdf:
        i0,i1 = np.clip(int((params.crop_ncdf_xmin-x[0])/(x[1]-x[0])),int((params.crop_ncdf_xmax-x[0])/(x[1]-x[0])), 0, x.shape[0]-1)
        j0,j1 = np.clip(int((params.crop_ncdf_ymin-y[0])/(y[1]-y[0])),int((params.crop_ncdf_ymax-y[0])/(y[1]-y[0])), 0, y.shape[0]-1)
        for var in nc.variables:
            if (not var in ["x", "y"]) & (vars()[var].ndim==2):
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


def finalize_load_ncdf_data(params, state):
    pass 
