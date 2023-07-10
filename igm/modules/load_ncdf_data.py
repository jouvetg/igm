#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file 

"""
This IGM module loads spatial raster data from a netcdf file (geology.nc) and
transform the fields into tensorflow variables. It also complete the data,
e.g. ge the basal topography from ice thickness and surface topography.
(there is no update function defined). It contains the two functions for
resampling and cropping the data.

==============================================================================

Input: geology.nc
Output: variables contained inside as tensorflow objects
"""

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
        help="Geology input file (default: geology.nc)",
    )
    parser.add_argument(
        "--resample",
        type=int,
        default=1,
        help="Resample the data of ncdf data file to a coarser resolution (default: 1)",
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




def init_load_ncdf_data(params, self):

    self.logger.info("LOAD NCDF file")

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
        i0,i1,j0,j1 = crop_field(params, self)
        for var in nc.variables:
            if not var in ["x", "y"]:
                vars(self)[var] = vars(self)[var][j0:j1,i0:i1]
        self.y = self.y[j0:j1]
        self.x = self.x[i0:i1]

    # transform from numpy to tensorflow
    for var in nc.variables:
        if var in ["x", "y"]:
            vars(self)[var] = tf.constant(vars()[var].astype("float32"))
        else:
            vars(self)[var] = tf.Variable(vars()[var].astype("float32"))
 
    nc.close()

    complete_data(self)


def update_load_ncdf_data(params, self):
    pass


def final_load_ncdf_data(params, self):
    pass 
