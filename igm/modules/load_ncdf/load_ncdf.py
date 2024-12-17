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


def params(parser):
    parser.add_argument(
        "--lncd_input_file",
        type=str,
        default="input.nc",
        help="NetCDF input data file",
    )
    parser.add_argument(
        "--lncd_method_coarsen",
        type=str,
        default="skipping",
        help="Method for coarsening the data from NetCDF file: skipping or cubic_spline",
    )
    parser.add_argument(
        "--lncd_coarsen",
        type=int,
        default=1,
        help="Coarsen the data from NetCDF file by a certain (integer) number: 2 would be twice coarser ignore data each 2 grid points",
    )
    parser.add_argument(
        "--lncd_crop",
        type=str2bool,
        default="False",
        help="Crop the data from NetCDF file with given top/down/left/right bounds",
    )
    parser.add_argument(
        "--lncd_xmin",
        type=float,
        help="X left coordinate for cropping the NetCDF data",
        default=-(10**20),
    )
    parser.add_argument(
        "--lncd_xmax",
        type=float,
        help="X right coordinate for cropping the NetCDF data",
        default=10**20,
    )
    parser.add_argument(
        "--lncd_ymin",
        type=float,
        help="Y bottom coordinate fro cropping the NetCDF data",
        default=-(10**20),
    )
    parser.add_argument(
        "--lncd_ymax",
        type=float,
        help="Y top coordinate for cropping the NetCDF data",
        default=10**20,
    )


def initialize(cfg, state):
    if hasattr(state, "logger"):
        state.logger.info("LOAD NCDF file")

    nc = Dataset(cfg.modules.load_ncdf.lncd_input_file, "r")

    x = np.squeeze(nc.variables["x"]).astype("float32")
    y = np.squeeze(nc.variables["y"]).astype("float32")

    # make sure the grid has same cell spacing in x and y
    # assert abs(x[1] - x[0]) == abs(y[1] - y[0])

    # load any field contained in the ncdf file, replace missing entries by nan

    if "time" in nc.variables:
        TIME = np.squeeze(nc.variables["time"]).astype("float32")
        I = np.where(TIME == cfg.modules.time_igm.time_start)[0][0]
        istheretime = True
    else:
        istheretime = False

    for var in nc.variables:
        if not var in ["x", "y", "z", "time"]:
            if istheretime:
                vars()[var] = np.squeeze(nc.variables[var][I]).astype("float32")
            else:
                vars()[var] = np.squeeze(nc.variables[var]).astype("float32")
            vars()[var] = np.where(vars()[var] > 10**35, np.nan, vars()[var])



    # coarsen if requested
    if cfg.modules.load_ncdf.lncd_coarsen > 1:
        xx = x[:: cfg.modules.load_ncdf.lncd_coarsen]
        yy = y[:: cfg.modules.load_ncdf.lncd_coarsen]
         
        if cfg.modules.load_ncdf.lncd_method_coarsen == "skipping":            
            for var in nc.variables:
                if (not var in ["x", "y"]) & (vars()[var].ndim == 2):
                    vars()[var] = vars()[var][
                        :: cfg.modules.load_ncdf.lncd_coarsen, :: cfg.modules.load_ncdf.lncd_coarsen
                    ]
        elif cfg.modules.load_ncdf.lncd_method_coarsen == "cubic_spline":
            for var in nc.variables:
                if (not var in ["x", "y"]) & (vars()[var].ndim == 2):
                                    
                    interp_spline = RectBivariateSpline(y, x, vars()[var])
                    vars()[var] = interp_spline(yy, xx)
        x = xx
        y = yy



    # crop if requested
    if cfg.modules.load_ncdf.lncd_crop:
        i0 = max(0, int((cfg.modules.load_ncdf.lncd_xmin - x[0]) / (x[1] - x[0])))
        i1 = min(int((cfg.modules.load_ncdf.lncd_xmax - x[0]) / (x[1] - x[0])), x.shape[0] - 1)
        i1 = max(i0 + 1, i1)
        j0 = max(0, int((cfg.modules.load_ncdf.lncd_ymin - y[0]) / (y[1] - y[0])))
        j1 = min(int((cfg.modules.load_ncdf.lncd_ymax - y[0]) / (y[1] - y[0])), y.shape[0] - 1)
        j1 = max(j0 + 1, j1)
        #        i0,i1 = int((cfg.modules.load_ncdf.lncd_xmin-x[0])/(x[1]-x[0])),int((cfg.modules.load_ncdf.lncd_xmax-x[0])/(x[1]-x[0]))
        #        j0,j1 = int((cfg.modules.load_ncdf.lncd_ymin-y[0])/(y[1]-y[0])),int((cfg.modules.load_ncdf.lncd_ymax-y[0])/(y[1]-y[0]))
        for var in nc.variables:
            if (not var in ["x", "y"]) & (vars()[var].ndim == 2):
                vars()[var] = vars()[var][j0:j1, i0:i1]
        y = y[j0:j1]
        x = x[i0:i1]

    # transform from numpy to tensorflow
    for var in nc.variables:
        if not var in ["z", "time"]:
            if var in ["x", "y"]:
                vars(state)[var] = tf.constant(vars()[var].astype("float32"))
            else:
                vars(state)[var] = tf.Variable(vars()[var].astype("float32"), trainable=False)

    nc.close()

    complete_data(state)


def update(cfg, state):
    pass


def finalize(cfg, state):
    pass
