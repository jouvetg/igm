import numpy as np
import os
import datetime, time
import tensorflow as tf
from netCDF4 import Dataset
from scipy.interpolate import RectBivariateSpline

from igm.modules.utils import *

from .include_icemask import include_icemask

def run(cfg, state):
    if hasattr(state, "logger"):
        state.logger.info("LOAD NCDF file")

    nc = Dataset(cfg.input.load_ncdf.lncd_input_file, "r")

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
    if cfg.input.load_ncdf.lncd_coarsen > 1:
        xx = x[:: cfg.input.load_ncdf.lncd_coarsen]
        yy = y[:: cfg.input.load_ncdf.lncd_coarsen]
         
        if cfg.input.load_ncdf.lncd_method_coarsen == "skipping":            
            for var in nc.variables:
                if (not var in ["x", "y"]) & (vars()[var].ndim == 2):
                    vars()[var] = vars()[var][
                        :: cfg.input.load_ncdf.lncd_coarsen, :: cfg.input.load_ncdf.lncd_coarsen
                    ]
        elif cfg.input.load_ncdf.lncd_method_coarsen == "cubic_spline":
            for var in nc.variables:
                if (not var in ["x", "y"]) & (vars()[var].ndim == 2):
                                    
                    interp_spline = RectBivariateSpline(y, x, vars()[var])
                    vars()[var] = interp_spline(yy, xx)
        x = xx
        y = yy



    # crop if requested
    if cfg.input.load_ncdf.lncd_crop:
        i0 = max(0, int((cfg.input.load_ncdf.lncd_xmin - x[0]) / (x[1] - x[0])))
        i1 = min(int((cfg.input.load_ncdf.lncd_xmax - x[0]) / (x[1] - x[0])), x.shape[0] - 1)
        i1 = max(i0 + 1, i1)
        j0 = max(0, int((cfg.input.load_ncdf.lncd_ymin - y[0]) / (y[1] - y[0])))
        j1 = min(int((cfg.input.load_ncdf.lncd_ymax - y[0]) / (y[1] - y[0])), y.shape[0] - 1)
        j1 = max(j0 + 1, j1)
        #        i0,i1 = int((cfg.input.load_ncdf.lncd_xmin-x[0])/(x[1]-x[0])),int((cfg.input.load_ncdf.lncd_xmax-x[0])/(x[1]-x[0]))
        #        j0,j1 = int((cfg.input.load_ncdf.lncd_ymin-y[0])/(y[1]-y[0])),int((cfg.input.load_ncdf.lncd_ymax-y[0])/(y[1]-y[0]))
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

    if cfg.input.load_ncdf.icemask_include:
        include_icemask(state, mask_shapefile=cfg.input.load_ncdf.icemask_shapefile, mask_invert=cfg.input.load_ncdf.icemask_invert)