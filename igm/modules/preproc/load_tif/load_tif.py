#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os, glob
import tensorflow as tf

from igm.modules.utils import *


def params(parser):
    parser.add_argument(
        "--ltif_coarsen",
        type=int,
        default=1,
        help="coarsen the data to a coarser resolution (default: 1), e.g. 2 would be twice coarser ignore data each 2 grid points",
    )
    parser.add_argument(
        "--ltif_crop",
        type=str2bool,
        default="False",
        help="Crop the data with xmin, xmax, ymin, ymax (default: False)",
    )
    parser.add_argument(
        "--ltif_xmin",
        type=float,
        help="crop_xmin",
        default=-(10**20),
    )
    parser.add_argument(
        "--ltif_xmax",
        type=float,
        help="crop_xmax",
        default=10**20,
    )
    parser.add_argument(
        "--ltif_ymin",
        type=float,
        help="crop_ymin",
        default=-(10**20),
    )
    parser.add_argument(
        "--ltif_ymax",
        type=float,
        help="crop_ymax",
        default=10**20,
    )


def initialize(params, state):
    import rasterio

    files = glob.glob("*.tif")

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

    # coarsen if requested
    if params.ltif_coarsen > 1:
        xx = x[:: params.ltif_coarsen]
        yy = y[:: params.ltif_coarsen]
        for file in files:
            var = os.path.split(file)[-1].split(".")[0]
            if (not var in ["x", "y"]) & (vars()[var].ndim == 2):
                vars()[var] = vars()[var][
                    :: params.ltif_coarsen, :: params.ltif_coarsen
                ]
        #                vars()[var] = RectBivariateSpline(y, x, vars()[var])(yy, xx) # does not work
        x = xx
        y = yy

    # crop if requested
    if params.ltif_crop:
        i0 = max(0, int((params.ltif_xmin - x[0]) / (x[1] - x[0])))
        i1 = min(int((params.ltif_xmax - x[0]) / (x[1] - x[0])), x.shape[0] - 1)
        i1 = max(i0 + 1, i1)
        j0 = max(0, int((params.ltif_ymin - y[0]) / (y[1] - y[0])))
        j1 = min(int((params.ltif_ymax - y[0]) / (y[1] - y[0])), y.shape[0] - 1)
        j1 = max(j0 + 1, j1)
        #        i0,i1 = int((params.ltif_xmin-x[0])/(x[1]-x[0])),int((params.ltif_xmax-x[0])/(x[1]-x[0]))
        #        j0,j1 = int((params.ltif_ymin-y[0])/(y[1]-y[0])),int((params.ltif_ymax-y[0])/(y[1]-y[0]))
        for file in files:
            var = os.path.split(file)[-1].split(".")[0]
            if not var in ["x", "y"]:
                vars()[var] = vars()[var][j0:j1, i0:i1]
        y = y[j0:j1]
        x = x[i0:i1]

    # transform from numpy to tensorflow
    for file in files:
        var = os.path.split(file)[-1].split(".")[0]
        vars(state)[var] = tf.Variable(vars()[var].astype("float32"))

    state.x = tf.constant(x.astype("float32"))
    state.y = tf.constant(y.astype("float32"))

    complete_data(state)


def update(params, state):
    pass


def finalize(params, state):
    pass
