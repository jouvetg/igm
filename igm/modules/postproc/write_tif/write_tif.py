#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os
import tensorflow as tf


def params(parser):
    parser.add_argument(
        "--wtif_vars_to_save",
        type=list,
        default=[
            "usurf",
            "thk",
        ],
        help="List of variables to be recorded in the NetCDF file",
    )


def initialize(params, state):
    pass


def update(params, state):
    import rasterio

    if state.saveresult:
        for var in params.wtif_vars_to_save:
            file = var + "-" + str(int(state.t)).zfill(6) + ".tif"

            os.system("echo rm " + file + " >> clean.sh")

            if hasattr(state, "profile_tif_file"):
                with rasterio.open(file, mode="w", **state.profile_tif_file) as src:
                    src.write(np.flipud(vars(state)[var]), 1)
            else:
                xres = (state.x[-1] - state.x[0]) / len(state.x)
                yres = (state.y[-1] - state.y[0]) / len(state.y)
                transform = rasterio.Affine.translation(
                    state.x[0] - xres / 2, state.y[0] - yres / 2
                ) * rasterio.Affine.scale(xres, yres)

                with rasterio.open(
                    file,
                    mode="w",
                    driver="GTiff",
                    height=vars(state)[var].shape[0],
                    width=vars(state)[var].shape[1],
                    count=1,
                    dtype=np.float32,
                    transform=transform,
                ) as src:
                    src.write(np.flipud(vars(state)[var]), 1)

            del src


def finalize(params, state):
    pass
