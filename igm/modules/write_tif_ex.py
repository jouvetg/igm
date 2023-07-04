#!/usr/bin/env python3

"""
Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
Published under the GNU GPL (Version 3), check at the LICENSE file
"""

import numpy as np
import os
import tensorflow as tf


def params_tif_ex(parser):
    parser.add_argument(
        "--vars_to_save",
        type=list,
        default=[
            "usurf",
            "thk",
        ],
        help="List of variables to be recorded in the ncdf file",
    )

def init_tif_ex(params, self):
    pass

def update_tif_ex(params, self):
    """
    Save variables in tiff file (alternative to update_ncdf_ex)
    Select available fields in variables, e.g. topg, usurf, ...
    Files will be created with names like thk-000040.tif in the workin direc.
    you need rasterio to play this function
    """

    import rasterio

    if self.saveresult:
        for var in params.vars_to_save:
            file = os.path.join(
                params.working_dir, var + "-" + str(int(self.t)).zfill(6) + ".tif"
            )

            with rasterio.open(file, mode="w", **self.profile_tif_file) as src:
                src.write(np.flipud(vars(self)[var]), 1)

            del src
