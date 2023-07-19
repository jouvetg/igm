#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file 

import numpy as np
import os
import tensorflow as tf

def params_write_tif_ex(parser):
    parser.add_argument(
        "--vars_to_save",
        type=list,
        default=[
            "usurf",
            "thk",
        ],
        help="List of variables to be recorded in the ncdf file",
    )

def init_write_tif_ex(params, state):
    pass

def update_write_tif_ex(params, state):

    import rasterio

    if state.saveresult:
        for var in params.vars_to_save:
            file = os.path.join(
                params.working_dir, var + "-" + str(int(state.t)).zfill(6) + ".tif"
            )

            with rasterio.open(file, mode="w", **state.profile_tif_file) as src:
                src.write(np.flipud(vars(state)[var]), 1)

            del src
            
            
def final_write_tif_ex(params, state):
    pass

