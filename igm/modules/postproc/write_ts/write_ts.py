#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os
from netCDF4 import Dataset


def params(parser):
    parser.add_argument(
        "--wts_output_file",
        type=str,
        default="output_ts.nc",
        help="Output ncdf data file (time serie)",
    )


def initialize(params, state):
    os.system( "echo rm " + params.wts_output_file + " >> clean.sh" )

    state.var_info_ncdf_ts = {}
    state.var_info_ncdf_ts["vol"] = ["Ice volume", "km^3"]
    state.var_info_ncdf_ts["area"] = ["Glaciated area", "km^2"]


def update(params, state):
    if state.saveresult:
        vol = np.sum(state.thk) * (state.dx**2) / 10**9
        area = np.sum(state.thk > 1) * (state.dx**2) / 10**6

        if not hasattr(state, "already_called_update_write_ts"):
            state.already_called_update_write_ts = True

            if hasattr(state, "logger"):
                state.logger.info("Initialize NCDF ts output Files")

            nc = Dataset( params.wts_output_file,"w", format="NETCDF4" )

            nc.createDimension("time", None)
            E = nc.createVariable("time", np.dtype("float32").char, ("time",))
            E.units = "yr"
            E.long_name = "time"
            E.axis = "T"
            E[0] = state.t.numpy()

            for var in ["vol", "area"]:
                E = nc.createVariable(var, np.dtype("float32").char, ("time"))
                E[0] = vars()[var].numpy()
                E.long_name = state.var_info_ncdf_ts[var][0]
                E.units = state.var_info_ncdf_ts[var][1]
            nc.close()

        else:
            if hasattr(state, "logger"):
                state.logger.info(
                    "Write NCDF ts file at time : " + str(state.t.numpy())
                )

            nc = Dataset( params.wts_output_file, "a", format="NETCDF4" )
            d = nc.variables["time"][:].shape[0]

            nc.variables["time"][d] = state.t.numpy()
            for var in ["vol", "area"]:
                nc.variables[var][d] = vars()[var].numpy()
            nc.close()


def finalize(params, state):
    pass
