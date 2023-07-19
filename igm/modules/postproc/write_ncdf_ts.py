#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file 

import numpy as np
import os, sys, shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from netCDF4 import Dataset

def params_write_ncdf_ts(state):
    pass


def init_write_ncdf_ts(params, state):
    os.system("echo rm " + os.path.join(params.working_dir, "ts.nc") + " >> clean.sh")

    state.var_info_ncdf_ts = {}
    state.var_info_ncdf_ts["vol"] = ["Ice volume", "km^3"]
    state.var_info_ncdf_ts["area"] = ["Glaciated area", "km^2"]


def update_write_ncdf_ts(params, state):

    if state.saveresult:
        vol = np.sum(state.thk) * (state.dx**2) / 10**9
        area = np.sum(state.thk > 1) * (state.dx**2) / 10**6

        if not hasattr(state, "already_called_update_write_ncdf_ts"):
            state.already_called_update_write_ncdf_ts = True

            state.logger.info("Initialize NCDF ts output Files")

            nc = Dataset(
                os.path.join(params.working_dir, "ts.nc"),
                "w",
                format="NETCDF4",
            )

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

            # os.system('echo rm '+ os.path.join(params.working_dir, "ts.nc") + ' >> clean.sh')

        else:
            state.logger.info("Write NCDF ts file at time : " + str(state.t.numpy()))

            nc = Dataset(
                os.path.join(params.working_dir, "ts.nc"),
                "a",
                format="NETCDF4",
            )
            d = nc.variables["time"][:].shape[0]

            nc.variables["time"][d] = state.t.numpy()
            for var in ["vol", "area"]:
                nc.variables[var][d] = vars()[var].numpy()
            nc.close()

    
def final_write_ncdf_ts(params, state):
    pass
