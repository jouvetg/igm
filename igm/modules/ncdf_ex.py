#!/usr/bin/env python3

"""
Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
Published under the GNU GPL (Version 3), check at the LICENSE file
"""

import numpy as np
import os, sys, shutil
import matplotlib.pyplot as plt
import datetime, time
import tensorflow as tf
import argparse
from netCDF4 import Dataset

from igm.modules.utils import getmag


def params_ncdf_ex(parser):
    parser.add_argument(
        "--vars_to_save",
        type=list,
        default=[
            "topg",
            "usurf",
            "thk",
            "smb",
            "velbar_mag",
            "velsurf_mag",
        ],
        help="List of variables to be recorded in the ncdf file",
    )

    # params = self.parser.parse_args()


def init_ncdf_ex(params, self):
    self.tcomp["ncdf_ex"] = []

    os.system("echo rm " + os.path.join(params.working_dir, "ex.nc") + " >> clean.sh")

    _def_var_info(self)


def update_ncdf_ex(params, self):
    """
    This function write 2D field variables defined in the list config.vars_to_save
    into the ncdf output file ex.nc
    """

    if self.saveresult:
        self.tcomp["ncdf_ex"].append(time.time())

        if "velbar_mag" in params.vars_to_save:
            self.velbar_mag = getmag(self.ubar, self.vbar)

        if "velsurf_mag" in params.vars_to_save:
            self.velsurf_mag = getmag(self.uvelsurf, self.vvelsurf)

        if "velbase_mag" in params.vars_to_save:
            self.velbase_mag = getmag(self.uvelbase, self.vvelbase)

        if "meanprec" in params.vars_to_save:
            self.meanprec = tf.math.reduce_mean(self.precipitation, axis=0)

        if "meantemp" in params.vars_to_save:
            self.meantemp = tf.math.reduce_mean(self.air_temp, axis=0)

        if not hasattr(self, "already_called_update_ncdf_ex"):
            self.already_called_update_ncdf_ex = True

            self.logger.info("Initialize NCDF ex output Files")

            nc = Dataset(
                os.path.join(params.working_dir, "ex.nc"),
                "w",
                format="NETCDF4",
            )

            nc.createDimension("time", None)
            E = nc.createVariable("time", np.dtype("float32").char, ("time",))
            E.units = "yr"
            E.long_name = "time"
            E.axis = "T"
            E[0] = self.t.numpy()

            nc.createDimension("y", len(self.y))
            E = nc.createVariable("y", np.dtype("float32").char, ("y",))
            E.units = "m"
            E.long_name = "y"
            E.axis = "Y"
            E[:] = self.y.numpy()

            nc.createDimension("x", len(self.x))
            E = nc.createVariable("x", np.dtype("float32").char, ("x",))
            E.units = "m"
            E.long_name = "x"
            E.axis = "X"
            E[:] = self.x.numpy()

            for var in params.vars_to_save:
                E = nc.createVariable(var, np.dtype("float32").char, ("time", "y", "x"))
                E.long_name = self.var_info[var][0]
                E.units = self.var_info[var][1]
                E[0, :, :] = vars(self)[var].numpy()

            nc.close()

        else:
            self.logger.info("Write NCDF ex file at time : " + str(self.t.numpy()))

            nc = Dataset(
                os.path.join(params.working_dir, "ex.nc"),
                "a",
                format="NETCDF4",
            )

            d = nc.variables["time"][:].shape[0]
            nc.variables["time"][d] = self.t.numpy()

            for var in params.vars_to_save:
                nc.variables[var][d, :, :] = vars(self)[var].numpy()

            nc.close()

        self.tcomp["ncdf_ex"][-1] -= time.time()
        self.tcomp["ncdf_ex"][-1] *= -1


def _def_var_info(self):
    # give information on variables for output ncdf, TODO: IMPROVE
    self.var_info = {}
    self.var_info["topg"] = ["Basal Topography", "m"]
    self.var_info["usurf"] = ["Surface Topography", "m"]
    self.var_info["thk"] = ["Ice Thickness", "m"]
    self.var_info["icemask"] = ["Ice mask", "NO UNIT"]
    self.var_info["smb"] = ["Surface Mass Balance", "m/y"]
    self.var_info["ubar"] = ["x depth-average velocity of ice", "m/y"]
    self.var_info["vbar"] = ["y depth-average velocity of ice", "m/y"]
    self.var_info["velbar_mag"] = ["Depth-average velocity magnitude of ice", "m/y"]
    self.var_info["uvelsurf"] = ["x surface velocity of ice", "m/y"]
    self.var_info["vvelsurf"] = ["y surface velocity of ice", "m/y"]
    self.var_info["wvelsurf"] = ["z surface velocity of ice", "m/y"]
    self.var_info["velsurf_mag"] = ["Surface velocity magnitude of ice", "m/y"]
    self.var_info["uvelbase"] = ["x basal velocity of ice", "m/y"]
    self.var_info["vvelbase"] = ["y basal velocity of ice", "m/y"]
    self.var_info["wvelbase"] = ["z basal velocity of ice", "m/y"]
    self.var_info["velbase_mag"] = ["Basal velocity magnitude of ice", "m/y"]
    self.var_info["divflux"] = ["Divergence of the ice flux", "m/y"]
    self.var_info["strflowctrl"] = ["arrhenius+1.0*slidingco", "MPa$^{-3}$ a$^{-1}$"]
    self.var_info["dtopgdt"] = ["Erosion rate", "m/y"]
    self.var_info["arrhenius"] = ["Arrhenius factor", "MPa$^{-3}$ a$^{-1}$"]
    self.var_info["slidingco"] = ["Sliding Coefficient", "km MPa$^{-3}$ a$^{-1}$"]
    self.var_info["meantemp"] = ["Mean anual surface temperatures", "°C"]
    self.var_info["meanprec"] = ["Mean anual precipitation", "m/y"]
    self.var_info["vol"] = ["Ice volume", "km^3"]
    self.var_info["area"] = ["Glaciated area", "km^2"]
    self.var_info["velsurfobs_mag"] = ["Obs. surf. speed of ice", "m/y"]
    self.var_info["weight_particles"] = ["weight_particles", "no"]
