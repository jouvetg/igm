import xarray as xr
import numpy as np
import tensorflow as tf
import os

from igm.processes.utils import getmag

def initialize(cfg, state):
    state.var_info_ncdf_ex = {
        "topg": ["Basal Topography", "m"],
        "usurf": ["Surface Topography", "m"],
        "thk": ["Ice Thickness", "m"],
        "icemask": ["Ice mask", "NO UNIT"],
        "smb": ["Surface Mass Balance", "m/y ice eq"],
        "ubar": ["x depth-average velocity of ice", "m/y"],
        "vbar": ["y depth-average velocity of ice", "m/y"],
        "velbar_mag": ["Depth-average velocity magnitude of ice", "m/y"],
        "uvelsurf": ["x surface velocity of ice", "m/y"],
        "vvelsurf": ["y surface velocity of ice", "m/y"],
        "wvelsurf": ["z surface velocity of ice", "m/y"],
        "velsurf_mag": ["Surface velocity magnitude of ice", "m/y"],
        "uvelbase": ["x basal velocity of ice", "m/y"],
        "vvelbase": ["y basal velocity of ice", "m/y"],
        "wvelbase": ["z basal velocity of ice", "m/y"],
        "velbase_mag": ["Basal velocity magnitude of ice", "m/y"],
        "divflux": ["Divergence of the ice flux", "m/y"],
        "strflowctrl": ["arrhenius+1.0*slidingco", "MPa$^{-3}$ a$^{-1}$"],
        "dtopgdt": ["Erosion rate", "m/y"],
        "arrhenius": ["Arrhenius factor", "MPa$^{-3}$ a$^{-1}$"],
        "slidingco": ["Sliding Coefficient", "km MPa$^{-3}$ a$^{-1}$"],
        "meantemp": ["Mean annual surface temperatures", "°C"],
        "meanprec": ["Mean annual precipitation", "Kg m^(-2) y^(-1)"],
        "velsurfobs_mag": ["Obs. surf. speed of ice", "m/y"],
        "weight_particles": ["weight_particles", "no"]
    }

def run(cfg, state):
    if not state.saveresult:
        return

    # Prepare any derived quantities
    if "velbar_mag" in cfg.outputs.local.vars_to_save:
        state.velbar_mag = getmag(state.ubar, state.vbar)

    if "velsurf_mag" in cfg.outputs.local.vars_to_save:
        state.velsurf_mag = getmag(state.uvelsurf, state.vvelsurf)

    if "velbase_mag" in cfg.outputs.local.vars_to_save:
        state.velbase_mag = getmag(state.uvelbase, state.vvelbase)

    if "meanprec" in cfg.outputs.local.vars_to_save:
        state.meanprec = tf.reduce_mean(state.precipitation, axis=0)

    if "meantemp" in cfg.outputs.local.vars_to_save:
        state.meantemp = tf.reduce_mean(state.air_temp, axis=0)

    # Build or update Dataset
    file_path = cfg.outputs.local.output_file
    var_list = cfg.outputs.local.vars_to_save

    def create_data_vars(time_idx=0):
        data_vars = {}
        for var in var_list:
            if not hasattr(state, var):
                continue
            arr = vars(state)[var].numpy()
            dims = ("y", "x") if arr.ndim == 2 else ("z", "y", "x")
            data = xr.DataArray(arr, dims=dims)
            data = data.expand_dims(time=[state.t.numpy()])
            attrs = {}
            if var in state.var_info_ncdf_ex:
                attrs["long_name"], attrs["units"] = state.var_info_ncdf_ex[var]
            data.attrs = attrs
            data_vars[var] = data
        return data_vars

    if not hasattr(state, "already_called_update_local"):
        if hasattr(state, "logger"):
            state.logger.info("Creating new NetCDF file with xarray")

        coords = {
            "x": ("x", state.x.numpy()),
            "y": ("y", state.y.numpy()),
            "time": ("time", [state.t.numpy()])
        }

        if "Nz" in cfg.processes.iceflow:
            coords["z"] = ("z", np.arange(cfg.processes.iceflow.numerics.Nz))

        ds = xr.Dataset(
            data_vars=create_data_vars(),
            coords=coords,
            attrs={"pyproj_srs": getattr(state, "pyproj_srs", "")},
        )
 
        ds.to_netcdf(file_path, mode="w")

        state.already_called_update_local = True
    else:
        if hasattr(state, "logger"):
            state.logger.info(f"Appending to NetCDF file at time {state.t.numpy()}")

        ds_existing = xr.open_dataset(file_path)
        new_data = xr.Dataset(
            data_vars=create_data_vars(),
            coords={"time": [state.t.numpy()]},
        )

        # concat and write again
        ds_concat = xr.concat([ds_existing, new_data], dim="time")
        os.remove(file_path)
        ds_concat.to_netcdf(file_path, mode="w")  # overwrite safely
