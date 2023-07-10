#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file 

"""
This IGM modules use OGGM utilities and GlaThiDa dataset to prepare data 
for the IGM model for a specific glacier given the RGI ID. One must provide
an RGI ID (check GLIMS VIeWER : https://www.glims.org/maps/glims) 
By default, data are already posprocessed, with spatial resolutio of 100 and 
border of 30. For custom spatial resolution, and the size of 'border' 
to keep a safe distance to the glacier margin, one need
to set preprocess option to False 
The script returns the geology.nc file as necessary for run 
IGM for a forward glacier evolution run, and optionaly 
observation.nc that permit to do a first step of data assimilation & inversion. 
Data are currently based on COPERNIUS DEM 90 
the RGI, and the ice thckness and velocity from (MIllan, 2022) 
For the ice thickness in geology.nc, the use can choose 
between consensus_ice_thickness (farinotti2019) or
millan_ice_thickness (millan2022) dataset 
When activating observation==True, ice thickness profiles are 
downloaded from the GlaThiDa depo (https://gitlab.com/wgms/glathida) 
and are rasterized on working grids 
Script written by G. Jouvet & F. Maussion & E. Welty

==============================================================================

Output: all input variable fields neede to run IGM inverse and/or forward
"""

import numpy as np
import matplotlib.pyplot as plt
import os, glob
from netCDF4 import Dataset
import tensorflow as tf
from igm.modules.utils import str2bool

from igm.modules.utils import complete_data

def params_prepare_data(parser):
    # aletsch  RGI60-11.01450
    # malspina RGI60-01.13696
    # brady    RGI60-01.20796
    # ethan    RGI60-01.00709

    parser.add_argument("--RGI", type=str, default="RGI60-11.01450", help="RGI ID")
    parser.add_argument(
        "--preprocess", type=str2bool, default=True, help="Use preprocessing"
    )
    parser.add_argument(
        "--dx",
        type=int,
        default=100,
        help="Spatial resolution (need preprocess false to change it)",
    )
    parser.add_argument(
        "--border",
        type=int,
        default=30,
        help="Safe border margin  (need preprocess false to change it)",
    )
    parser.add_argument(
        "--thk_source",
        type=str,
        default="consensus_ice_thickness",
        help="millan_ice_thickness or consensus_ice_thickness in geology.nc",
    )
    parser.add_argument(
        "--observation",
        type=str2bool,
        default=False,
        help="Make observation file (for IGM inverse)",
    )
    parser.add_argument(
        "--path_glathida",
        type=str,
        default="/home/gjouvet/",
        help="Path where the Glathida Folder is store, so that you don't need \
              to redownload it at any use of the script",
    )
    parser.add_argument(
        "--output_geology",
        type=str2bool,
        default=True,
        help="Write prepared data into a geology file",
    )
    # parser.add_argument(
    #     "--geology_file",
    #     type=str,
    #     default="geology.nc",
    #     help="Name of the geology file"
    # )


def init_prepare_data(params, state):

    import json

    gdirs, paths_ncdf = _oggm_util([params.RGI], params)

    state.logger.info("Prepare data using oggm and glathida")

    nc = Dataset(paths_ncdf[0], "r+")

    x = np.squeeze(nc.variables["x"]).astype("float32")
    y = np.flip(np.squeeze(nc.variables["y"]).astype("float32"))

    thk = np.flipud(np.squeeze(nc.variables[params.thk_source]).astype("float32"))
    thk = np.where(np.isnan(thk), 0, thk)

    usurf = np.flipud(np.squeeze(nc.variables["topo"]).astype("float32"))

    icemask = np.flipud(np.squeeze(nc.variables["glacier_mask"]).astype("float32"))

    if params.observation:
        usurfobs = np.flipud(np.squeeze(nc.variables["topo"]).astype("float32"))
        icemaskobs = np.flipud(
            np.squeeze(nc.variables["glacier_mask"]).astype("float32")
        )
        uvelsurfobs = np.flipud(np.squeeze(nc.variables["millan_vx"]).astype("float32"))
        vvelsurfobs = np.flipud(np.squeeze(nc.variables["millan_vy"]).astype("float32"))
        thkinit = np.flipud(
            np.squeeze(nc.variables["millan_ice_thickness"]).astype("float32")
        )

        uvelsurfobs = np.where(np.isnan(uvelsurfobs), 0, uvelsurfobs)
        uvelsurfobs = np.where(icemaskobs, uvelsurfobs, 0)

        vvelsurfobs = np.where(np.isnan(vvelsurfobs), 0, vvelsurfobs)
        vvelsurfobs = np.where(icemaskobs, vvelsurfobs, 0)

        thkinit = np.where(np.isnan(thkinit), 0, thkinit)
        thkinit = np.where(icemaskobs, thkinit, 0)

        fff = paths_ncdf[0].split("gridded_data.nc")[0] + "glacier_grid.json"
        with open(fff, "r") as f:
            data = json.load(f)
        proj = data["proj"]

        thkobs = _read_glathida(x, y, usurfobs, proj, params.path_glathida)
        thkobs = np.where(icemaskobs, thkobs, np.nan)

    nc.close()

    #########################################################

    vars_to_save = ["usurf", "thk", "icemask"]

    if params.observation:
        vars_to_save += [
            "usurfobs",
            "thkobs",
            "icemaskobs",
            "uvelsurfobs",
            "vvelsurfobs",
            "thkinit",
        ]

    ########################################################

    # transform from numpy to tensorflow
  
    for var in ['x','y']:
        vars(state)[var] = tf.constant(vars()[var].astype("float32"))

    for var in vars_to_save:
        vars(state)[var] = tf.Variable(vars()[var].astype("float32"))

    complete_data(state)

    ########################################################

    if params.output_geology:

        var_info = {}
        var_info["thk"] = ["Ice Thickness", "m"]
        var_info["usurf"] = ["Surface Topography", "m"]
        var_info["icemaskobs"] = ["Accumulation Mask", "bool"]
        var_info["usurfobs"] = ["Surface Topography", "m"]
        var_info["thkobs"] = ["Ice Thickness", "m"]
        var_info["thkinit"] = ["Ice Thickness", "m"]
        var_info["uvelsurfobs"] = ["x surface velocity of ice", "m/y"]
        var_info["vvelsurfobs"] = ["y surface velocity of ice", "m/y"]
        var_info["icemask"] = ["Ice mask", "no unit"]

        nc = Dataset(os.path.join(params.working_dir, "geology.nc"), "w", format="NETCDF4")

        nc.createDimension("y", len(y))
        yn = nc.createVariable("y", np.dtype("float32").char, ("y",))
        yn.units = "m"
        yn.long_name = "y"
        yn.standard_name = "y"
        yn.axis = "Y"
        yn[:] = y

        nc.createDimension("x", len(x))
        xn = nc.createVariable("x", np.dtype("float32").char, ("x",))
        xn.units = "m"
        xn.long_name = "x"
        xn.standard_name = "x"
        xn.axis = "X"
        xn[:] = x

        for v in vars_to_save:
            E = nc.createVariable(v, np.dtype("float32").char, ("y", "x"))
            E.long_name = var_info[v][0]
            E.units = var_info[v][1]
            E.standard_name = v
            E[:] = vars()[v]

        nc.close()

        state.logger.setLevel(params.logging_level)


def update_prepare_data(params, state):
    pass


def final_prepare_data(params, state):
    pass


#########################################################################


def _oggm_util(RGIs, params):
    """
    Function written by Fabien Maussion
    """

    import oggm.cfg as cfg
    from oggm import utils, workflow, tasks, graphics

    # Initialize OGGM and set up the default run parameters
    cfg.initialize()

    cfg.PARAMS["continue_on_error"] = False
    cfg.PARAMS["use_multiprocessing"] = False

    if params.preprocess:
        WD = "OGGM-prepro"

        # Where to store the data for the run - should be somewhere you have access to
        cfg.PATHS["working_dir"] = utils.gettempdir(dirname=WD, reset=True)

        # We need the outlines here
        rgi_ids = utils.get_rgi_glacier_entities(RGIs)

        base_url = (
            "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/exps/igm_v1"
        )
        gdirs = workflow.init_glacier_directories(
            rgi_ids, prepro_border=30, from_prepro_level=2, prepro_base_url=base_url
        )

    else:
        WD = "OGGM-dir"

        # Map resolution parameters
        cfg.PARAMS["grid_dx_method"] = "fixed"
        cfg.PARAMS["fixed_dx"] = params.dx  # m spacing
        cfg.PARAMS[
            "border"
        ] = params.border  # can now be set to any value since we start from scratch
        cfg.PARAMS["map_proj"] = "utm"

        # Where to store the data for the run - should be somewhere you have access to
        cfg.PATHS["working_dir"] = utils.gettempdir(dirname=WD, reset=True)

        # We need the outlines here
        rgi_ids = utils.get_rgi_glacier_entities(RGIs)

        # Go - we start from scratch, i.e. we cant download from Bremen
        gdirs = workflow.init_glacier_directories(rgi_ids)

        # # gdirs is a list of glaciers. Let's pick one
        for gdir in gdirs:
            # https://oggm.org/tutorials/stable/notebooks/dem_sources.html
            tasks.define_glacier_region(gdir, source="DEM3")
            # Glacier masks and all
            tasks.simple_glacier_masks(gdir)

        # https://oggm.org/tutorials/master/notebooks/oggm_shop.html
        # If you want data we havent processed yet, you have to use OGGM shop
        from oggm.shop.millan22 import (
            thickness_to_gdir,
            velocity_to_gdir,
            compile_millan_statistics,
        )

        # This applies a task to a list of gdirs
        workflow.execute_entity_task(thickness_to_gdir, gdirs)
        workflow.execute_entity_task(velocity_to_gdir, gdirs)

        from oggm.shop import bedtopo

        workflow.execute_entity_task(bedtopo.add_consensus_thickness, gdirs)

        # We also have some diagnostics if you want
        df = compile_millan_statistics(gdirs)
    #        print(df.T)

    path_ncdf = []
    for gdir in gdirs:
        path_ncdf.append(gdir.get_filepath("gridded_data"))

    return gdirs, path_ncdf


def _read_glathida(x, y, usurf, proj, path_glathida):
    """
    Function written by Ethan Welthy & Guillaume Jouvet
    """

    from pyproj import Transformer
    from scipy.interpolate import RectBivariateSpline
    import pandas as pd

    if not os.path.exists(os.path.join(path_glathida, "glathida")):
        os.system("git clone https://gitlab.com/wgms/glathida " + path_glathida)
    else:
        print("glathida data already at " + path_glathida)

    files = [os.path.join(path_glathida, "glathida", "data", "point.csv")]
    files += glob.glob(
        os.path.join(path_glathida, "glathida", "submissions", "*", "point.csv")
    )

    transformer = Transformer.from_crs(proj, "epsg:4326", always_xy=True)

    lonmin, latmin = transformer.transform(min(x), min(y))
    lonmax, latmax = transformer.transform(max(x), max(y))

    transformer = Transformer.from_crs("epsg:4326", proj, always_xy=True)

    #    print(x.shape, y.shape, usurf.shape)

    fsurf = RectBivariateSpline(x, y, np.transpose(usurf))

    df = pd.concat(
        [pd.read_csv(file, low_memory=False) for file in files], ignore_index=True
    )
    mask = (
        (lonmin <= df["lon"])
        & (df["lon"] <= lonmax)
        & (latmin <= df["lat"])
        & (df["lat"] <= latmax)
        & df["elevation"].notnull()
        & df["date"].notnull()
        & df["elevation_date"].notnull()
    )
    df = df[mask]

    # Filter by date gap in second step for speed
    mask = (
        (
            df["date"].str.slice(0, 4).astype(int)
            - df["elevation_date"].str.slice(0, 4).astype(int)
        )
        .abs()
        .le(1)
    )
    df = df[mask]

    if df.index.shape[0] == 0:
        print("No ice thickness profiles found")
        thkobs = np.ones_like(usurf)
        thkobs[:] = np.nan

    else:
        # Compute thickness relative to prescribed surface
        print("Nb of profiles found : ", df.index.shape[0])

        xx, yy = transformer.transform(df["lon"], df["lat"])
        bedrock = df["elevation"] - df["thickness"]
        elevation_normalized = fsurf(xx, yy, grid=False)
        thickness_normalized = np.maximum(elevation_normalized - bedrock, 0)

        # Rasterize thickness
        thickness_gridded = (
            pd.DataFrame(
                {
                    "col": np.floor((xx - np.min(x)) / (x[1] - x[0])).astype(int),
                    "row": np.floor((yy - np.min(y)) / (y[1] - y[0])).astype(int),
                    "thickness": thickness_normalized,
                }
            )
            .groupby(["row", "col"])["thickness"]
            .mean()
        )
        thkobs = np.full((y.shape[0], x.shape[0]), np.nan)
        thkobs[tuple(zip(*thickness_gridded.index))] = thickness_gridded

    return thkobs
