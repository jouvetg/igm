#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import matplotlib.pyplot as plt
import os, glob, shutil, scipy
from netCDF4 import Dataset
import tensorflow as tf
import pandas as pd
from igm.modules.utils import str2bool

from igm.modules.utils import complete_data


def params(parser):
    # aletsch  RGI60-11.01450
    # malspina RGI60-01.13696
    # brady    RGI60-01.20796
    # ethan    RGI60-01.00709

    parser.add_argument(
        "--oggm_RGI_ID", type=str, default="RGI60-11.01450", help="RGI ID"
    )
    parser.add_argument(
        "--oggm_preprocess", type=str2bool, default=True, help="Use preprocessing"
    )
    parser.add_argument(
        "--oggm_RGI_version", type=int, default=6, help="this is temporary fix, is 6 or 7"
    )
    parser.add_argument(
        "--oggm_dx",
        type=int,
        default=100,
        help="Spatial resolution (need preprocess false to change it)",
    )
    parser.add_argument(
        "--oggm_border",
        type=int,
        default=30,
        help="Safe border margin  (need preprocess false to change it)",
    )
    parser.add_argument(
        "--oggm_thk_source",
        type=str,
        default="consensus_ice_thickness",
        help="millan_ice_thickness or consensus_ice_thickness",
    )
    parser.add_argument(
        "--oggm_vel_source",
        type=str,
        default="millan_ice_velocity",
        help="Source of the surface velocities (millan_ice_velocity or its_live)",
    )
    parser.add_argument(
        "--oggm_incl_glathida",
        type=str2bool,
        default=False,
        help="Make observation file (for IGM inverse)",
    )
    parser.add_argument(
        "--oggm_path_glathida",
        type=str,
        default="",
        help="Path where the Glathida Folder is store, so that you don't need \
              to redownload it at any use of the script, if empty it will be in the home directory",
    )
    parser.add_argument(
        "--oggm_save_in_ncdf",
        type=str2bool,
        default=True,
        help="Write prepared data into a geology file",
    )


def initialize(params, state):
    import json

    _oggm_util([params.oggm_RGI_ID], params)

    if hasattr(state, "logger"):
        state.logger.info("Prepare data using oggm and glathida")

    nc = Dataset(os.path.join(params.oggm_RGI_ID, "gridded_data.nc"), "r+")

    x = np.squeeze(nc.variables["x"]).astype("float32")
    y = np.flip(np.squeeze(nc.variables["y"]).astype("float32"))

    thk = np.flipud(np.squeeze(nc.variables[params.oggm_thk_source]).astype("float32"))
    thk = np.where(np.isnan(thk), 0, thk)

    usurf = np.flipud(np.squeeze(nc.variables["topo"]).astype("float32"))

    icemask = np.flipud(np.squeeze(nc.variables["glacier_mask"]).astype("float32"))

    usurfobs = np.flipud(np.squeeze(nc.variables["topo"]).astype("float32"))
    icemaskobs = np.flipud(np.squeeze(nc.variables["glacier_mask"]).astype("float32"))

    vars_to_save = ["usurf", "thk", "icemask", "usurfobs", "thkobs", "icemaskobs"]

    if params.oggm_vel_source == "millan_ice_velocity":
        if "millan_vx" in nc.variables:
            uvelsurfobs = np.flipud(
                np.squeeze(nc.variables["millan_vx"]).astype("float32")
            )
            uvelsurfobs = np.where(np.isnan(uvelsurfobs), 0, uvelsurfobs)
            
            uvelsurfobs = np.where(icemaskobs, uvelsurfobs, 0)
            vars_to_save += ["uvelsurfobs"]
        if "millan_vy" in nc.variables:
            vvelsurfobs = np.flipud(
                np.squeeze(nc.variables["millan_vy"]).astype("float32")
            )
            vvelsurfobs = np.where(np.isnan(vvelsurfobs), 0, vvelsurfobs)            
            vvelsurfobs = np.where(icemaskobs, vvelsurfobs, 0)
            vars_to_save += ["vvelsurfobs"]
    else:
        if "itslive_vx" in nc.variables:
            uvelsurfobs = np.flipud(
                np.squeeze(nc.variables["itslive_vx"]).astype("float32")
            )
            uvelsurfobs = np.where(np.isnan(uvelsurfobs), 0, uvelsurfobs)
            uvelsurfobs = np.where(icemaskobs, uvelsurfobs, 0)
            vars_to_save += ["uvelsurfobs"]
        if "itslive_vy" in nc.variables:
            vvelsurfobs = np.flipud(
                np.squeeze(nc.variables["itslive_vy"]).astype("float32")
            )
            vvelsurfobs = np.where(np.isnan(vvelsurfobs), 0, vvelsurfobs)
            vvelsurfobs = np.where(icemaskobs, vvelsurfobs, 0)
            vars_to_save += ["vvelsurfobs"]

    uvelsurfobs = scipy.signal.medfilt2d(uvelsurfobs, kernel_size=3) # remove outliers
    vvelsurfobs = scipy.signal.medfilt2d(vvelsurfobs, kernel_size=3) # remove outliers

    if "millan_ice_thickness" in nc.variables:
        thkinit = np.flipud(
            np.squeeze(nc.variables["millan_ice_thickness"]).astype("float32")
        )
        thkinit = np.where(np.isnan(thkinit), 0, thkinit)
        thkinit = np.where(icemaskobs, thkinit, 0)
        vars_to_save += ["thkinit"]
        
    if "hugonnet_dhdt" in nc.variables:
        dhdt = np.flipud(
            np.squeeze(nc.variables["hugonnet_dhdt"]).astype("float32")
        )
        dhdt = np.where(np.isnan(dhdt), 0, dhdt)
        dhdt = np.where(icemaskobs, dhdt, 0)
        vars_to_save += ["dhdt"]

    thkobs = np.zeros_like(thk) * np.nan

    if params.oggm_incl_glathida:
        if params.oggm_RGI_version==6:
            with open(os.path.join(params.oggm_RGI_ID, "glacier_grid.json"), "r") as f:
                data = json.load(f)
            proj = data["proj"]
    
            try:
                thkobs = _read_glathida(
                    x, y, usurfobs, proj, params.oggm_path_glathida, state
                )
                thkobs = np.where(icemaskobs, thkobs, np.nan)
            except:
                thkobs = np.zeros_like(thk) * np.nan
        elif params.oggm_RGI_version==7:
            path_glathida = os.path.join(params.oggm_RGI_ID, "glathida_data.csv")
    
            try:
                thkobs = _read_glathida_v7(
                    x, y, path_glathida
                )
                thkobs = np.where(icemaskobs, thkobs, np.nan)
            except:
                thkobs = np.zeros_like(thk) * np.nan

    nc.close()

    ########################################################

    # transform from numpy to tensorflow

    for var in ["x", "y"]:
        vars(state)[var] = tf.constant(vars()[var].astype("float32"))

    for var in vars_to_save:
        vars(state)[var] = tf.Variable(vars()[var].astype("float32"))

    complete_data(state)

    ########################################################

    if params.oggm_save_in_ncdf:
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
        var_info["dhdt"] = ["Ice thickness change", "m/y"]

        nc = Dataset(
            os.path.join("input_saved.nc"), "w", format="NETCDF4"
        )

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


#        if hasattr(state,'logger'):
#            state.logger.setLevel(params.logging_level)


def update(params, state):
    pass


def finalize(params, state):
    pass


#########################################################################


def _oggm_util(RGIs, params):
    """
    Function written by Fabien Maussion
    """

    import oggm.cfg as cfg
    from oggm import utils, workflow, tasks, graphics

    if params.oggm_preprocess:
        # This uses OGGM preprocessed directories
        # I think that a minimal environment should be enough for this to run
        # Required packages:
        #   - numpy
        #   - geopandas
        #   - salem
        #   - matplotlib
        #   - configobj
        #   - netcdf4
        #   - xarray
        #   - oggm

        # Initialize OGGM and set up the default run parameters
        cfg.initialize_minimal()

        cfg.PARAMS["continue_on_error"] = False
        cfg.PARAMS["use_multiprocessing"] = False

        WD = "OGGM-prepro"

        # Where to store the data for the run - should be somewhere you have access to
        cfg.PATHS["working_dir"] = utils.gettempdir(dirname=WD, reset=True)

        # We need the outlines here
        if params.oggm_RGI_version==6:
            rgi_ids = RGIs  # rgi_ids = utils.get_rgi_glacier_entities(RGIs)
            base_url = ( "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/exps/igm_v2" )
            gdirs = workflow.init_glacier_directories(
                # Start from level 3 if you want some climate data in them
                rgi_ids,
                prepro_border=40,
                from_prepro_level=3,
                prepro_base_url=base_url,
            )
        else:
            rgi_ids = RGIs
            base_url = ( "https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/exps/igm_v3" )
            gdirs = workflow.init_glacier_directories(
                # Start from level 3 if you want some climate data in them
                rgi_ids,
                prepro_border=40,
                from_prepro_level=3,
                prepro_rgi_version='70C',
                prepro_base_url=base_url,
            )

    else:
        # Note: if you start from here you'll need most of the packages
        # needed by OGGM, since you start "from scratch" entirely
        # In my view this code should almost never be needed

        WD = "OGGM-dir"

        # Initialize OGGM and set up the default run parameters
        cfg.initialize()

        cfg.PARAMS["continue_on_error"] = False
        cfg.PARAMS["use_multiprocessing"] = False
        cfg.PARAMS["use_intersects"] = False

        # Map resolution parameters
        cfg.PARAMS["grid_dx_method"] = "fixed"
        cfg.PARAMS["fixed_dx"] = params.oggm_dx  # m spacing
        cfg.PARAMS[
            "border"
        ] = (
            params.oggm_border
        )  # can now be set to any value since we start from scratch
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
            compile_millan_statistics,
        )

        try:
            workflow.execute_entity_task(thickness_to_gdir, gdirs)
            workflow.execute_entity_task(velocity_to_gdir, gdirs)
        except ValueError:
            print("No millan22 velocity & thk data available!")

        # We also have some diagnostics if you want
        df = compile_millan_statistics(gdirs)
        #        print(df.T)

        from oggm.shop.its_live import velocity_to_gdir

        try:
            workflow.execute_entity_task(velocity_to_gdir, gdirs)
        except ValueError:
            print("No its_live velocity data available!")

        from oggm.shop import bedtopo

        workflow.execute_entity_task(bedtopo.add_consensus_thickness, gdirs)
        
        from oggm.shop import glathida

        workflow.execute_entity_task(glathida.glathida_to_gdir, gdirs)
        
        from oggm.shop.w5e5 import process_w5e5_data

        workflow.execute_entity_task(process_w5e5_data, gdirs)
        
        workflow.execute_entity_task(tasks.elevation_band_flowline, gdirs)
        workflow.execute_entity_task(tasks.fixed_dx_elevation_band_flowline, gdirs)
        workflow.execute_entity_task(tasks.mb_calibration_from_geodetic_mb,
                                                gdirs, informed_threestep=True)

    source_folder = gdirs[0].get_filepath("gridded_data").split("gridded_data.nc")[0]
    destination_folder = params.oggm_RGI_ID

    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder)
    shutil.copytree(source_folder, destination_folder)

    os.system( "echo rm -r " + params.oggm_RGI_ID + " >> clean.sh" )


def _read_glathida(x, y, usurf, proj, path_glathida, state):
    """
    Function written by Ethan Welthy, Guillaume Jouvet and Samuel Cook
    """

    from pyproj import Transformer
    from scipy.interpolate import RectBivariateSpline
    import pandas as pd

    if path_glathida == "":
        path_glathida = os.path.expanduser("~")

    if not os.path.exists(os.path.join(path_glathida, "glathida")):
        os.system("git clone https://gitlab.com/wgms/glathida " + path_glathida)
    else:
        if hasattr(state, "logger"):
            state.logger.info("glathida data already at " + path_glathida)

    files = glob.glob(os.path.join(path_glathida, "glathida", "data", "*", "point.csv"))
    files += glob.glob(os.path.join(path_glathida, "glathida", "data", "point.csv"))
   
    os.path.expanduser

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
        (lonmin <= df["longitude"])
        & (df["longitude"] <= lonmax)
        & (latmin <= df["latitude"])
        & (df["latitude"] <= latmax)
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
        if hasattr(state, "logger"):
            state.logger.info("Nb of profiles found : " + str(df.index.shape[0]))

        xx, yy = transformer.transform(df["longitude"], df["latitude"])
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
        thickness_gridded[thickness_gridded == 0] = np.nan
        thkobs[tuple(zip(*thickness_gridded.index))] = thickness_gridded

    return thkobs

def _read_glathida_v7(x, y, path_glathida):
    #Function written by Samuel Cook
    
    #Read GlaThiDa file
    gdf = pd.read_csv(path_glathida)
    
    gdf_sel = gdf.loc[gdf.thickness > 0]  # you may not want to do that, but be aware of: https://gitlab.com/wgms/glathida/-/issues/25
    gdf_per_grid = gdf_sel.groupby(by='ij_grid')[['i_grid', 'j_grid', 'elevation', 'thickness', 'thickness_uncertainty']].mean()  # just average per grid point
    # Average does not preserve ints
    gdf_per_grid['i_grid'] = gdf_per_grid['i_grid'].astype(int)
    gdf_per_grid['j_grid'] = gdf_per_grid['j_grid'].astype(int)
    
    #Get GlaThiDa data onto model grid  
    thkobs = np.full((y.shape[0], x.shape[0]), np.nan)
    thkobs[gdf_per_grid['j_grid'],gdf_per_grid['i_grid']] = gdf_per_grid['thickness']
    thkobs = np.flipud(thkobs)
    
    return thkobs
