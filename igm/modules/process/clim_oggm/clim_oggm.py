#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os, sys, shutil
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from netCDF4 import Dataset
import json
from igm.modules.utils import interp1d_tf


def params(parser):
    # CLIMATE PARAMETERS
    parser.add_argument(
        "--clim_oggm_update_freq",
        type=float,
        default=1,
        help="Update the climate each X years",
    )
    parser.add_argument(
        "--smb_oggm_file",
        type=str,
        default="smb_oggm_param.txt",
        help="Name of the imput file for the climate outide the given datatime frame (time, delta_temp, prec_scali)",
    )
    parser.add_argument(
        "--clim_oggm_clim_trend_array",
        type=list,
        default=[
            ["time", "delta_temp", "prec_scal"],
            [1900, 0.0, 1.0],
            [2020, 0.0, 1.0],
        ],
        help="Define climate trend outside available time window",
    )
    parser.add_argument(
        "--clim_oggm_ref_period",
        type=list,
        default=[1960, 1990],
        help="Define the reference period to pick year outside available time window",
    )
    parser.add_argument(
        "--clim_oggm_seed_par",
        type=list,
        default=123,
        help="Seeding parameter to fix for pickying randomly yer in the ref period",
    )


def initialize(params, state):
    # load the given parameters from the json file
    
    with open(os.path.join(params.oggm_RGI_ID, "mb_calib.json"), "r") as json_file:
        jsonString = json_file.read()

    oggm_mb_calib = json.loads(jsonString)

    params.temp_default_gradient = oggm_mb_calib["mb_global_params"][
        "temp_default_gradient"
    ]
    params.temp_bias = oggm_mb_calib["temp_bias"]
    params.prcp_fac = oggm_mb_calib["prcp_fac"]

    # load climate data from netcdf file climate_historical.nc
    nc = Dataset(
        os.path.join(params.oggm_RGI_ID, "climate_historical.nc")
    )

    time = np.squeeze(nc.variables["time"]).astype("float32")  # unit : year
    prcp = np.squeeze(nc.variables["prcp"]).astype("float32")  # unit : kg * m^(-2)
    temp = np.squeeze(nc.variables["temp"]).astype("float32")  # unit : degree celcius
    temp_std = np.squeeze(nc.variables["temp_std"]).astype(
        "float32"
    )  # unit : degree celcius

    params.ref_hgt = nc.ref_hgt
    params.yr_0 = nc.yr_0

    nc.close()

    # reshape climate data per year and month
    nb_y = int(time.shape[0] / 12)
    nb_m = 12

    state.prec = prcp.reshape((nb_y, nb_m))
    state.temp = temp.reshape((nb_y, nb_m))
    state.temp_std = temp_std.reshape((nb_y, nb_m))

    # correct the temperature and precipitation with factor and bias
    state.temp = state.temp + params.temp_bias
    state.prec = state.prec * params.prcp_fac

    # fix the units of precipitation
    state.prec = nb_m * state.prec  # kg * m^(-2) * month^(-1) ->  kg * m^(-2) * y^(-1)

    # intitalize air_temp and precipitation fields
    state.air_temp = tf.Variable(
        tf.zeros((nb_m, state.y.shape[0], state.x.shape[0])),
        dtype="float32",
    )
    state.air_temp_std = tf.Variable(
        tf.zeros((nb_m, state.y.shape[0], state.x.shape[0])),
        dtype="float32",
    )
    state.precipitation = tf.Variable(
        tf.zeros((nb_m, state.y.shape[0], state.x.shape[0])),
        dtype="float32",
    )

    state.tlast_clim_oggm = tf.Variable(-(10**10), dtype="float32")
    state.tcomp_clim_oggm = []

    if params.clim_oggm_clim_trend_array == []:
        state.climpar = np.loadtxt(
            params.clim_oggm_file,
            skiprows=1,
            dtype=np.float32,
        )
    else:
        state.climpar = np.array(params.clim_oggm_clim_trend_array[1:]).astype(
            np.float32
        )

    np.random.seed(params.clim_oggm_seed_par)  # fix the seed


def update(params, state):
    if (state.t - state.tlast_clim_oggm) >= params.clim_oggm_update_freq:
        if hasattr(state, "logger"):
            state.logger.info("update climate at time : " + str(state.t.numpy()))

        state.tcomp_clim_oggm.append(time.time())

        # find out the index that corresponds to the current year
        index_year = int(state.t - params.yr_0)

        if (index_year >= 0) & (index_year < state.prec.shape[0]):
            II = index_year
            delta_temp = 0.0
            prec_scal = 1.0
        else:
            i0, i1 = np.round(params.clim_oggm_ref_period - params.yr_0)
            II = np.random.randint(i0, i1)
            delta_temp = interp1d_tf(state.climpar[:, 0], state.climpar[:, 1], state.t)
            prec_scal = interp1d_tf(state.climpar[:, 0], state.climpar[:, 2], state.t)

        PREC = tf.expand_dims(
            tf.expand_dims(np.squeeze(state.prec[II, :]), axis=-1), axis=-1
        )
        TEMP = tf.expand_dims(
            tf.expand_dims(np.squeeze(state.temp[II, :]), axis=-1), axis=-1
        )
        TEMP_STD = tf.expand_dims(
            tf.expand_dims(np.squeeze(state.temp_std[II, :]), axis=-1), axis=-1
        )

        # apply delta temp and precp scaling
        TEMP += delta_temp
        PREC *= prec_scal

        # extend air_temp and precipitation over the entire glacier and all day of the year
        state.precipitation = tf.tile(PREC, (1, state.y.shape[0], state.x.shape[0]))
        state.air_temp = tf.tile(TEMP, (1, state.y.shape[0], state.x.shape[0]))
        state.air_temp_std = tf.tile(TEMP_STD, (1, state.y.shape[0], state.x.shape[0]))

        # vertical correction (lapse rates)
        temp_corr_addi = params.temp_default_gradient * (state.usurf - params.ref_hgt)
        temp_corr_addi = tf.expand_dims(temp_corr_addi, axis=0)
        temp_corr_addi = tf.tile(temp_corr_addi, (state.temp.shape[1], 1, 1))

        # the final precipitation and temperature must have shape (12,ny,nx)
        state.air_temp = state.air_temp + temp_corr_addi
        
        state.meanprec = tf.math.reduce_mean(state.precipitation, axis=0)
        state.meantemp = tf.math.reduce_mean(state.air_temp, axis=0)

        state.tlast_clim_oggm.assign(state.t)

        state.tcomp_clim_oggm[-1] -= time.time()
        state.tcomp_clim_oggm[-1] *= -1


def finalize(params, state):
    pass
