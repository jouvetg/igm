#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os, time
import matplotlib.pyplot as plt
import tensorflow as tf
from netCDF4 import Dataset
from scipy.interpolate import RectBivariateSpline, interp1d
from igm.modules.utils import interp1d_tf
import igm


def params(parser):
    # CLIMATE PARAMETERS
    parser.add_argument(
        "--clim_update_freq",
        type=float,
        default=100,
        help="Update the climate each X years (default: 1)",
    )
    parser.add_argument(
        "--climate_0_file", type=str, default="data/climate.nc", help="climate file"
    )
    parser.add_argument(
        "--climate_1_file", type=str, default="data/climate1.nc", help="climate file"
    )
    parser.add_argument(
        "--climate_signal_file",
        type=str,
        default="data/GI.dat",
        help="climate signal file for GI",
    )


def initialize(params, state):
    load_climate_data_glacialindex(params, state)

    state.air_temp = tf.Variable(state.air_temp_snap[:, :, :, 0], dtype="float32")
    state.air_temp_sd = tf.Variable(state.air_temp_sd_snap[:, :, :, 0], dtype="float32")
    state.precipitation = tf.Variable(
        state.precipitation_snap[:, :, :, 0], dtype="float32"
    )
    state.tempsurfref = tf.Variable(state.tempsurfref_snap[:, :, 0], dtype="float32")
    state.LR = tf.Variable(state.LR_snap[0], dtype="float32")

    state.tlast_clim = tf.Variable(-1.0e50, dtype="float32")
    state.tcomp_clim = []


def update(params, state):
    if (state.t - state.tlast_clim) >= params.clim_update_freq:
        if hasattr(state, "logger"):
            state.logger.info("update climate at time : " + str(state.t.numpy()))

        state.tcomp_clim.append(time.time())

        s = interp1d_tf(state.signal[:, 0], state.signal[:, 1], state.t)

        state.LR.assign((1 - s) * state.LR_snap[0] + s * state.LR_snap[1])

        lapse_rate_cor = (
            state.tempsurfref_snap[:, :, 1] - state.tempsurfref_snap[:, :, 0]
        ) * (state.LR / 1000.0)
        air_temp_0_on_surf_1 = state.air_temp_snap[:, :, :, 0] - lapse_rate_cor

        state.air_temp.assign(
            (1 - s) * air_temp_0_on_surf_1 + s * state.air_temp_snap[:, :, :, 1]
        )
        state.precipitation.assign(
            (1 - s) * state.precipitation_snap[:, :, :, 0]
            + s * state.precipitation_snap[:, :, :, 1]
        )
        state.air_temp_sd.assign(
            (1 - s) * state.air_temp_sd_snap[:, :, :, 0]
            + s * state.air_temp_sd_snap[:, :, :, 1]
        )
        state.tempsurfref.assign(state.tempsurfref_snap[:, :, 1])

        lapse_rate_cor = (state.usurf - state.tempsurfref) * (state.LR / 1000.0)

        state.air_temp.assign(state.air_temp - lapse_rate_cor)

        state.tlast_clim.assign(state.t)

        state.tcomp_clim[-1] -= time.time()
        state.tcomp_clim[-1] *= -1


def finalize(params, state):
    pass


############################################################################################################


def load_climate_data_one_snapshot(params, state, filename):
    """
    load the ncdf climate file containing precipitation and temperatures
    """

    nc = Dataset(filename, "r")

    x = np.squeeze(nc.variables["x"]).astype("float32")
    y = np.squeeze(nc.variables["y"]).astype("float32")

    air_temp = np.squeeze(nc.variables["air_temp"]).astype("float32")
    precipitation = np.squeeze(nc.variables["precipitation"]).astype("float32")
    surf_clim_ref = np.squeeze(nc.variables["usurf"]).astype("float32")

    if "air_temp_sd" in [var for var in nc.variables]:
        air_temp_sd = np.squeeze(nc.variables["air_temp_sd"]).astype("float32")
    else:
        air_temp_sd = np.ones_like(air_temp) * params.temp_std

    air_temp_sd = np.where(air_temp_sd > 0, air_temp_sd, 5.0)

    if not (y.shape == state.y.shape) & (x.shape == state.x.shape):
        print("Resample climate data")

        air_tempN = np.zeros((air_temp.shape[0], state.y.shape[0], state.x.shape[0]))
        precipitationN = np.zeros(
            (air_temp.shape[0], state.y.shape[0], state.x.shape[0])
        )
        air_temp_sdN = np.zeros((air_temp.shape[0], state.y.shape[0], state.x.shape[0]))
        surf_clim_refN = np.zeros((state.y.shape[0], state.x.shape[0]))

        for i in range(len(air_temp)):
            air_tempN[i] = RectBivariateSpline(y, x, air_temp[i])(state.y, state.x)
            air_temp_sdN[i] = RectBivariateSpline(y, x, air_temp_sd[i])(
                state.y, state.x
            )
            precipitationN[i] = RectBivariateSpline(y, x, precipitation[i])(
                state.y, state.x
            )

        surf_clim_refN = RectBivariateSpline(y, x, surf_clim_ref)(state.y, state.x)

        air_temp = air_tempN
        air_temp_sd = air_temp_sdN
        precipitation = precipitationN
        surf_clim_ref = surf_clim_refN

    air_temp -= 273.15
    precipitation *= 31556952.0 / 910.0

    nc.close()

    return [air_temp, air_temp_sd, precipitation, surf_clim_ref]


def load_signal(params, state):
    """
    load signal file to force a transient climate
    """
    climate_signal_file = os.path.join(params.working_dir, params.climate_signal_file)

    state.signal = np.loadtxt(climate_signal_file, dtype=np.float32)


def interpolate_climate_over_a_year(climate_snapshot):
    """
    this applies interpolation, and shift to all climatic variables
    """
    climate_snapshot_out = []
    for cs in climate_snapshot:
        air_temp, air_temp_sd, precipitation, usurf, LR = cs

        # currently we do not resample in time, but keep monthly resolution to save memory
        # use 52 instead of 12 for weekly resolution, but this will increase memory usage
        air_temp = climate_upsampling_and_shift(air_temp, resampling=12)
        air_temp_sd = climate_upsampling_and_shift(air_temp_sd, resampling=12)
        precipitation = climate_upsampling_and_shift(precipitation, resampling=12)

        climate_snapshot_out.append([air_temp, air_temp_sd, precipitation, usurf, LR])

    return climate_snapshot_out


def load_climate_data_glacialindex(params, state):
    """
    load climate data and transient signal, interpolate weekly, and shift
    """

    climate_snapshot_0 = load_climate_data_one_snapshot(
        params, state, os.path.join(params.working_dir, params.climate_0_file)
    ) + [6.0]
    climate_snapshot_1 = load_climate_data_one_snapshot(
        params, state, os.path.join(params.working_dir, params.climate_1_file)
    ) + [5.74]
    climate_snapshot = [climate_snapshot_0, climate_snapshot_1]

    ##############

    climate_snapshot = interpolate_climate_over_a_year(climate_snapshot)

    load_signal(params, state)

    ##############

    state.air_temp_snap = tf.constant(
        np.concatenate(
            (
                np.expand_dims(climate_snapshot[0][0], axis=-1),
                np.expand_dims(climate_snapshot[1][0], axis=-1),
            ),
            axis=-1,
        ),
        dtype="float32",
    )
    state.air_temp_sd_snap = tf.constant(
        np.concatenate(
            (
                np.expand_dims(climate_snapshot[0][1], axis=-1),
                np.expand_dims(climate_snapshot[1][1], axis=-1),
            ),
            axis=-1,
        ),
        dtype="float32",
    )
    state.precipitation_snap = tf.constant(
        np.concatenate(
            (
                np.expand_dims(climate_snapshot[0][2], axis=-1),
                np.expand_dims(climate_snapshot[1][2], axis=-1),
            ),
            axis=-1,
        ),
        dtype="float32",
    )
    state.tempsurfref_snap = tf.constant(
        np.concatenate(
            (
                np.expand_dims(climate_snapshot[0][3], axis=-1),
                np.expand_dims(climate_snapshot[1][3], axis=-1),
            ),
            axis=-1,
        ),
        dtype="float32",
    )
    state.LR_snap = tf.constant(
        np.concatenate(
            (
                np.expand_dims(climate_snapshot[0][4], axis=-1),
                np.expand_dims(climate_snapshot[1][4], axis=-1),
            ),
            axis=-1,
        ),
        dtype="float32",
    )


def climate_upsampling_and_shift(field, resampling, shift=0.0):
    """
    temporally resample (up) and shift to hydrological year
    """

    if resampling == len(field):
        Y = field

    else:
        assert resampling > len(field)
        mid = (field[-1] + field[0]) / 2.0
        x = np.concatenate(([0], (np.arange(len(field)) + 0.5) / len(field), [1]))
        y = np.concatenate(([mid], field, [mid]))
        X = (np.arange(resampling) + 0.5) / resampling
        #            Y = CubicSpline(x, y, bc_type='periodic', axis=0)(X)
        Y = interp1d(x, y, kind="linear", axis=0)(X)

    # the shift serves to adjust to hyroloogical year, i.e. start Oct 1st
    if shift > 0:
        Y = np.roll(Y, int(len(Y) * (1 - shift)), axis=0)

    return Y


def climate_donwsampling_and_shift(field, resampling, shift=0.75):
    """
    temporally resample (down) and shift to hydrological year
    """

    assert resampling < len(field)

    k = resampling
    m = field.shape[0]
    o = field.shape[1:]
    Y = np.nanmean(field[: (m // k) * k].reshape((m // k, k) + o), axis=0)
    Z = np.nanstd(field[: (m // k) * k].reshape((m // k, k) + o), axis=0)
    Z[Z < 0.1] = 0.1  # ensure the std is not zero

    # the shift serves to adjust to hyroloogical year, i.e. start Oct 1st
    if shift > 0:
        Y = np.roll(Y, int(len(Y) * (1 - shift)), axis=0)
        Z = np.roll(Z, int(len(Z) * (1 - shift)), axis=0)

    return Y, Z
