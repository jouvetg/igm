#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file 

import numpy as np
import os, sys, shutil
import matplotlib.pyplot as plt
import tensorflow as tf
import time

def params(parser):

    # CLIMATE PARAMETERS
    parser.add_argument(
        "--clim_update_freq",
        type=float,
        default=1,
        help="Update the climate each X years (default: 1)",
    )
    parser.add_argument(
        "--clim_time_resolution",
        type=float,
        default=365,
        help="Give the resolution the climate forcing should be (monthly=12, daily=365)",
    ) 

def  initialize(params,state):
    """
        load climate data to run the Aletsch Glacier simulation
    """
    
    # altitude of the weather station for climate data
    state.zws = 2766

    # read temperature and precipitation data from temp_prec.dat
    temp_prec = np.loadtxt(
        "temp_prec.dat",
        dtype=np.float32,
        skiprows=2,
    )

    # find the min and max years from data
    ymin = int(min(temp_prec[:, 0]))
    ymax = int(max(temp_prec[:, 0]))

    state.temp = np.zeros((365, ymax - ymin + 1), dtype=np.float32)
    state.prec = np.zeros((365, ymax - ymin + 1), dtype=np.float32)
    state.year = np.zeros((ymax - ymin + 1), dtype=np.float32)

    # retrieve temp [unit °C] and prec [unit is kg m^(-2) y^(-1) water eq] and year
    for k, y in enumerate(range(ymin, ymax + 1)):
        IND = (temp_prec[:, 0] == y) & (temp_prec[:, 1] <= 365)
        state.prec[:, k] = (
            temp_prec[IND, -1] * 365.0
        )  # new unit is  kg m^(-2) y^(-1) water eq
        state.temp[:, k] = temp_prec[IND, -2]  # new unit is °C
        state.year[k] = y

    # this make monthly temp and prec if this is wished
    if params.clim_time_resolution==12:
        II = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 364]
        state.prec = np.stack([np.mean(state.prec[II[i]:II[i+1]],axis=0) 
                                for i in range(0,12) ] )
        state.temp = np.stack([np.mean(state.temp[II[i]:II[i+1]],axis=0) 
                                for i in range(0,12) ] )

    # intitalize air_temp and precipitation fields
    state.air_temp = tf.Variable(
        tf.zeros((len(state.temp), state.y.shape[0], state.x.shape[0])),
        dtype="float32",
    )
    state.precipitation = tf.Variable(
        tf.zeros((len(state.prec), state.y.shape[0], state.x.shape[0])),
        dtype="float32",
    )

    state.tlast_clim_aletsch = tf.Variable(params.time_start)
    state.tcomp_clim_aletsch = []


def update(params,state):

    if ((state.t - state.tlast_clim_aletsch) >= params.clim_update_freq):

        if hasattr(state, "logger"):
            state.logger.info("update climate at time : " + str(state.t.numpy()))

        state.tcomp_clim_aletsch.append(time.time())

        dP = 0.00035   # Precipitation vertical gradient
        dT = -0.00552  # Temperature vertical gradient

        # find out the precipitation and temperature at the weather station
        II = state.year == int(state.t)
        PREC = tf.expand_dims(
            tf.expand_dims(np.squeeze(state.prec[:, II]), axis=-1), axis=-1
        )
        TEMP = tf.expand_dims(
            tf.expand_dims(np.squeeze(state.temp[:, II]), axis=-1), axis=-1
        )

        # extend air_temp and precipitation over the entire glacier and all day of the year
        state.precipitation = (tf.tile(PREC, (1, state.y.shape[0], state.x.shape[0])))
        state.air_temp      = (tf.tile(TEMP, (1, state.y.shape[0], state.x.shape[0])))

        # vertical correction (lapse rates)
        prec_corr_mult = 1 + dP * (state.usurf - state.zws)
        temp_corr_addi = dT * (state.usurf - state.zws)

        prec_corr_mult = tf.expand_dims(prec_corr_mult, axis=0)
        temp_corr_addi = tf.expand_dims(temp_corr_addi, axis=0)

        prec_corr_mult = tf.tile(prec_corr_mult, (len(state.prec), 1, 1))
        temp_corr_addi = tf.tile(temp_corr_addi, (len(state.temp), 1, 1))

        # the final precipitation and temperature must have shape (365,ny,ny)
        state.precipitation = (tf.clip_by_value(state.precipitation * prec_corr_mult, 0, 10 ** 10))
        state.air_temp      = (state.air_temp + temp_corr_addi)
        
        state.meanprec = tf.math.reduce_mean(state.precipitation, axis=0)
        state.meantemp = tf.math.reduce_mean(state.air_temp, axis=0)

        state.tlast_clim_aletsch.assign(state.t)

        state.tcomp_clim_aletsch[-1] -= time.time()
        state.tcomp_clim_aletsch[-1] *= -1


def  finalize(params,state):
    pass
 
