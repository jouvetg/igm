#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from igm.modules.utils import complete_data, compute_gradient_tf
from igm.modules.process.enthalpy.enthalpy import vertically_discretize_tf
import igm

from igm.modules.process.time import params as params_time
 
from igm.modules.process.iceflow.iceflow import *


def params(parser):
    params_time(parser)
    igm.modules.process.iceflow.params(parser)


def initialize(params, state):
    # Initialize the time with starting time
    state.t = tf.Variable(float(params.time_start))

    state.it = 0

    state.dt = tf.Variable(float(params.time_step_max))

    state.dt_target = tf.Variable(float(params.time_step_max))

    state.time_save = np.ndarray.tolist(
        np.arange(params.time_start, params.time_end, params.time_save)
    ) + [params.time_end]

    state.time_save = tf.constant(state.time_save, dtype="float32")

    state.itsave = 0

    state.saveresult = True

    x = np.arange(-100, 101) * 1000  # make x-axis, lenght 100 km,
    y = np.arange(-100, 101) * 1000  # make y-axis, lenght 100 km,

    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    topg = np.zeros_like(X)
    usurf = np.maximum(1500 * (1 - (R / 100000) ** 2), topg)
    thk = usurf

    state.x = tf.constant(x.astype("float32"))
    state.y = tf.constant(y.astype("float32"))

    state.topg = tf.Variable(topg.astype("float32"))
    state.thk = tf.Variable(thk.astype("float32"))

    complete_data(state)

    state.depth, state.dz = vertically_discretize_tf(
        state.thk, params.iflo_Nz, params.iflo_vert_spacing
    )

    rho = 910.0  # ice density (g/m^3)
    g = 9.81  # earth's gravity (m/s^2)
    A = 78 * 10 ** (-18)  # rate factor (Pa^-3 y^-1)
    n = 3
    k = A * (rho * g) ** n  # Generic constant (y^-1 m^-3)

    state.U = tf.Variable(
        tf.zeros((params.iflo_Nz, state.thk.shape[0], state.thk.shape[1]))
    )
    state.V = tf.Variable(
        tf.zeros((params.iflo_Nz, state.thk.shape[0], state.thk.shape[1]))
    )

    thk = tf.tile(tf.expand_dims(state.thk, 0), (params.iflo_Nz, 1, 1))

    slopsurfx, slopsurfy = compute_gradient_tf(state.usurf, state.dx, state.dx)

    dsx = tf.tile(tf.expand_dims(slopsurfx, 0), (params.iflo_Nz, 1, 1))
    dsy = tf.tile(tf.expand_dims(slopsurfy, 0), (params.iflo_Nz, 1, 1))

    ds = (dsx**2 + dsy**2) ** 0.5

    state.U.assign(
        -2
        * A
        * ((rho * g) ** n)
        * ds ** (n - 1)
        * dsx
        * (thk ** (n + 1) - state.depth ** (n + 1))
    )
    state.V.assign(
        -2
        * A
        * ((rho * g) ** n)
        * ds ** (n - 1)
        * dsy
        * (thk ** (n + 1) - state.depth ** (n + 1))
    )

    state.air_temp = tf.expand_dims(tf.ones_like(state.usurf), axis=0) * (-10)

    state.arrhenius = tf.Variable(
        tf.ones((params.iflo_Nz, state.thk.shape[0], state.thk.shape[1]))
        * params.iflo_init_arrhenius
    )

    state.slidingco = tf.Variable(tf.ones_like(state.thk) * params.iflo_init_slidingco)

    define_vertical_weight(params, state)

    update_2d_iceflow_variables(params, state)


def update(params, state):
    state.dt = 1.0

    # modify dt such that times of requested savings are reached exactly
    if state.time_save[state.itsave + 1] <= state.t + state.dt:
        state.dt = state.time_save[state.itsave + 1] - state.t
        state.saveresult = True
        state.itsave += 1
    else:
        state.saveresult = False

    state.t.assign(state.t + state.dt)

    state.it += 1


def finalize(params, state):
    pass
