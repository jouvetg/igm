#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
This IGM modules compute time step dt (computed to satisfy the CFL condition),
updated time t, and a boolean telling whether results must be saved or not.
For stability reasons of the transport scheme for the ice thickness evolution,
the time step must respect a CFL condition, controlled by parameter params.cfl,
which is the maximum number of cells crossed in one iteration
(this parameter cannot exceed one).

==============================================================================

Input  : state.ubar, state.vbar, state.dx 
Output : state.dt, state.t, state.it, state.saveresult 
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime, time
import tensorflow as tf


def params_time_step(parser):
    parser.add_argument(
        "--tstart",
        type=float,
        default=2000.0,
        help="Start modelling time (default 2000)",
    )
    parser.add_argument(
        "--tend", type=float, default=2100.0, help="End modelling time (default: 2100)"
    )
    parser.add_argument(
        "--tsave", type=float, default=10, help="Save result each X years (default: 10)"
    )
    parser.add_argument(
        "--cfl",
        type=float,
        default=0.3,
        help="CFL number for the stability of the mass conservation scheme, \
        it must be below 1 (Default: 0.3)",
    )
    parser.add_argument(
        "--dtmax",
        type=float,
        default=10.0,
        help="Maximum time step allowed, used only with slow ice (default: 10.0)",
    )


def init_time_step(params, state):
    state.tcomp_time_step = []

    # Initialize the time with starting time
    state.t = tf.Variable(float(params.tstart))

    state.it = 0

    state.dt = tf.Variable(float(params.dtmax))

    state.dt_target = tf.Variable(float(params.dtmax))

    state.tsave = np.ndarray.tolist(
        np.arange(params.tstart, params.tend, params.tsave)
    ) + [params.tend]

    state.tsave = tf.constant(state.tsave)

    state.itsave = 0

    state.saveresult = True


def update_time_step(params, state):

    state.logger.info(
        "Update DT from the CFL condition at time : " + str(state.t.numpy())
    )

    state.tcomp_time_step.append(time.time())

    # compute maximum ice velocitiy magnitude
    velomax = max(
        tf.math.reduce_max(tf.math.abs(state.ubar)),
        tf.math.reduce_max(tf.math.abs(state.vbar)),
    )

    # dt_target account for both cfl and dt_max
    if velomax > 0:
        state.dt_target = min(params.cfl * state.dx / velomax, params.dtmax)
    else:
        state.dt_target = params.dtmax

    state.dt = state.dt_target

    # modify dt such that times of requested savings are reached exactly
    if state.tsave[state.itsave + 1] <= state.t + state.dt:
        state.dt = state.tsave[state.itsave + 1] - state.t
        state.saveresult = True
        state.itsave += 1
    else:
        state.saveresult = False

    state.t.assign(state.t + state.dt)

    state.it += 1

    state.tcomp_time_step[-1] -= time.time()
    state.tcomp_time_step[-1] *= -1


def final_time_step(params, state):
    pass
