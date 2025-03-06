#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import matplotlib.pyplot as plt
import datetime, time
import tensorflow as tf


# def params(parser):
#     parser.add_argument(
#         "--time_start",
#         type=float,
#         default=2000.0,
#         help="Start modelling time",
#     )
#     parser.add_argument(
#         "--time_end", type=float, default=2100.0, help="End modelling time"
#     )
#     parser.add_argument(
#         "--time_save",
#         type=float,
#         default=10,
#         help="Save result frequency for many modules (in year)",
#     )
#     parser.add_argument(
#         "--time_cfl",
#         type=float,
#         default=0.3,
#         help="CFL number for the stability of the mass conservation scheme, it must be below 1",
#     )
#     parser.add_argument(
#         "--time_step_max",
#         type=float,
#         default=1.0,
#         help="Maximum time step allowed, used only with slow ice",
#     )


def initialize(cfg, state):

    state.tcomp_time = []

    # Initialize the time with starting time
    state.t = tf.Variable(float(cfg.processes.time.start))

    # the first loop is not advancing
    state.it = -1
    state.itsave = -1

    state.dt = tf.Variable(float(cfg.processes.time.step_max))

    state.dt_target = tf.Variable(float(cfg.processes.time.step_max))

    state.time_save = np.ndarray.tolist(
        np.arange(cfg.processes.time.start, cfg.processes.time.end, cfg.processes.time.save)
    ) + [cfg.processes.time.end]

    state.time_save = tf.constant(state.time_save, dtype="float32")

    state.saveresult = True


def update(cfg, state):
    if hasattr(state, "logger"):
        state.logger.info(
            "Update DT from the CFL condition at time : " + str(state.t.numpy())
        )

    state.tcomp_time.append(time.time())

    # compute maximum ice velocitiy magnitude 
    velomax = tf.maximum(
        tf.reduce_max(tf.abs(state.ubar)),
        tf.reduce_max(tf.abs(state.vbar)),
    )
    # dt_target account for both cfl and dt_max
    if (velomax > 0) & (cfg.processes.time.cfl>0):
        state.dt_target =  tf.minimum(
            cfg.processes.time.cfl * state.dx / velomax, cfg.processes.time.step_max
        )
    else:
        state.dt_target = cfg.processes.time.step_max

    state.dt = state.dt_target

    # modify dt such that times of requested savings are reached exactly
    if state.time_save[state.itsave + 1] <= state.t + state.dt:
        state.dt = state.time_save[state.itsave + 1] - state.t
        state.saveresult = True
        state.itsave += 1
    else:
        state.saveresult = False

    # the first loop is not advancing
    if state.it >= 0:
        state.t.assign(state.t + state.dt)

    state.it += 1

    state.tcomp_time[-1] -= time.time()
    state.tcomp_time[-1] *= -1


def finalize(params, state):
    pass
