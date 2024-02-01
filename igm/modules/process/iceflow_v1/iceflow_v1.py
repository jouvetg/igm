#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os
import matplotlib.pyplot as plt
import datetime, time
import math
import tensorflow as tf
import importlib_resources

from igm.modules.utils import *
from igm import emulators


def params(parser):
    parser.add_argument(
        "--init_strflowctrl",
        type=float,
        default=78,
        help="Initial strflowctrl (default 78)",
    )

    parser.add_argument(
        "--iflo_emulator",
        type=str,
        default="f15_cfsflow_GJ_22_a",
        help="Directory path of the deep-learning ice flow model, \
              create a new if empty string",
    )
    parser.add_argument(
        "--iflo_init_slidingco",
        type=float,
        default=0,
        help="Initial sliding coeeficient slidingco (default: 0)",
    )
    parser.add_argument(
        "--iflo_init_arrhenius",
        type=float,
        default=78,
        help="Initial arrhenius factor arrhenuis (default: 78)",
    )
    parser.add_argument(
        "--iflo_multiple_window_size",
        type=int,
        default=0,
        help="If a U-net, this force window size a multiple of 2**N (default: 0)",
    )
    parser.add_argument(
        "--iflo_force_max_velbar",
        type=float,
        default=0,
        help="This permits to artif. upper-bound velocities, active if > 0 (default: 0)",
    )


def initialize(params, state):
    state.tcomp_iceflow = []

    # here we initialize variable parmaetrizing ice flow
    if not hasattr(state, "strflowctrl"):
        state.strflowctrl = tf.Variable(
            tf.ones_like(state.thk) * params.init_strflowctrl
        )

    if not hasattr(state, "arrhenius"):
        state.arrhenius = tf.Variable(
            tf.ones_like(state.thk) * params.iflo_init_arrhenius
        )

    if not hasattr(state, "slidingco"):
        state.slidingco = tf.Variable(
            tf.ones_like(state.thk) * params.iflo_init_slidingco
        )

    if os.path.exists(
        importlib_resources.files(emulators).joinpath(params.iflo_emulator)
    ):
        dirpath = importlib_resources.files(emulators).joinpath(params.iflo_emulator)
    else:
        dirpath = params.iflo_emulator

    dirpath = os.path.join(dirpath, str(int(state.dx)))

    # fieldin, fieldout, fieldbounds contains name of I/O variables, and bounds for scaling
    fieldin, fieldout, fieldbounds = _read_fields_and_bounds(state, dirpath)

    state.iceflow_mapping = {}
    state.iceflow_mapping["fieldin"] = fieldin
    state.iceflow_mapping["fieldout"] = fieldout
    state.iceflow_fieldbounds = fieldbounds

    state.iceflow_model = tf.keras.models.load_model(os.path.join(dirpath, "model.h5"))

    # print(state.iceflow_model.summary())

    Ny = state.thk.shape[0]
    Nx = state.thk.shape[1]

    # In case of a U-net, must make sure the I/O size is multiple of 2**N
    if params.iflo_multiple_window_size > 0:
        NNy = params.iflo_multiple_window_size * math.ceil(
            Ny / params.iflo_multiple_window_size
        )
        NNx = params.iflo_multiple_window_size * math.ceil(
            Nx / params.iflo_multiple_window_size
        )
        state.PAD = [[0, NNy - Ny], [0, NNx - Nx]]
    else:
        state.PAD = [[0, 0], [0, 0]]


def update(params, state):
    if hasattr(state, "logger"):
        state.logger.info("Update ICEFLOW at time : " + str(state.t.numpy()))

    state.tcomp_iceflow.append(time.time())

    # update gradients of the surface (slopes)
    state.slopsurfx, state.slopsurfy = compute_gradient_tf(
        state.usurf, state.dx, state.dx
    )

    # Define the input of the NN, include scaling
    X = tf.expand_dims(
        tf.stack(
            [
                tf.pad(vars(state)[f], state.PAD, "CONSTANT")
                / state.iceflow_fieldbounds[f]
                for f in state.iceflow_mapping["fieldin"]
            ],
            axis=-1,
        ),
        axis=0,
    )

    # Get the ice flow after applying the NN
    Y = state.iceflow_model(X)

    # Appplying scaling, and update variables
    Ny, Nx = state.thk.shape
    for kk, f in enumerate(state.iceflow_mapping["fieldout"]):
        vars(state)[f] = (
            tf.where(state.thk > 0, Y[0, :Ny, :Nx, kk], 0)
            * state.iceflow_fieldbounds[f]
        )

    # If requested, the speeds are artifically upper-bounded
    if params.iflo_force_max_velbar > 0:
        state.velbar_mag = state.getmag(state.ubar, state.vbar)

        state.ubar = tf.where(
            state.velbar_mag >= params.iflo_force_max_velbar,
            params.iflo_force_max_velbar * (state.ubar / state.velbar_mag),
            state.ubar,
        )
        state.vbar = tf.where(
            state.velbar_mag >= params.iflo_force_max_velbar,
            params.iflo_force_max_velbar * (state.vbar / state.velbar_mag),
            state.vbar,
        )

    state.tcomp_iceflow[-1] -= time.time()
    state.tcomp_iceflow[-1] *= -1


def finalize(params, state):
    pass


def _read_fields_and_bounds(state, path):
    fieldbounds = {}
    fieldin = []
    fieldout = []

    fid = open(os.path.join(path, "fieldin.dat"), "r")
    for fileline in fid:
        part = fileline.split()
        fieldin.append(part[0])
        fieldbounds[part[0]] = float(part[1])
    fid.close()

    fid = open(os.path.join(path, "fieldout.dat"), "r")
    for fileline in fid:
        part = fileline.split()
        fieldout.append(part[0])
        fieldbounds[part[0]] = float(part[1])
    fid.close()

    return fieldin, fieldout, fieldbounds
