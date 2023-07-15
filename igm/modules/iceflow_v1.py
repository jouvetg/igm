#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file 

"""
This IGM module models ice flow using a Convolutional Neural Network
following the former online training from external data.

You may find trained and ready-to-use ice flow emulators in the folder
`emulators/T_M_I_Y_V/R/`, where 'T_M_I_Y_V' defines the emulator, and
R defines the spatial resolution. Make sure that the resolution of the
picked emulator is available in the database. Results produced with IGM
will strongly rely on the chosen emulator. Make sure that you use the
emulator within the hull of its training dataset (e.g., do not model
an ice sheet with an emulator trained with mountain glaciers) to ensure
reliability (or fidelity w.r.t to the instructor model) -- the emulator
is probably much better at interpolating than at extrapolating.
Information on the training dataset is provided in a dedicated README
coming along with the emulator.

At the time of writing, I recommend using *f15_cfsflow_GJ_22_a*, which
takes ice thickness, top surface slopes, the sliding coefficient c
('slidingco'), and Arrhenuis factor A ('arrhenius'), and return basal,
vertical-average and surface x- and y- velocity components.

I have trained *f15_cfsflow_GJ_22_a* using a large dataset of modeled
glaciers (based on a Stokes-based CfsFlow ice flow solver) and varying
sliding coefficient c, and Arrhenius factor A into a 2D space.

==============================================================================

Input: thk, usurf, arrhenuis, slidingco
Output: ubar,vbar, uvelsurf, vvelsurf, uvelbase, vvelbase
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import datetime, time
import math
import tensorflow as tf
import importlib_resources

from igm.modules.utils import *
from igm import emulators

def params_iceflow_v1(parser):
    parser.add_argument(
        "--init_strflowctrl",
        type=float,
        default=78,
        help="Initial strflowctrl (default 78)",
    )

    parser.add_argument(
        "--emulator",
        type=str,
        default="f15_cfsflow_GJ_22_a",
        help="Directory path of the deep-learning ice flow model, \
              create a new if empty string",
    )
    parser.add_argument(
        "--init_slidingco",
        type=float,
        default=0,
        help="Initial sliding coeeficient slidingco (default: 0)",
    )
    parser.add_argument(
        "--init_arrhenius",
        type=float,
        default=78,
        help="Initial arrhenius factor arrhenuis (default: 78)",
    )
    parser.add_argument(
        "--multiple_window_size",
        type=int,
        default=0,
        help="If a U-net, this force window size a multiple of 2**N (default: 0)",
    )
    parser.add_argument(
        "--force_max_velbar",
        type=float,
        default=0,
        help="This permits to artif. upper-bound velocities, active if > 0 (default: 0)",
    )


def init_iceflow_v1(params, state):

    state.tcomp_iceflow = []

    # here we initialize variable parmaetrizing ice flow
    if not hasattr(state, "strflowctrl"):
        state.strflowctrl = tf.Variable(tf.ones_like(state.thk) * params.init_strflowctrl)

    if not hasattr(state, "arrhenius"):
        state.arrhenius = tf.Variable(tf.ones_like(state.thk) * params.init_arrhenius)

    if not hasattr(state, "slidingco"):
        state.slidingco = tf.Variable(tf.ones_like(state.thk) * params.init_slidingco)
        
    if os.path.exists(importlib_resources.files(emulators).joinpath(params.emulator)):
        dirpath = importlib_resources.files(emulators).joinpath(params.emulator)
    else:
        dirpath = params.emulator

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
    if params.multiple_window_size > 0:
        NNy = params.multiple_window_size * math.ceil(Ny / params.multiple_window_size)
        NNx = params.multiple_window_size * math.ceil(Nx / params.multiple_window_size)
        state.PAD = [[0, NNy - Ny], [0, NNx - Nx]]
    else:
        state.PAD = [[0, 0], [0, 0]]


def update_iceflow_v1(params, state):

    state.logger.info("Update ICEFLOW at time : " + str(state.t.numpy()))

    state.tcomp_iceflow.append(time.time())

    # update gradients of the surface (slopes)
    state.slopsurfx, state.slopsurfy = compute_gradient_tf(state.usurf, state.dx, state.dx)

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
            tf.where(state.thk > 0, Y[0, :Ny, :Nx, kk], 0) * state.iceflow_fieldbounds[f]
        )

    # If requested, the speeds are artifically upper-bounded
    if params.force_max_velbar > 0:
        state.velbar_mag = state.getmag(state.ubar, state.vbar)

        state.ubar = tf.where(
            state.velbar_mag >= params.force_max_velbar,
            params.force_max_velbar * (state.ubar / state.velbar_mag),
            state.ubar,
        )
        state.vbar = tf.where(
            state.velbar_mag >= params.force_max_velbar,
            params.force_max_velbar * (state.vbar / state.velbar_mag),
            state.vbar,
        )

    state.tcomp_iceflow[-1] -= time.time()
    state.tcomp_iceflow[-1] *= -1


def final_iceflow_v1(params, state):
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
