#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file 

import numpy as np
import datetime, time
import tensorflow as tf

from igm.modules.utils import *


def params_vert_flow(parser):
    pass


def initialize_vert_flow(params, state):
    state.tcomp_vert_flow=[]

#    state.W = tf.zeros_like(state.U[0])


def update_vert_flow(params, state):
    """

    """

    state.tcomp_vert_flow.append(time.time())

    state.W = _compute_vertical_velocity_tf(params, state, state.U, state.thk, state.dX)
    
    state.wvelbase = state.W[0]
    state.wvelsurf = state.W[-1]

    state.tcomp_vert_flow[-1] -= time.time()
    state.tcomp_vert_flow[-1] *= -1
    

def finalize_vert_flow(params, state):
    pass


# @tf.function(experimental_relax_shapes=True)
def _compute_vertical_velocity_tf(params, state, U, thk, dX):

    # Compute horinzontal derivatives
    dUdx = (U[0, :, :, 2:] - U[0, :, :, :-2]) / (2 * dX[0, 0])
    dVdy = (U[1, :, 2:, :] - U[1, :, :-2, :]) / (2 * dX[0, 0])

    dUdx = tf.pad(dUdx, [[0, 0], [0, 0], [1, 1]], "CONSTANT")
    dVdy = tf.pad(dVdy, [[0, 0], [1, 1], [0, 0]], "CONSTANT")

    dUdx = (dUdx[1:] + dUdx[:-1]) / 2  # compute between the layers
    dVdy = (dVdy[1:] + dVdy[:-1]) / 2  # compute between the layers

    # get dVdz from impcrompressibility condition
    dVdz = - dUdx - dVdy

    # get the basal vertical velocities
    sloptopgx, sloptopgy = compute_gradient_tf(state.topg, state.dx, state.dx)
    wvelbase = state.U[0, 0] * sloptopgx + state.U[1, 0] * sloptopgy

    # get the vertical thickness layers
    zeta = np.arange(params.iflo_Nz) / (params.iflo_Nz - 1)
    temp = (zeta / params.iflo_vert_spacing) * (1.0 + (params.iflo_vert_spacing - 1.0) * zeta)
    temd = temp[1:] - temp[:-1]
    dz = tf.stack([state.thk * z for z in temd], axis=0)

    W = []
    W.append(wvelbase)
    for l in range(dVdz.shape[0]):
        W.append(W[-1] + dVdz[l] * dz[l])
    W = tf.stack(W)

    return W
