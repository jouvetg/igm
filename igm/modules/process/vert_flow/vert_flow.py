#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import datetime, time
import tensorflow as tf

from igm.modules.utils import *


def params(parser):
    parser.add_argument(
        "--vflo_method",
        type=str,
        default="kinematic",
        help="Method to retrive the vertical velocity (kinematic, incompressibility)",
    )


def initialize(params, state):
    state.tcomp_vert_flow = []


#    state.W = tf.zeros_like(state.U[0])


def update(params, state):
    """ """

    state.tcomp_vert_flow.append(time.time())

    if params.vflo_method == "kinematic":
        state.W = _compute_vertical_velocity_kinematic(
            params, state, state.U, state.V, state.thk, state.dX
        )
    else:
        state.W = _compute_vertical_velocity_incompressibility(
            params, state, state.U, state.V, state.thk, state.dX
        )

    state.wvelbase = state.W[0]
    state.wvelsurf = state.W[-1]

    state.tcomp_vert_flow[-1] -= time.time()
    state.tcomp_vert_flow[-1] *= -1


def finalize(params, state):
    pass

def _compute_vertical_velocity_kinematic(params, state, U, V, thk, dX):
 
    # use the formula w = u dot \nabla l + \nable \cdot (u l)
 
    # get the vertical thickness layers
    zeta = np.arange(params.iflo_Nz) / (params.iflo_Nz - 1)
    temp = (zeta / params.iflo_vert_spacing) * (
        1.0 + (params.iflo_vert_spacing - 1.0) * zeta
    )
    temd = temp[1:] - temp[:-1]
    dz = tf.stack([state.thk * z for z in temd], axis=0)

    sloptopgx, sloptopgy = compute_gradient_tf(state.topg, state.dx, state.dx)
    
    sloplayx = [sloptopgx]
    sloplayy = [sloptopgy]
    divfl    = [tf.zeros_like(state.thk)]
    
    for l in range(1,U.shape[0]):

        cumdz = tf.reduce_sum(dz[:l], axis=0)
         
        sx, sy = compute_gradient_tf(state.topg + cumdz, state.dx, state.dx)
        
        sloplayx.append(sx)
        sloplayy.append(sy)

        ub = tf.reduce_sum(state.vert_weight[:l] * state.U[:l], axis=0) / tf.reduce_sum(state.vert_weight[:l], axis=0)
        vb = tf.reduce_sum(state.vert_weight[:l] * state.V[:l], axis=0) / tf.reduce_sum(state.vert_weight[:l], axis=0)         
        div = compute_divflux(ub, vb, cumdz, state.dx, state.dx, method='centered')

        divfl.append(div)
    
    sloplayx = tf.stack(sloplayx, axis=0)
    sloplayy = tf.stack(sloplayy, axis=0)
    divfl    = tf.stack(divfl, axis=0)
     
    W = state.U * sloplayx + state.V * sloplayy - divfl
    
    return W

# @tf.function(experimental_relax_shapes=True)
def _compute_vertical_velocity_incompressibility(params, state, U, V, thk, dX):
    # Compute horinzontal derivatives
    dUdx = (U[:, :, 2:] - U[:, :, :-2]) / (2 * dX[0, 0])
    dVdy = (V[:, 2:, :] - V[:, :-2, :]) / (2 * dX[0, 0])

    dUdx = tf.pad(dUdx, [[0, 0], [0, 0], [1, 1]], "CONSTANT")
    dVdy = tf.pad(dVdy, [[0, 0], [1, 1], [0, 0]], "CONSTANT")

    dUdx = (dUdx[1:] + dUdx[:-1]) / 2  # compute between the layers
    dVdy = (dVdy[1:] + dVdy[:-1]) / 2  # compute between the layers

    # get dVdz from impcrompressibility condition
    dVdz = -dUdx - dVdy

    # get the basal vertical velocities
    sloptopgx, sloptopgy = compute_gradient_tf(state.topg, state.dx, state.dx)
    wvelbase = state.U[0] * sloptopgx + state.V[0] * sloptopgy

    # get the vertical thickness layers
    zeta = np.arange(params.iflo_Nz) / (params.iflo_Nz - 1)
    temp = (zeta / params.iflo_vert_spacing) * (
        1.0 + (params.iflo_vert_spacing - 1.0) * zeta
    )
    temd = temp[1:] - temp[:-1]
    dz = tf.stack([state.thk * z for z in temd], axis=0)

    W = []
    W.append(wvelbase)
    for l in range(dVdz.shape[0]):
        W.append(W[-1] + dVdz[l] * dz[l])
    W = tf.stack(W)

    return W
