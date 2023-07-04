#!/usr/bin/env python3

"""
Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
Published under the GNU GPL (Version 3), check at the LICENSE file
"""


import numpy as np
import datetime, time
import tensorflow as tf

from igm.modules.utils import *


def params_vertical_iceflow(parser):
    pass


def init_vertical_iceflow(params, self):
    pass


def update_vertical_iceflow(params, self):
    """
    Updtae the third component of the velocity
    """

    self.tcomp["vertical_iceflow"].append(time.time())

    self.W = compute_vertical_velocity_tf(params, self, self.U, self.thk, self.dX)

    self.tcomp["vertical_iceflow"][-1] -= time.time()
    self.tcomp["vertical_iceflow"][-1] *= -1


@tf.function(experimental_relax_shapes=True)
def compute_vertical_velocity_tf(self, U, thk, dX):
    """
    Compute the vertical component of the velocity
    by integrating the imcompressibility condition
    """

    # Compute horinzontal derivatives
    dUdx = (U[0, :, :, 2:] - U[0, :, :, :-2]) / (2 * dX[0, 0])
    dVdy = (U[1, :, 2:, :] - U[1, :, :-2, :]) / (2 * dX[0, 0])

    dUdx = tf.pad(dUdx, [[0, 0], [0, 0], [1, 1]], "CONSTANT")
    dVdy = tf.pad(dVdy, [[0, 0], [1, 1], [0, 0]], "CONSTANT")

    dUdx = (dUdx[1:] + dUdx[:-1]) / 2  # compute between the layers
    dVdy = (dVdy[1:] + dVdy[:-1]) / 2  # compute between the layers

    # get dVdz from impcrompressibility condition
    dVdz = -dUdx - dVdy

    # get the basal vertical velocities
    sloptopgx, sloptopgy = compute_gradient_tf(self.topg, self.dx, self.dx)
    wvelbase = self.U[0, 0] * sloptopgx + self.U[1, 0] * sloptopgy

    # get the vertical thickness layers
    zeta = np.arange(params.Nz) / (params.Nz - 1)
    temp = (zeta / params.vert_spacing) * (1.0 + (params.vert_spacing - 1.0) * zeta)
    temd = temp[1:] - temp[:-1]
    dz = tf.stack([self.thk * z for z in temd], axis=0)

    W = []
    W.append(wvelbase)
    for l in range(dVdz.shape[0]):
        W.append(W[-1] + dVdz[l] * dz[l])
    W = tf.stack(W)

    return W
