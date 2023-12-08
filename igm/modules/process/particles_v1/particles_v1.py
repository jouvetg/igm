#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os, sys, shutil
import matplotlib.pyplot as plt
import datetime, time
import tensorflow as tf

from igm.modules.utils import *


def params(parser):
    parser.add_argument(
        "--part_tracking_method",
        type=str,
        default="3d",
        help="Method for tracking particles (3d or simple)",
    )
    parser.add_argument(
        "--part_frequency_seeding",
        type=int,
        default=10,
        help="Frequency of seeding (default: 10)",
    )
    parser.add_argument(
        "--part_density_seeding",
        type=int,
        default=0.2,
        help="Density of seeding (default: 0.2)",
    )


def initialize(params, state):
    state.tlast_seeding = -1.0e5000
    state.tcomp_particles = []

    # initialize trajectories
    state.xpos = tf.Variable([])
    state.ypos = tf.Variable([])
    state.zpos = tf.Variable([])
    state.rhpos = tf.Variable([])
    state.wpos = tf.Variable([])  # this is to give a weight to the particle
    state.tpos = tf.Variable([])
    state.englt = tf.Variable([])

    # build the gridseed
    state.gridseed = np.zeros_like(state.thk) == 1
    rr = int(1.0 / params.part_density_seeding)
    state.gridseed[::rr, ::rr] = True


def update(params, state):
    if hasattr(state, "logger"):
        state.logger.info("Update particle tracking at time : " + str(state.t.numpy()))

    if (state.t.numpy() - state.tlast_seeding) >= params.part_frequency_seeding:
        seeding_particles(params, state)

        # merge the new seeding points with the former ones
        state.xpos = tf.Variable(tf.concat([state.xpos, state.nxpos], axis=-1))
        state.ypos = tf.Variable(tf.concat([state.ypos, state.nypos], axis=-1))
        state.zpos = tf.Variable(tf.concat([state.zpos, state.nzpos], axis=-1))
        state.rhpos = tf.Variable(tf.concat([state.rhpos, state.nrhpos], axis=-1))
        state.wpos = tf.Variable(tf.concat([state.wpos, state.nwpos], axis=-1))
        state.tpos = tf.Variable(tf.concat([state.tpos, state.ntpos], axis=-1))
        state.englt = tf.Variable(tf.concat([state.englt, state.nenglt], axis=-1))

        state.tlast_seeding = state.t.numpy()

    state.tcomp_particles.append(time.time())

    # find the indices of trajectories
    # these indicies are real values to permit 2D interpolations
    i = (state.xpos - state.x[0]) / state.dx
    j = (state.ypos - state.y[0]) / state.dx

    indices = tf.expand_dims(
        tf.concat([tf.expand_dims(j, axis=-1), tf.expand_dims(i, axis=-1)], axis=-1),
        axis=0,
    )

    uvelbase = interpolate_bilinear_tf(
        tf.expand_dims(tf.expand_dims(state.uvelbase, axis=0), axis=-1),
        indices,
        indexing="ij",
    )[0, :, 0]

    vvelbase = interpolate_bilinear_tf(
        tf.expand_dims(tf.expand_dims(state.vvelbase, axis=0), axis=-1),
        indices,
        indexing="ij",
    )[0, :, 0]

    uvelsurf = interpolate_bilinear_tf(
        tf.expand_dims(tf.expand_dims(state.uvelsurf, axis=0), axis=-1),
        indices,
        indexing="ij",
    )[0, :, 0]

    vvelsurf = interpolate_bilinear_tf(
        tf.expand_dims(tf.expand_dims(state.vvelsurf, axis=0), axis=-1),
        indices,
        indexing="ij",
    )[0, :, 0]

    othk = interpolate_bilinear_tf(
        tf.expand_dims(tf.expand_dims(state.thk, axis=0), axis=-1),
        indices,
        indexing="ij",
    )[0, :, 0]

    topg = interpolate_bilinear_tf(
        tf.expand_dims(tf.expand_dims(state.topg, axis=0), axis=-1),
        indices,
        indexing="ij",
    )[0, :, 0]

    smb = interpolate_bilinear_tf(
        tf.expand_dims(tf.expand_dims(state.smb, axis=0), axis=-1),
        indices,
        indexing="ij",
    )[0, :, 0]

    if params.part_tracking_method == "simple":
        nthk = othk + smb * state.dt  # new ice thicnkess after smb update

        # adjust the relative height within the ice column with smb
        state.rhpos = tf.where(
            nthk > 0.1, tf.clip_by_value(state.rhpos * othk / nthk, 0, 1), 1
        )

        uvel = uvelbase + (uvelsurf - uvelbase) * (
            1 - (1 - state.rhpos) ** 4
        )  # SIA-like
        vvel = vvelbase + (vvelsurf - vvelbase) * (
            1 - (1 - state.rhpos) ** 4
        )  # SIA-like

        state.xpos = state.xpos + state.dt * uvel  # forward euler
        state.ypos = state.ypos + state.dt * vvel  # forward euler

        state.zpos = topg + nthk * state.rhpos

    elif params.part_tracking_method == "3d":
        # This was a test of smoothing the surface topography to regaluraze the vertical velocitiy.
        #                import tensorflow_addons as tfa
        #                susurf = tfa.image.gaussian_filter2d(state.usurf, sigma=5, filter_shape=5, padding="CONSTANT")
        #                stopg  = tfa.image.gaussian_filter2d(state.topg , sigma=3, filter_shape=5, padding="CONSTANT")

        slopsurfx, slopsurfy = compute_gradient_tf(state.usurf, state.dx, state.dx)
        sloptopgx, sloptopgy = compute_gradient_tf(state.topg, state.dx, state.dx)

        state.divflux = compute_divflux(
            state.ubar, state.vbar, state.thk, state.dx, state.dx
        )

        # the vertical velocity is the scalar product of horizont. velo and bedrock gradient
        state.wvelbase = state.uvelbase * sloptopgx + state.vvelbase * sloptopgy
        # Using rules of derivative the surface vertical velocity can be found from the
        # divergence of the flux considering that the ice 3d velocity is divergence-free.
        state.wvelsurf = (
            state.uvelsurf * slopsurfx + state.vvelsurf * slopsurfy - state.divflux
        )

        wvelbase = interpolate_bilinear_tf(
            tf.expand_dims(tf.expand_dims(state.wvelbase, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]

        wvelsurf = interpolate_bilinear_tf(
            tf.expand_dims(tf.expand_dims(state.wvelsurf, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]

        #           print('at the surface? : ',all(state.zpos == topg+othk))

        # make sure the particle remian withi the ice body
        state.zpos = tf.clip_by_value(state.zpos, topg, topg + othk)

        # get the relative height
        state.rhpos = tf.where(othk > 0.1, (state.zpos - topg) / othk, 1)

        uvel = uvelbase + (uvelsurf - uvelbase) * (
            1 - (1 - state.rhpos) ** 4
        )  # SIA-like
        vvel = vvelbase + (vvelsurf - vvelbase) * (
            1 - (1 - state.rhpos) ** 4
        )  # SIA-like
        wvel = wvelbase + (wvelsurf - wvelbase) * (
            1 - (1 - state.rhpos) ** 4
        )  # SIA-like

        state.xpos = state.xpos + state.dt * uvel  # forward euler
        state.ypos = state.ypos + state.dt * vvel  # forward euler
        state.zpos = state.zpos + state.dt * wvel  # forward euler

        # make sur the particle remains in the horiz. comp. domain
        state.xpos = tf.clip_by_value(state.xpos, state.x[0], state.x[-1])
        state.ypos = tf.clip_by_value(state.ypos, state.y[0], state.y[-1])

    indices = tf.concat(
        [
            tf.expand_dims(tf.cast(j, dtype="int32"), axis=-1),
            tf.expand_dims(tf.cast(i, dtype="int32"), axis=-1),
        ],
        axis=-1,
    )
    updates = tf.cast(tf.where(state.rhpos == 1, state.wpos, 0), dtype="float32")

    # this computes the sum of the weight of particles on a 2D grid
    state.weight_particles = tf.tensor_scatter_nd_add(
        tf.zeros_like(state.thk), indices, updates
    )

    # compute the englacial time
    state.englt = state.englt + tf.cast(
        tf.where(state.rhpos < 1, state.dt, 0.0), dtype="float32"
    )

    state.tcomp_particles[-1] -= time.time()
    state.tcomp_particles[-1] *= -1


def finalize(params, state):
    pass


def seeding_particles(params, state):
    """
    here we define (xpos,ypos) the horiz coordinate of tracked particles
    and rhpos is the relative position in the ice column (scaled bwt 0 and 1)

    here we seed only the accum. area (a bit more), where there is
    significant ice, and in some points of a regular grid state.gridseed
    (density defined by density_seeding)

    """

    #        This will serve to remove imobile particles, but it is not active yet.

    #        indices = tf.expand_dims( tf.concat(
    #                       [tf.expand_dims((state.ypos - state.y[0]) / state.dx, axis=-1),
    #                        tf.expand_dims((state.xpos - state.x[0]) / state.dx, axis=-1)],
    #                       axis=-1 ), axis=0)

    #        thk = interpolate_bilinear_tf(
    #                    tf.expand_dims(tf.expand_dims(state.thk, axis=0), axis=-1),
    #                    indices,indexing="ij",      )[0, :, 0]

    #        J = (thk>1)

    I = (
        (state.thk > 10) & (state.smb > -2) & state.gridseed
    )  # seed where thk>10, smb>-2, on a coarse grid
    state.nxpos = state.X[I]
    state.nypos = state.Y[I]
    state.nzpos = state.usurf[I]
    state.nrhpos = tf.ones_like(state.X[I])
    state.nwpos = tf.ones_like(state.X[I])
    state.ntpos = tf.ones_like(state.X[I]) * state.t
    state.nenglt = tf.zeros_like(state.X[I])
