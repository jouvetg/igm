#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os, sys, shutil
import matplotlib.pyplot as plt
import datetime, time
import tensorflow as tf
import igm
from netCDF4 import Dataset

from igm.modules.utils import *

from igm.modules.particles_v1.particles_v1 import seeding_particles

# from igm.modules.process.particles_v1.particles_v1 import seeding_particles


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
        default=50,
        help="Frequency of seeding (unit : year)",
    )
    parser.add_argument(
        "--part_density_seeding",
        type=float,
        default=0.2,
        help="Density of seeding (1 means we seed all pixels, 0.2 means we seed each 5 grid cell, ect.)",
    )
    parser.add_argument(
        "--tlast_seeding_init",
        type=int,
        default=-1.0e5000,
        help="Initialize the date of last seeding. If default value, the seeding will start the first year of the simulation. Changing this value alouds to differ it",
    )


def initialize(cfg, state):
    state.tlast_seeding = -1.0e5000
    state.tcomp_particles = []

    # initialize trajectories
    state.particle_x = tf.Variable([])
    state.particle_y = tf.Variable([])
    state.particle_z = tf.Variable([])
    state.particle_r = tf.Variable([])
    state.particle_w = tf.Variable([])  # this is to give a weight to the particle
    state.particle_t = tf.Variable([])
    state.particle_englt = tf.Variable([])  # this computes the englacial time
    state.particle_topg = tf.Variable([])
    state.particle_thk = tf.Variable([])

    state.pswvelbase = tf.Variable(tf.zeros_like(state.thk), trainable=False)
    state.pswvelsurf = tf.Variable(tf.zeros_like(state.thk), trainable=False)

    # build the gridseed, we don't want to seed all pixels!
    state.gridseed = np.zeros_like(state.thk) == 1
    # uniform seeding on the grid
    rr = int(1.0 / cfg.modules.particles.part_density_seeding)
    state.gridseed[::rr, ::rr] = True


def update(cfg, state):

    if 'iceflow' not in cfg.modules:
        raise ValueError("The 'iceflow' module is required to use the particles module")

    if hasattr(state, "logger"):
        state.logger.info("Update particle tracking at time : " + str(state.t.numpy()))

    if (
        state.t.numpy() - state.tlast_seeding
    ) >= cfg.modules.particles.part_frequency_seeding:
        print(cfg.modules)
        print(dir(state))
        seeding_particles(cfg, state)

        # merge the new seeding points with the former ones
        state.particle_x = tf.Variable(
            tf.concat([state.particle_x, state.nparticle_x], axis=-1), trainable=False
        )
        state.particle_y = tf.Variable(
            tf.concat([state.particle_y, state.nparticle_y], axis=-1), trainable=False
        )
        state.particle_z = tf.Variable(
            tf.concat([state.particle_z, state.nparticle_z], axis=-1), trainable=False
        )
        state.particle_r = tf.Variable(
            tf.concat([state.particle_r, state.nparticle_r], axis=-1), trainable=False
        )
        state.particle_w = tf.Variable(
            tf.concat([state.particle_w, state.nparticle_w], axis=-1), trainable=False
        )
        state.particle_t = tf.Variable(
            tf.concat([state.particle_t, state.nparticle_t], axis=-1), trainable=False
        )
        state.particle_englt = tf.Variable(
            tf.concat([state.particle_englt, state.nparticle_englt], axis=-1),
            trainable=False,
        )
        state.particle_topg = tf.Variable(
            tf.concat([state.particle_topg, state.nparticle_topg], axis=-1),
            trainable=False,
        )
        state.particle_thk = tf.Variable(
            tf.concat([state.particle_thk, state.nparticle_thk], axis=-1),
            trainable=False,
        )

        state.tlast_seeding = state.t.numpy()

    if (state.particle_x.shape[0] > 0) & (state.it >= 0):
        state.tcomp_particles.append(time.time())

        # find the indices of trajectories
        # these indicies are real values to permit 2D interpolations (particles are not necessary on points of the grid)
        i = (state.particle_x) / state.dx
        j = (state.particle_y) / state.dx

        indices = tf.expand_dims(
            tf.concat(
                [tf.expand_dims(j, axis=-1), tf.expand_dims(i, axis=-1)], axis=-1
            ),
            axis=0,
        )

        u = interpolate_bilinear_tf(
            tf.expand_dims(state.U, axis=-1),
            indices,
            indexing="ij",
        )[:, :, 0]

        v = interpolate_bilinear_tf(
            tf.expand_dims(state.V, axis=-1),
            indices,
            indexing="ij",
        )[:, :, 0]

        thk = interpolate_bilinear_tf(
            tf.expand_dims(tf.expand_dims(state.thk, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]
        state.particle_thk = thk

        topg = interpolate_bilinear_tf(
            tf.expand_dims(tf.expand_dims(state.topg, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]
        state.particle_topg = topg

        smb = interpolate_bilinear_tf(
            tf.expand_dims(tf.expand_dims(state.smb, axis=0), axis=-1),
            indices,
            indexing="ij",
        )[0, :, 0]

        zeta = _rhs_to_zeta(cfg, state.particle_r)  # get the position in the column
        I0 = tf.cast(
            tf.math.floor(zeta * (cfg.modules.iceflow.iceflow.iflo_Nz - 1)),
            dtype="int32",
        )
        I0 = tf.minimum(
            I0, cfg.modules.iceflow.iceflow.iflo_Nz - 2
        )  # make sure to not reach the upper-most pt
        I1 = I0 + 1
        zeta0 = tf.cast(I0 / (cfg.modules.iceflow.iceflow.iflo_Nz - 1), dtype="float32")
        zeta1 = tf.cast(I1 / (cfg.modules.iceflow.iceflow.iflo_Nz - 1), dtype="float32")

        lamb = (zeta - zeta0) / (zeta1 - zeta0)

        ind0 = tf.transpose(tf.stack([I0, tf.range(I0.shape[0])]))
        ind1 = tf.transpose(tf.stack([I1, tf.range(I1.shape[0])]))

        wei = tf.zeros_like(u)
        wei = tf.tensor_scatter_nd_add(wei, indices=ind0, updates=1 - lamb)
        wei = tf.tensor_scatter_nd_add(wei, indices=ind1, updates=lamb)

        if cfg.modules.particles.part_tracking_method == "simple":
            # adjust the relative height within the ice column with smb
            state.particle_r = tf.where(
                thk > 0.1,
                tf.clip_by_value(state.particle_r * (thk - smb * state.dt) / thk, 0, 1),
                1,
            )

            state.particle_x = state.particle_x + state.dt * tf.reduce_sum(
                wei * u, axis=0
            )
            state.particle_y = state.particle_y + state.dt * tf.reduce_sum(
                wei * v, axis=0
            )
            state.particle_z = topg + thk * state.particle_r

        elif cfg.modules.particles.part_tracking_method == "3d":
            # uses the vertical velocity w computed in the vert_flow module

            w = interpolate_bilinear_tf(
                tf.expand_dims(state.W, axis=-1),
                indices,
                indexing="ij",
            )[:, :, 0]

            state.particle_x = state.particle_x + state.dt * tf.reduce_sum(
                wei * u, axis=0
            )
            state.particle_y = state.particle_y + state.dt * tf.reduce_sum(
                wei * v, axis=0
            )
            state.particle_z = state.particle_z + state.dt * tf.reduce_sum(
                wei * w, axis=0
            )

            # make sure the particle vertically remain within the ice body
            state.particle_z = tf.clip_by_value(state.particle_z, topg, topg + thk)
            # relative height of the particle within the glacier
            state.particle_r = (state.particle_z - topg) / thk
            # if thk=0, state.rhpos takes value nan, so we set rhpos value to one in this case :
            state.particle_r = tf.where(
                thk == 0, tf.ones_like(state.particle_r), state.particle_r
            )

        else:
            print("Error : Name of the particles tracking method not recognised")

        # make sur the particle remains in the horiz. comp. domain
        state.particle_x = tf.clip_by_value(
            state.particle_x, 0, state.x[-1] - state.x[0]
        )
        state.particle_y = tf.clip_by_value(
            state.particle_y, 0, state.y[-1] - state.y[0]
        )

        indices = tf.concat(
            [
                tf.expand_dims(tf.cast(j, dtype="int32"), axis=-1),
                tf.expand_dims(tf.cast(i, dtype="int32"), axis=-1),
            ],
            axis=-1,
        )
        updates = tf.cast(
            tf.where(state.particle_r == 1, state.particle_w, 0), dtype="float32"
        )

        # this computes the sum of the weight of particles on a 2D grid
        state.weight_particles = tf.tensor_scatter_nd_add(
            tf.zeros_like(state.thk), indices, updates
        )

        # compute the englacial time
        state.particle_englt = state.particle_englt + tf.cast(
            tf.where(state.particle_r < 1, state.dt, 0.0), dtype="float32"
        )

        #    if int(state.t)%10==0:
        #        print("nb of part : ",state.xpos.shape)

        state.tcomp_particles[-1] -= time.time()
        state.tcomp_particles[-1] *= -1


def finalize(cfg, state):
    pass


def _zeta_to_rhs(cfg, zeta):
    return (zeta / cfg.modules.iceflow.iceflow.iflo_vert_spacing) * (
        1.0 + (cfg.modules.iceflow.iceflow.iflo_vert_spacing - 1.0) * zeta
    )


def _rhs_to_zeta(cfg, rhs):
    if cfg.modules.iceflow.iceflow.iflo_vert_spacing == 1:
        rhs = zeta
    else:
        DET = tf.sqrt(
            1
            + 4
            * (cfg.modules.iceflow.iceflow.iflo_vert_spacing - 1)
            * cfg.modules.iceflow.iceflow.iflo_vert_spacing
            * rhs
        )
        zeta = (DET - 1) / (2 * (cfg.modules.iceflow.iceflow.iflo_vert_spacing - 1))

    #           temp = cfg.modules.iceflow.iceflow.iflo_Nz*(DET-1)/(2*(cfg.modules.iceflow.iceflow.iflo_vert_spacing-1))
    #           I=tf.cast(tf.minimum(temp-1,cfg.modules.iceflow.iceflow.iflo_Nz-1),dtype='int32')

    return zeta


def seeding_particles(cfg, state):
    """
    here we define (xpos,ypos) the horiz coordinate of tracked particles
    and rhpos is the relative position in the ice column (scaled bwt 0 and 1)

    here we seed only the accum. area (a bit more), where there is
    significant ice, and in some points of a regular grid state.gridseed
    (density defined by density_seeding)

    """

    # ! THK and SMB modules are required. Insert in the init function of the particles module (actually, don't because the modules can be
    # ! initialized in any order, and the particles module is not guaranteed to be initialized after the thk and smb modules)
    # ! Instead, insert it HERE when needed (although it might call it multiple times and be less efficient...)

    if not hasattr(state, "thk"):
        raise ValueError("The thk module is required to use the particles module")
    if not hasattr(state, "smb"):
        raise ValueError(
            "A smb module is required to use the particles module. Please use the built-in smb module or create a custom one that overwrites the 'state.smb' value."
        )

    #        This will serve to remove imobile particles, but it is not active yet.

    #        indices = tf.expand_dims( tf.concat(
    #                       [tf.expand_dims((state.ypos - state.y[0]) / state.dx, axis=-1),
    #                        tf.expand_dims((state.xpos - state.x[0]) / state.dx, axis=-1)],
    #                       axis=-1 ), axis=0)

    #        thk = interpolate_bilinear_tf(
    #                    tf.expand_dims(tf.expand_dims(state.thk, axis=0), axis=-1),
    #                    indices,indexing="ij",      )[0, :, 0]

    #        J = (thk>1)

    # here we seed where i) thickness is higher than 1 m
    #                    ii) the seeding field of geology.nc is active
    #                    iii) on the gridseed (which permit to control the seeding density)
    #                    iv) on the accumulation area
    I = (
        (state.thk > 1) & state.gridseed & (state.smb > 0)
    ) # here you may redefine how you want to seed particles
    state.nparticle_x = state.X[I] - state.x[0]  # x position of the particle
    state.nparticle_y = state.Y[I] - state.y[0]  # y position of the particle
    state.nparticle_z = state.usurf[I]  # z position of the particle
    state.nparticle_r = tf.ones_like(state.X[I])  # relative position in the ice column
    state.nparticle_w = tf.ones_like(state.X[I])  # weight of the particle
    state.nparticle_t = (
        tf.ones_like(state.X[I]) * state.t
    )  # "date of birth" of the particle (useful to compute its age)
    state.nparticle_englt = tf.zeros_like(
        state.X[I]
    )  # time spent by the particle burried in the glacier
    state.nparticle_thk = state.thk[I]  # ice thickness at position of the particle
    state.nparticle_topg = state.topg[I]  # z position of the bedrock under the particle
