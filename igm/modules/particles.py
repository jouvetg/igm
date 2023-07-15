#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file 

"""
This IGM module implments a particle tracking routine, which can compute 
a large number of trajectories (as it is implemented with TensorFlow to 
run in parallel) in live time during the forward model run. The routine 
produces some seeding of particles (by default in the accumulation area
 at regular intervals), and computes the time trajectory of the resulting 
 particle in time advected by the velocity field in 3D. 
 There are currently 2 implementations:
* 'simple', Horizontal and vertical directions are treated differently: 
i) In the horizontal plan, particles are advected with the horizontal velocity 
field (interpolated bi-linearly). The positions are recorded in vector 
(glacier.xpos,glacier.ypos). ii) In the vertical direction, particles are 
tracked along the ice column scaled between 0 and 1 (0 at the bed, 1 at 
the top surface). The relative position along the ice column is recorded 
in vector glacier.rhpos (same dimension as glacier.xpos and iglaciergm.ypos). 
Particles are always initialized at 1 (assumed to be on the surface). 
The evolution of the particle within the ice column through time is 
computed according to the surface mass balance: the particle deepens when 
the surface mass balance is positive (then igm.rhpos decreases), 
and re-emerge when the surface mass balance is negative.
* '3d', The vertical velocity is reconsructed by integrating the divergence 
of the horizontal velocity, this permits in turn to perform 3D particle tracking. 
state.zpos is the z- position within the ice.
Note that in both case, the velocity in the ice layer is reconstructed from 
bottom and surface one assuming 4rth order polynomial profile (SIA-like)

To include this feature, make sure:
* To adapt the seeding to your need. You may keep the default seeding in the 
accumulation area setting the seeding frequency with igm.config.frequency_seeding 
and the seeding density glacier.config.density_seeding. Alternatively, you may 
define your own seeding strategy (e.g. seeding close to rock walls/nunataks). 
To do so, you may redefine the function seeding_particles.

* At each time step, the weight of surface debris contains in each cell the 2D
 horizontal grid is computed, and stored in variable igm.weight_particles.

==============================================================================

Input: U,W
Output: state.xpos, ...
"""

import numpy as np
import os, sys, shutil
import matplotlib.pyplot as plt
import datetime, time
import tensorflow as tf
import igm

from igm.modules.utils import *


def params_particles(parser):
    parser.add_argument(
        "--tracking_method",
        type=str,
        default="simple",
        help="Method for tracking particles (3d or simple)",
    )
    parser.add_argument(
        "--frequency_seeding",
        type=int,
        default=50,
        help="Frequency of seeding (default: 10)",
    )
    parser.add_argument(
        "--density_seeding",
        type=int,
        default=0.2,
        help="Density of seeding (default: 0.2)",
    )


def init_particles(params, state):
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
    rr = int(1.0 / params.density_seeding)
    state.gridseed[::rr, ::rr] = True


def update_particles(params, state):
 
    import tensorflow_addons as tfa

    state.logger.info("Update particle tracking at time : " + str(state.t.numpy()))

    if (state.t.numpy() - state.tlast_seeding) >= params.frequency_seeding:
        igm.seeding_particles(params, state)

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

    u = tfa.image.interpolate_bilinear(
        tf.expand_dims(state.U[0], axis=-1),
        indices,
        indexing="ij",
    )[:, :, 0]

    v = tfa.image.interpolate_bilinear(
        tf.expand_dims(state.U[1], axis=-1),
        indices,
        indexing="ij",
    )[:, :, 0]

    othk = tfa.image.interpolate_bilinear(
        tf.expand_dims(tf.expand_dims(state.thk, axis=0), axis=-1),
        indices,
        indexing="ij",
    )[0, :, 0]

    topg = tfa.image.interpolate_bilinear(
        tf.expand_dims(tf.expand_dims(state.topg, axis=0), axis=-1),
        indices,
        indexing="ij",
    )[0, :, 0]

    smb = tfa.image.interpolate_bilinear(
        tf.expand_dims(tf.expand_dims(state.smb, axis=0), axis=-1),
        indices,
        indexing="ij",
    )[0, :, 0]

    zeta = _rhs_to_zeta(params, state.rhpos)  # get the position in the column
    I0 = tf.cast(tf.math.floor(zeta * (params.Nz - 1)), dtype="int32")
    I0 = tf.minimum(I0, params.Nz - 2)  # make sure to not reach the upper-most pt
    I1 = I0 + 1
    zeta0 = tf.cast(I0 / (params.Nz - 1), dtype="float32")
    zeta1 = tf.cast(I1 / (params.Nz - 1), dtype="float32")

    lamb = (zeta - zeta0) / (zeta1 - zeta0)

    ind0 = tf.transpose(tf.stack([I0, tf.range(I0.shape[0])]))
    ind1 = tf.transpose(tf.stack([I1, tf.range(I1.shape[0])]))

    wei = tf.zeros_like(u)
    wei = tf.tensor_scatter_nd_add(wei, indices=ind0, updates=1 - lamb)
    wei = tf.tensor_scatter_nd_add(wei, indices=ind1, updates=lamb)

    if params.tracking_method == "simple":
        nthk = othk + smb * state.dt  # new ice thicnkess after smb update

        # adjust the relative height within the ice column with smb
        state.rhpos = tf.where(
            nthk > 0.1, tf.clip_by_value(state.rhpos * othk / nthk, 0, 1), 1
        )

        state.xpos = state.xpos + state.dt * tf.reduce_sum( wei * u, axis=0 ) 
        state.ypos = state.ypos + state.dt * tf.reduce_sum( wei * v, axis=0 ) 
        state.zpos = topg + nthk * state.rhpos

    elif params.tracking_method == "3d":
        # make sure the particle remian withi the ice body
        state.zpos = tf.clip_by_value(state.zpos, topg, topg + othk)

        # get the relative height
        state.rhpos = tf.where(othk > 0.1, (state.zpos - topg) / othk, 1)

        w = tfa.image.interpolate_bilinear(
                tf.expand_dims(state.W, axis=-1),
                indices,
                indexing="ij",
        )[:, :, 0]

        state.xpos = state.xpos + state.dt * tf.reduce_sum( wei * u, axis=0 )  
        state.ypos = state.ypos + state.dt * tf.reduce_sum( wei * v, axis=0 ) 
        state.zpos = state.zpos + state.dt * tf.reduce_sum( wei * w, axis=0 )

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
    

def final_particles(params, state):
    pass


def _zeta_to_rhs(state, zeta):
    return (zeta / params.vert_spacing) * (1.0 + (params.vert_spacing - 1.0) * zeta)


def _rhs_to_zeta(params, rhs):
    if params.vert_spacing == 1:
        rhs = zeta
    else:
        DET = tf.sqrt(1 + 4 * (params.vert_spacing - 1) * params.vert_spacing * rhs)
        zeta = (DET - 1) / (2 * (params.vert_spacing - 1))

    #           temp = params.Nz*(DET-1)/(2*(params.vert_spacing-1))
    #           I=tf.cast(tf.minimum(temp-1,params.Nz-1),dtype='int32')

    return zeta


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

    #        import tensorflow_addons as tfa

    #        thk = tfa.image.interpolate_bilinear(
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
