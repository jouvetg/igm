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
self.zpos is the z- position within the ice.
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
Output: self.xpos, ...
"""

import numpy as np
import os, sys, shutil
import matplotlib.pyplot as plt
import datetime, time
import tensorflow as tf

from igm.modules.utils import *


def params_particles(parser):
    parser.add_argument(
        "--tracking_method",
        type=str,
        default="3d",
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


def init_particles(params, self):
    self.tlast_seeding = -1.0e5000
    self.tcomp["particles"] = []

    # initialize trajectories
    self.xpos = tf.Variable([])
    self.ypos = tf.Variable([])
    self.zpos = tf.Variable([])
    self.rhpos = tf.Variable([])
    self.wpos = tf.Variable([])  # this is to give a weight to the particle
    self.tpos = tf.Variable([])
    self.englt = tf.Variable([])

    # build the gridseed
    self.gridseed = np.zeros_like(self.thk) == 1
    rr = int(1.0 / params.density_seeding)
    self.gridseed[::rr, ::rr] = True


def update_particles(params, self):
 
    import tensorflow_addons as tfa

    self.logger.info("Update particle tracking at time : " + str(self.t.numpy()))

    if (self.t.numpy() - self.tlast_seeding) >= params.frequency_seeding:
        seeding_particles(params, self)

        # merge the new seeding points with the former ones
        self.xpos = tf.Variable(tf.concat([self.xpos, self.nxpos], axis=-1))
        self.ypos = tf.Variable(tf.concat([self.ypos, self.nypos], axis=-1))
        self.zpos = tf.Variable(tf.concat([self.zpos, self.nzpos], axis=-1))
        self.rhpos = tf.Variable(tf.concat([self.rhpos, self.nrhpos], axis=-1))
        self.wpos = tf.Variable(tf.concat([self.wpos, self.nwpos], axis=-1))
        self.tpos = tf.Variable(tf.concat([self.tpos, self.ntpos], axis=-1))
        self.englt = tf.Variable(tf.concat([self.englt, self.nenglt], axis=-1))

        self.tlast_seeding = self.t.numpy()

    self.tcomp["particles"].append(time.time())

    # find the indices of trajectories
    # these indicies are real values to permit 2D interpolations
    i = (self.xpos - self.x[0]) / self.dx
    j = (self.ypos - self.y[0]) / self.dx

    indices = tf.expand_dims(
        tf.concat([tf.expand_dims(j, axis=-1), tf.expand_dims(i, axis=-1)], axis=-1),
        axis=0,
    )

    u = tfa.image.interpolate_bilinear(
        tf.expand_dims(self.U[0], axis=-1),
        indices,
        indexing="ij",
    )[:, :, 0]

    v = tfa.image.interpolate_bilinear(
        tf.expand_dims(self.U[1], axis=-1),
        indices,
        indexing="ij",
    )[:, :, 0]

    othk = tfa.image.interpolate_bilinear(
        tf.expand_dims(tf.expand_dims(self.thk, axis=0), axis=-1),
        indices,
        indexing="ij",
    )[0, :, 0]

    topg = tfa.image.interpolate_bilinear(
        tf.expand_dims(tf.expand_dims(self.topg, axis=0), axis=-1),
        indices,
        indexing="ij",
    )[0, :, 0]

    smb = tfa.image.interpolate_bilinear(
        tf.expand_dims(tf.expand_dims(self.smb, axis=0), axis=-1),
        indices,
        indexing="ij",
    )[0, :, 0]

    zeta = _rhs_to_zeta(params, self.rhpos)  # get the position in the column
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
        nthk = othk + smb * self.dt  # new ice thicnkess after smb update

        # adjust the relative height within the ice column with smb
        self.rhpos = tf.where(
            nthk > 0.1, tf.clip_by_value(self.rhpos * othk / nthk, 0, 1), 1
        )

        self.xpos = self.xpos + self.dt * tf.reduce_sum(
            wei * u, axis=0
        )  # forward euler
        self.ypos = self.ypos + self.dt * tf.reduce_sum(
            wei * v, axis=0
        )  # forward euler
        self.zpos = topg + nthk * self.rhpos

    elif params.tracking_method == "3d":
        method = 0

        # make sure the particle remian withi the ice body
        self.zpos = tf.clip_by_value(self.zpos, topg, topg + othk)

        # get the relative height
        self.rhpos = tf.where(othk > 0.1, (self.zpos - topg) / othk, 1)

        if method == 0:
            # This is the former working methd

            slopsurfx, slopsurfy = compute_gradient_tf(self.usurf, self.dx, self.dx)
            sloptopgx, sloptopgy = compute_gradient_tf(self.topg, self.dx, self.dx)

            self.divflux = compute_divflux(
                self.ubar, self.vbar, self.thk, self.dx, self.dx
            )

            # the vertical velocity is the scalar product of horizont. velo and bedrock gradient
            self.wvelbase = self.U[0, 0] * sloptopgx + self.U[1, 0] * sloptopgy
            # Using rules of derivative the surface vertical velocity can be found from the
            # divergence of the flux considering that the ice 3d velocity is divergence-free.
            self.wvelsurf = (
                self.U[0, -1] * slopsurfx + self.U[1, -1] * slopsurfy - self.divflux
            )

            wvelbase = tfa.image.interpolate_bilinear(
                tf.expand_dims(tf.expand_dims(self.wvelbase, axis=0), axis=-1),
                indices,
                indexing="ij",
            )[0, :, 0]

            wvelsurf = tfa.image.interpolate_bilinear(
                tf.expand_dims(tf.expand_dims(self.wvelsurf, axis=0), axis=-1),
                indices,
                indexing="ij",
            )[0, :, 0]

            wvel = wvelbase + (wvelsurf - wvelbase) * (
                1 - (1 - self.rhpos) ** 4
            )  # SIA-like

        else:
            # This is the new attemps not working yet :-(

            assert hasattr(self, W)

            w = tfa.image.interpolate_bilinear(
                tf.expand_dims(self.W, axis=-1),
                indices,
                indexing="ij",
            )[:, :, 0]

            wvel = tf.reduce_sum(wei * w, axis=0)

        self.xpos = self.xpos + self.dt * tf.reduce_sum(
            wei * u, axis=0
        )  # forward euler
        self.ypos = self.ypos + self.dt * tf.reduce_sum(
            wei * v, axis=0
        )  # forward euler
        self.zpos = self.zpos + self.dt * wvel  # forward euler

    # make sur the particle remains in the horiz. comp. domain
    self.xpos = tf.clip_by_value(self.xpos, self.x[0], self.x[-1])
    self.ypos = tf.clip_by_value(self.ypos, self.y[0], self.y[-1])

    indices = tf.concat(
        [
            tf.expand_dims(tf.cast(j, dtype="int32"), axis=-1),
            tf.expand_dims(tf.cast(i, dtype="int32"), axis=-1),
        ],
        axis=-1,
    )
    updates = tf.cast(tf.where(self.rhpos == 1, self.wpos, 0), dtype="float32")

    # this computes the sum of the weight of particles on a 2D grid
    self.weight_particles = tf.tensor_scatter_nd_add(
        tf.zeros_like(self.thk), indices, updates
    )

    # compute the englacial time
    self.englt = self.englt + tf.cast(
        tf.where(self.rhpos < 1, self.dt, 0.0), dtype="float32"
    )

    self.tcomp["particles"][-1] -= time.time()
    self.tcomp["particles"][-1] *= -1
    

def final_particles(params, self):
    pass


def _zeta_to_rhs(self, zeta):
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


def seeding_particles(params, self):
    """
    here we define (xpos,ypos) the horiz coordinate of tracked particles
    and rhpos is the relative position in the ice column (scaled bwt 0 and 1)

    here we seed only the accum. area (a bit more), where there is
    significant ice, and in some points of a regular grid self.gridseed
    (density defined by density_seeding)

    """

    #        This will serve to remove imobile particles, but it is not active yet.

    #        indices = tf.expand_dims( tf.concat(
    #                       [tf.expand_dims((self.ypos - self.y[0]) / self.dx, axis=-1),
    #                        tf.expand_dims((self.xpos - self.x[0]) / self.dx, axis=-1)],
    #                       axis=-1 ), axis=0)

    #        import tensorflow_addons as tfa

    #        thk = tfa.image.interpolate_bilinear(
    #                    tf.expand_dims(tf.expand_dims(self.thk, axis=0), axis=-1),
    #                    indices,indexing="ij",      )[0, :, 0]

    #        J = (thk>1)

    I = (
        (self.thk > 10) & (self.smb > -2) & self.gridseed
    )  # seed where thk>10, smb>-2, on a coarse grid
    self.nxpos = self.X[I]
    self.nypos = self.Y[I]
    self.nzpos = self.usurf[I]
    self.nrhpos = tf.ones_like(self.X[I])
    self.nwpos = tf.ones_like(self.X[I])
    self.ntpos = tf.ones_like(self.X[I]) * self.t
    self.nenglt = tf.zeros_like(self.X[I])
