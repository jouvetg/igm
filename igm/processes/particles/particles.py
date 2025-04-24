#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os, sys, shutil
import matplotlib.pyplot as plt
import datetime, time
import tensorflow as tf
import igm
from netCDF4 import Dataset

from igm.processes.utils import *

import nvtx

from numba import cuda
import cupy as cp
import math

def srange(message, color):
    tf.test.experimental.sync_devices()
    return nvtx.start_range(message, color)


def erange(rng):
    tf.test.experimental.sync_devices()
    nvtx.end_range(rng)


def initialize(cfg, state):

    if cfg.processes.particles.tracking_method == "3d":
        if "vert_flow" not in cfg.processes:
            raise ValueError(
                "The 'vert_flow' module is required to use the 3d tracking method in the 'particles' module."
            )

    state.tlast_seeding = cfg.processes.particles.tlast_seeding_init

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
    rr = int(1.0 / cfg.processes.particles.density_seeding)
    state.gridseed[::rr, ::rr] = True

    if cfg.processes.particles.write_trajectories:
        initialize_write_particle(cfg, state)

def interpolate_bilinear_custom(grid_2d, sampling_points):
    """Bilinear interpolation on a 2D regular grid.

    Args:
        grid_2d: A tensor with shape `[A1, ..., An, H, W, C]` where H and W are
        height and width of the grid and C is the number of channels.
        sampling_points: A tensor with shape `[A1, ..., An, M, 2]` where M is the
        number of sampling points. Sampling points outside the grid are projected
        in the grid borders.
        name:  A name for this op that defaults to "bilinear_interpolate".

    Returns:
        A tensor of shape `[A1, ..., An, M, C]`
    """
    grid_shape = tf.shape(grid_2d)[-3:-1]
    batch_dims = tf.shape(sampling_points)[:-2]
    num_points = tf.shape(sampling_points)[-2]

    bottom_left = tf.floor(sampling_points)
    top_right = bottom_left + 1
    bottom_left_index = tf.cast(bottom_left, tf.int32)
    top_right_index = tf.cast(top_right, tf.int32)
    x0_index, y0_index = tf.unstack(bottom_left_index, axis=-1)
    x1_index, y1_index = tf.unstack(top_right_index, axis=-1)
    index_x = tf.concat([x0_index, x1_index, x0_index, x1_index], axis=-1)
    index_y = tf.concat([y0_index, y0_index, y1_index, y1_index], axis=-1)
    indices = tf.stack([index_x, index_y], axis=-1)
    clip_value = tf.convert_to_tensor([grid_shape - 1], dtype=indices.dtype)
    indices = tf.clip_by_value(indices, 0, clip_value)
    content = tf.gather_nd(grid_2d, indices, batch_dims=tf.size(batch_dims))
    distance_to_bottom_left = sampling_points - bottom_left
    distance_to_top_right = top_right - sampling_points
    x_x0, y_y0 = tf.unstack(distance_to_bottom_left, axis=-1)
    x1_x, y1_y = tf.unstack(distance_to_top_right, axis=-1)
    weights_x = tf.concat([x1_x, x_x0, x1_x, x_x0], axis=-1)
    weights_y = tf.concat([y1_y, y1_y, y_y0, y_y0], axis=-1)
    weights = tf.expand_dims(weights_x * weights_y, axis=-1)

    interpolated_values = weights * content

    return tf.add_n(tf.split(interpolated_values, [num_points] * 4, -2))


@cuda.jit # device function vs ufunc?
def interpolate_2d(interpolated_grid, grid_values, array_particles, depth):
    particle_id = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    
    if particle_id < array_particles.shape[0]:
        
        for depth_layer in range(depth):

            y_pos = array_particles[particle_id, 0]
            x_pos = array_particles[particle_id, 1]
            
            x_1 = int(x_pos) # left x coordinate
            y_1 = int(y_pos) # bottom y coordinate
            x_2 = x_1 + 1 # right x coordinate
            y_2 = y_1 + 1 # top y coordinate
            
            # print("depth_layer", depth_layer)
            # print("y_1", y_1)
            # print("x_1", x_1)
            
            Q_11 = grid_values[depth_layer, y_1, x_1] # bottom left corner
            Q_12 = grid_values[depth_layer, y_2, x_1] # top left corner
            Q_21 = grid_values[depth_layer, y_1, x_2] # bottom right corner
            Q_22 = grid_values[depth_layer, y_2, x_2] # top right corner
            
            # print("Q_11", Q_11)
            # print("Q_12", Q_12)
            # print("Q_21", Q_21)
            # print("Q_22", Q_22)
            
            # Interpolating on x
            dx = (x_2 - x_1)
            x_left_weight = (x_pos - x_1) / dx
            x_right_weight = (x_2 - x_pos) / dx
            R_1 = x_left_weight * Q_21 + x_right_weight * Q_11 # bottom x interpolation for fixed y_1 (f(x, y_1))
            R_2 = x_left_weight * Q_22 + x_right_weight * Q_12 # top x interpolation for fixed y_2 (f(x, y_2))
                        
            # Interpolating on y
            dy = (y_2 - y_1)
            y_bottom_weight = (y_pos - y_1) / dy
            y_top_weight = (y_2 - y_pos) / dy
            
            P = y_bottom_weight * R_2 + y_top_weight * R_1 # final interpolation for fixed x (f(x, y))
            interpolated_grid[depth_layer, particle_id] = P

# @cuda.jit # device function vs ufunc?
# def interpolate_2d(interpolated_grid, grid_values, array_particles, depth):
#     particle_id = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    
#     if particle_id < array_particles.shape[0]:
        
#         for depth_layer in range(depth):
#             x_pos = array_particles[particle_id, 0]
#             y_pos = array_particles[particle_id, 1]
            
#             x_1 = int(x_pos) # left x coordinate
#             y_1 = int(y_pos) # bottom y coordinate
#             x_2 = x_1 + 1 # right x coordinate
#             y_2 = y_1 + 1 # top y coordinate
            
            
#             Q_11 = grid_values[depth_layer, y_1, x_1] # bottom left corner
#             Q_12 = grid_values[depth_layer, y_2, x_1] # top left corner
#             Q_21 = grid_values[depth_layer, y_1, x_2] # bottom right corner
#             Q_22 = grid_values[depth_layer, y_2, x_2] # top right corner
            
#             # Interpolating on x
#             dx = (x_2 - x_1)
#             x_left_weight = (x_pos - x_1) / dx
#             x_right_weight = (x_2 - x_pos) / dx
#             R_1 = x_left_weight * Q_11 + x_right_weight * Q_21 # bottom x interpolation for fixed y_1 (f(x, y_1))
#             R_2 = x_left_weight * Q_12 + x_right_weight * Q_22 # top x interpolation for fixed y_2 (f(x, y_2))
            
#             # Interpolating on y
#             dy = (y_2 - y_1)
#             y_bottom_weight = (y_pos - y_1) / dy
#             y_top_weight = (y_2 - y_pos) / dy
            
#             P = y_bottom_weight * R_1 + y_top_weight * R_2 # final interpolation for fixed x (f(x, y))
#             print("P", P)
#             interpolated_grid[depth_layer, particle_id] = P

def interpolate_particles_2d(U, V, thk, topg, smb, indices):
    # print("tracing interpolate_particles_2d")
    
    depth = U.shape[0]
    number_of_particles = indices.shape[1]
    
    # rng = srange(message="interpolating_u (numba)", color="black")
    print("UUUUU", U)
    
    U_numba = tf.experimental.dlpack.to_dlpack(U)
    U_numba = cp.from_dlpack(U_numba)
    # U = cuda.as_cuda_array(U_numba)
    
    indices_numba = tf.squeeze(indices) # (N, 2) instead of (1, N, 2)
    particles = tf.experimental.dlpack.to_dlpack(indices_numba)
    array_particles = cp.from_dlpack(particles)
    # array_particles = cuda.as_cuda_array(array_particles)
        
    threadsperblock = 32
    blockspergrid = math.ceil(number_of_particles / threadsperblock)
    
    u_device = cuda.device_array(shape=(depth, number_of_particles), dtype="float32")
    # print(array_particles.shape)
    # print('number of particles', number_of_particles)
    # exit()
    # print(array_particles)
    interpolate_2d[blockspergrid, threadsperblock](u_device, U_numba, array_particles, depth)
    print(U_numba.shape)
    # u = cp.array(u_device)
    u = cp.asarray(u_device)
    u = tf.experimental.dlpack.from_dlpack(u.toDlpack())
    # erange(rng)
    
    # rng = srange(message="interpolating_u (tf)", color="white")
    u_tf = interpolate_bilinear_tf(
        tf.expand_dims(U, axis=-1),
        indices,
        indexing="ij",
    )
    
    # u_custom = interpolate_bilinear_custom(
    #     tf.expand_dims(U, axis=-1),
    #     indices,
    # )
    u = u_tf[:, :, 0]
    # u_tf = u_tf[:, :, 0]
    
    # erange(rng)

    # rng = srange(message="slicing_u", color="pink")
    # print("U shape", U.shape, u.shape, u_tf.shape)
    # erange(rng)
    # print("UUUUU_numba", U_numba)
    # print(tf.reduce_max(u))
    # print(tf.reduce_max(u_device))
    # print(tf.reduce_mean(abs(u - u_tf)))
    # print(tf.reduce_max(abs(u - u_tf)))
    # plt.figure()
    # plt.imshow(u)
    # plt.imshow(u_tf)
    # plt.show()
    # exit()
    # print(tf.reduce_mean(abs(u - u_tf)))
    # print(tf.reduce_max(abs(u - u_tf)))
    
    rng = srange(message="interpolating_v", color="white")
    v = interpolate_bilinear_tf(
        tf.expand_dims(V, axis=-1),
        indices,
        indexing="ij",
    )
    erange(rng)

    # print("V shape", V.shape, v.shape)
    rng = srange(message="slicing_v", color="green")
    v = v[:, :, 0]
    erange(rng)

    rng = srange(message="interpolating_thk", color="white")
    thk = interpolate_bilinear_tf(
        tf.expand_dims(tf.expand_dims(thk, axis=0), axis=-1),
        indices,
        indexing="ij",
    )
    erange(rng)

    # print("thk shape", thk.shape)
    rng = srange(message="slicing_thk", color="purple")
    thk = thk[0, :, 0]
    erange(rng)

    rng = srange(message="interpolating_topg", color="white")
    topg = interpolate_bilinear_tf(
        tf.expand_dims(tf.expand_dims(topg, axis=0), axis=-1),
        indices,
        indexing="ij",
    )
    erange(rng)

    # print("topg shape", topg.shape)
    rng = srange(message="slicing_topg", color="red")
    topg = topg[0, :, 0]
    erange(rng)

    rng = srange(message="interpolating_smb", color="white")
    smb = interpolate_bilinear_tf(
        tf.expand_dims(tf.expand_dims(smb, axis=0), axis=-1),
        indices,
        indexing="ij",
    )
    erange(rng)

    # print("smb shape", smb.shape)
    rng = srange(message="slicing_smb", color="blue")
    smb = smb[0, :, 0]
    erange(rng)

    return u, v, thk, topg, smb


# @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),
#                               tf.TensorSpec(shape=None, dtype=tf.float32),
#                               tf.TensorSpec(shape=[], dtype=tf.float32),
#                                 tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32)])
def get_weights(vertical_spacing, number_z_layers, particle_r, u):
    "What is this function doing? Name it properly.."
    # print("tracing weights")
    zeta = _rhs_to_zeta(vertical_spacing, particle_r)  # get the position in the column
    I0 = tf.cast(
        tf.math.floor(zeta * (number_z_layers - 1)),
        dtype="int32",
    )
    I0 = tf.minimum(I0, number_z_layers - 2)  # make sure to not reach the upper-most pt
    I1 = I0 + 1
    zeta0 = tf.cast(I0 / (number_z_layers - 1), dtype="float32")
    zeta1 = tf.cast(I1 / (number_z_layers - 1), dtype="float32")

    lamb = (zeta - zeta0) / (zeta1 - zeta0)

    ind0 = tf.transpose(tf.stack([I0, tf.range(I0.shape[0])]))
    ind1 = tf.transpose(tf.stack([I1, tf.range(I1.shape[0])]))

    weights = tf.zeros_like(u)
    weights = tf.tensor_scatter_nd_add(weights, indices=ind0, updates=1 - lamb)
    weights = tf.tensor_scatter_nd_add(weights, indices=ind1, updates=lamb)

    return weights


def update(cfg, state):

    rng_outer = srange(message="particle_update", color="white")
    if "iceflow" not in cfg.processes:
        raise ValueError("The 'iceflow' module is required to use the particles module")

    if hasattr(state, "logger"):
        state.logger.info("Update particle tracking at time : " + str(state.t.numpy()))

    if (
        state.t.numpy() - state.tlast_seeding
    ) >= cfg.processes.particles.frequency_seeding:

        rng = srange(message="seeding_particles", color="black")
        # seed new particles
        (
            nparticle_x,
            nparticle_y,
            nparticle_z,
            nparticle_r,
            nparticle_w,
            nparticle_t,
            nparticle_englt,
            nparticle_topg,
            nparticle_thk,
        ) = seeding_particles(cfg, state)
        erange(rng)

        rng = srange(message="concat_new_particles", color="blue")
        # merge the new seeding points with the former ones
        particle_x = tf.Variable(
            tf.concat([state.particle_x, nparticle_x], axis=-1), trainable=False
        )
        particle_y = tf.Variable(
            tf.concat([state.particle_y, nparticle_y], axis=-1), trainable=False
        )
        particle_z = tf.Variable(
            tf.concat([state.particle_z, nparticle_z], axis=-1), trainable=False
        )
        state.particle_r = tf.Variable(
            tf.concat([state.particle_r, nparticle_r], axis=-1), trainable=False
        )
        state.particle_w = tf.Variable(
            tf.concat([state.particle_w, nparticle_w], axis=-1), trainable=False
        )
        state.particle_t = tf.Variable(
            tf.concat([state.particle_t, nparticle_t], axis=-1), trainable=False
        )
        particle_englt = tf.Variable(
            tf.concat([state.particle_englt, nparticle_englt], axis=-1),
            trainable=False,
        )
        # state.particle_topg = tf.Variable(
        #     tf.concat([state.particle_topg, nparticle_topg], axis=-1),
        #     trainable=False,
        # )
        # state.particle_thk = tf.Variable(
        #     tf.concat([state.particle_thk, nparticle_thk], axis=-1),
        #     trainable=False,
        # )
        erange(rng)

        state.tlast_seeding = state.t.numpy()
    else:
        # use the old particles
        particle_x = state.particle_x
        particle_y = state.particle_y
        particle_z = state.particle_z
        particle_englt = state.particle_englt

    
    
    if (particle_x.shape[0] > 0) & (state.it >= 0):

        # find the indices of trajectories
        # these indicies are real values to permit 2D interpolations (particles are not necessary on points of the grid)
        i = (particle_x) / state.dx
        j = (particle_y) / state.dx

        indices = tf.expand_dims(
            tf.concat(
                [tf.expand_dims(j, axis=-1), tf.expand_dims(i, axis=-1)], axis=-1
            ),
            axis=0,
        )
        # print(f"Number of particles: {indices.shape}")

        U_input = state.U
        V_input = state.V
        thk_input = state.thk
        topg_input = state.topg
        smb_input = state.smb
        # rng_reading = srange(message="reading_from_state", color="blue")
        # erange(rng_reading)
        
        
        rng = srange(message="interpolate_bilinear_section", color="red")
        u, v, thk, topg, smb = (
            interpolate_particles_2d(  # only need smb for the simple tracking
                U_input,
                V_input,
                thk_input,
                topg_input,
                smb_input,
                indices,
            )
        )
        erange(rng)
        state.particle_thk = thk
        state.particle_topg = topg

        rng = srange(message="misc_compuations", color="green")
        vertical_spacing = cfg.processes.iceflow.numerics.vert_spacing
        number_z_layers = cfg.processes.iceflow.numerics.Nz
        weights = get_weights(
            vertical_spacing=vertical_spacing,
            number_z_layers=number_z_layers,
            particle_r=state.particle_r,
            u=u,
        )

        erange(rng)

        # uses the vertical velocity w computed in the vert_flow module

        rng = srange(message="3d interpolate bilinear", color="black")
        w = interpolate_bilinear_tf(
            tf.expand_dims(state.W, axis=-1),
            indices,
            indexing="ij",
        )[:, :, 0]
        erange(rng)

        rng = srange(message="compute_new_particle_locations", color="red")
        particle_x = particle_x + state.dt * tf.reduce_sum(weights * u, axis=0)
        particle_y = particle_y + state.dt * tf.reduce_sum(weights * v, axis=0)
        particle_z = particle_z + state.dt * tf.reduce_sum(weights * w, axis=0)

        # make sure the particle vertically remain within the ice body
        state.particle_z = tf.clip_by_value(particle_z, topg, topg + thk)
        # relative height of the particle within the glacier
        particle_r = (state.particle_z - topg) / thk
        # if thk=0, state.rhpos takes value nan, so we set rhpos value to one in this case :
        state.particle_r = tf.where(thk == 0, tf.ones_like(particle_r), particle_r)
        erange(rng)

        rng = srange(message="final_computations", color="yellow")
        # make sur the particle remains in the horiz. comp. domain
        state.particle_x = tf.clip_by_value(particle_x, 0, state.x[-1] - state.x[0])
        state.particle_y = tf.clip_by_value(particle_y, 0, state.y[-1] - state.y[0])

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
        state.particle_englt = particle_englt + tf.cast(
            tf.where(state.particle_r < 1, state.dt, 0.0), dtype="float32"
        )

        #    if int(state.t)%10==0:
        #        print("nb of part : ",state.xpos.shape)
        erange(rng)

    if cfg.processes.particles.write_trajectories:
        rng = srange(message="update_write_particle", color="yellow")
        update_write_particle(cfg, state)
        erange(rng)

    erange(rng_outer)


def finalize(cfg, state):
    pass


def _zeta_to_rhs(cfg, zeta):
    return (zeta / cfg.processes.iceflow.iceflow.vert_spacing) * (
        1.0 + (cfg.processes.iceflow.iceflow.vert_spacing - 1.0) * zeta
    )


def _rhs_to_zeta(verticle_spacing, rhs):
    if verticle_spacing == 1:
        rhs = zeta
    else:
        DET = tf.sqrt(1 + 4 * (verticle_spacing - 1) * verticle_spacing * rhs)
        zeta = (DET - 1) / (2 * (verticle_spacing - 1))

    #           temp = cfg.processes.iceflow.iceflow.Nz*(DET-1)/(2*(cfg.processes.iceflow.iceflow.vert_spacing-1))
    #           I=tf.cast(tf.minimum(temp-1,cfg.processes.iceflow.iceflow.Nz-1),dtype='int32')

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
    )  # here you may redefine how you want to seed particles
    nparticle_x = state.X[I] - state.x[0]  # x position of the particle
    nparticle_y = state.Y[I] - state.y[0]  # y position of the particle
    nparticle_z = state.usurf[I]  # z position of the particle
    nparticle_r = tf.ones_like(state.X[I])  # relative position in the ice column
    nparticle_w = tf.ones_like(state.X[I])  # weight of the particle
    nparticle_t = (
        tf.ones_like(state.X[I]) * state.t
    )  # "date of birth" of the particle (useful to compute its age)
    nparticle_englt = tf.zeros_like(
        state.X[I]
    )  # time spent by the particle burried in the glacier
    nparticle_thk = state.thk[I]  # ice thickness at position of the particle
    nparticle_topg = state.topg[I]  # z position of the bedrock under the particle

    return (
        nparticle_x,
        nparticle_y,
        nparticle_z,
        nparticle_r,
        nparticle_w,
        nparticle_t,
        nparticle_englt,
        nparticle_topg,
        nparticle_thk,
    )


# def seeding_particles(cfg, state):
#     """
#     here we define (xpos,ypos) the horiz coordinate of tracked particles
#     and rhpos is the relative position in the ice column (scaled bwt 0 and 1)

#     here we seed only the accum. area (a bit more), where there is
#     significant ice, and in some points of a regular grid state.gridseed
#     (density defined by density_seeding)

#     """

#     # ! THK and SMB modules are required. Insert in the init function of the particles module (actually, don't because the modules can be
#     # ! initialized in any order, and the particles module is not guaranteed to be initialized after the thk and smb modules)
#     # ! Instead, insert it HERE when needed (although it might call it multiple times and be less efficient...)

#     if not hasattr(state, "thk"):
#         raise ValueError("The thk module is required to use the particles module")
#     if not hasattr(state, "smb"):
#         raise ValueError(
#             "A smb module is required to use the particles module. Please use the built-in smb module or create a custom one that overwrites the 'state.smb' value."
#         )

#     #        This will serve to remove imobile particles, but it is not active yet.

#     #        indices = tf.expand_dims( tf.concat(
#     #                       [tf.expand_dims((state.ypos - state.y[0]) / state.dx, axis=-1),
#     #                        tf.expand_dims((state.xpos - state.x[0]) / state.dx, axis=-1)],
#     #                       axis=-1 ), axis=0)

#     #        thk = interpolate_bilinear_tf(
#     #                    tf.expand_dims(tf.expand_dims(state.thk, axis=0), axis=-1),
#     #                    indices,indexing="ij",      )[0, :, 0]

#     #        J = (thk>1)

#     # here we seed where i) thickness is higher than 1 m
#     #                    ii) the seeding field of geology.nc is active
#     #                    iii) on the gridseed (which permit to control the seeding density)
#     #                    iv) on the accumulation area
#     I = (
#         (state.thk > 1) & state.gridseed & (state.smb > 0)
#     )  # here you may redefine how you want to seed particles
#     state.nparticle_x = state.X[I] - state.x[0]  # x position of the particle
#     state.nparticle_y = state.Y[I] - state.y[0]  # y position of the particle
#     state.nparticle_z = state.usurf[I]  # z position of the particle
#     state.nparticle_r = tf.ones_like(state.X[I])  # relative position in the ice column
#     state.nparticle_w = tf.ones_like(state.X[I])  # weight of the particle
#     state.nparticle_t = (
#         tf.ones_like(state.X[I]) * state.t
#     )  # "date of birth" of the particle (useful to compute its age)
#     state.nparticle_englt = tf.zeros_like(
#         state.X[I]
#     )  # time spent by the particle burried in the glacier
#     state.nparticle_thk = state.thk[I]  # ice thickness at position of the particle
#     state.nparticle_topg = state.topg[I]  # z position of the bedrock under the particle


def initialize_write_particle(cfg, state):

    directory = "trajectories"
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)

    if cfg.processes.particles.add_topography:
        ftt = os.path.join("trajectories", "topg.csv")
        array = tf.transpose(
            tf.stack(
                [state.X[state.X > 0], state.Y[state.X > 0], state.topg[state.X > 0]]
            )
        )
        np.savetxt(ftt, array, delimiter=",", fmt="%.2f", header="x,y,z")


def update_write_particle(cfg, state):
    if state.saveresult:

        f = os.path.join(
            "trajectories",
            "traj-" + "{:06d}".format(int(state.t.numpy())) + ".csv",
        )

        ID = tf.cast(tf.range(state.particle_x.shape[0]), dtype="float32")
        array = tf.transpose(
            tf.stack(
                [
                    ID,
                    state.particle_x.numpy().astype(np.float64)
                    + state.x[0].numpy().astype(np.float64),
                    state.particle_y.numpy().astype(np.float64)
                    + state.y[0].numpy().astype(np.float64),
                    state.particle_z,
                    state.particle_r,
                    state.particle_t,
                    state.particle_englt,
                    state.particle_topg,
                    state.particle_thk,
                ],
                axis=0,
            )
        )
        np.savetxt(
            f, array, delimiter=",", fmt="%.2f", header="Id,x,y,z,rh,t,englt,topg,thk"
        )

        ft = os.path.join("trajectories", "time.dat")
        with open(ft, "a") as f:
            print(state.t.numpy(), file=f)

        if cfg.processes.particles.add_topography:
            ftt = os.path.join(
                "trajectories",
                "usurf-" + "{:06d}".format(int(state.t.numpy())) + ".csv",
            )
            array = tf.transpose(
                tf.stack(
                    [
                        state.X[state.X > 1],
                        state.Y[state.X > 1],
                        state.usurf[state.X > 1],
                    ]
                )
            )
            np.savetxt(ftt, array, delimiter=",", fmt="%.2f", header="x,y,z")
