#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time

from igm.modules.utils import *

def params(parser):
    pass
 
def initialize(cfg, state):
    Ny, Nx = state.thk.shape
 
    # initialize age of ice to zero
    
    if "iceflow" not in cfg.modules:
        raise ValueError("The 'iceflow' module is required for the 'age_of_ice' module.")
    
    state.age_of_ice = tf.Variable(tf.zeros((cfg.modules.iceflow.iceflow.iflo_Nz, Ny, Nx)))
    
    state.tcomp_age_of_ice = []
  
def update(cfg, state):
    if hasattr(state, "logger"):
        state.logger.info("Update AGE OF ICE at time : " + str(state.t.numpy()))

    state.tcomp_age_of_ice.append(time.time())
    
    # get the vertical discretization
    dz = vertically_disc_tf( state.thk, cfg.modules.iceflow.iceflow.iflo_Nz, cfg.modules.iceflow.iceflow.iflo_vert_spacing)
    
    # Assign zero age for the top surface elevation ice located in the accumulation area
    state.age_of_ice[-1].assign( tf.where(state.smb > 0, 0.0, state.age_of_ice[-1]) )

    # one explicit step for the advection
    state.age_of_ice.assign( state.age_of_ice - state.dt * compute_upwind_3d_tf(
        state.U, state.V, state.W, state.age_of_ice, state.dx, dz, cfg.modules.iceflow.iceflow.iflo_thr_ice_thk
    ) )
    
    # add the time step to the age
    state.age_of_ice.assign( state.age_of_ice + state.dt )
  
    state.tcomp_age_of_ice[-1] -= time.time()
    state.tcomp_age_of_ice[-1] *= -1

def finalize(cfg, state):
    pass

#######################################################################################

@tf.function()
def vertically_disc_tf(thk, Nz, vert_spacing):
    zeta = tf.cast(tf.range(Nz) / (Nz - 1), "float32")
    levels = (zeta / vert_spacing) * (1.0 + (vert_spacing - 1.0) * zeta)
    ddz = levels[1:] - levels[:-1]

    dz = tf.expand_dims(thk, 0) * tf.expand_dims(tf.expand_dims(ddz, -1), -1)
    
    return dz


@tf.function()
def compute_upwind_3d_tf(U, V, W, E, dx, dz, thr):
    #  upwind computation of u dE/dx + v dE/dy, unit are [E s^{-1}]

    # Extend E with constant value at the domain boundaries
    Ex = tf.pad(E, [[0, 0], [0, 0], [1, 1]], "SYMMETRIC")  # has shape (nz,ny,nx+2)
    Ey = tf.pad(E, [[0, 0], [1, 1], [0, 0]], "SYMMETRIC")  # has shape (nz,ny+2,nx)
    Ez = tf.pad(E, [[1, 1], [0, 0], [0, 0]], "SYMMETRIC")  # has shape (nz+2,ny,nx)
    
    DZ2 = tf.concat([dz[0:1], dz[:-1] + dz[1:], dz[-1:]], axis=0)

    ## Compute the product selcting the upwind quantities  :-2, 1:-1 , 2:
    Rx = U * tf.where(
        U > 0,
        (Ex[:, :, 1:-1] - Ex[:, :, :-2]) / dx,
        (Ex[:, :, 2:] - Ex[:, :, 1:-1]) / dx,
    )  # has shape (nz,ny,nx)
    Ry = V * tf.where(
        V > 0,
        (Ey[:, 1:-1:, :] - Ey[:, :-2, :]) / dx,
        (Ey[:, 2:, :] - Ey[:, 1:-1, :]) / dx,
    )  # has shape (nz,ny,nx)
    Rz = W * tf.where(
        W > 0,
        (Ez[1:-1:, :, :] - Ez[:-2, :, :]) / tf.maximum(DZ2, thr),
        (Ez[2:, :, :] - Ez[1:-1, :, :]) / tf.maximum(DZ2, thr),
    )  # has shape (nz,ny,nx)

    ##  Final shape is (nz,ny,nx)
    return Rx + Ry + Rz

 