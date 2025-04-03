#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np 
import tensorflow as tf 

from ..utils import *
from ..solve.solve import *
from ..emulate.emulate import *

import time

def initialize_iceflow_diagnostic(cfg,state):

    initialize_iceflow_emulator(cfg,state)
    
    initialize_iceflow_solver(cfg,state)

    state.diagno = []

    state.velsurf_mag_exa = tf.zeros_like(state.thk)
    state.velsurf_mag_app = tf.zeros_like(state.thk)

    # state.UT = tf.Variable(
    #     tf.zeros((cfg.processes.iceflow.numerics.Nz, state.thk.shape[0], state.thk.shape[1]))
    # )
    # state.VT = tf.Variable(
    #     tf.zeros((cfg.processes.iceflow.numerics.Nz, state.thk.shape[0], state.thk.shape[1]))
    # )

    print("it,l1,l2,COST_Glen,COST_Emulator,nb_it_solve,nb_it_emula,time_solve,time_retra")

def update_iceflow_diagnostic(cfg, state):

    ################ Solve

    time_solve = time.time()

    state.U, state.V, Cost_Glen = solve_iceflow(cfg, state, state.U, state.V)

    COST_Glen     = Cost_Glen[-1].numpy()

    time_solve -= time.time()
    time_solve *= -1
  
    update_2d_iceflow_variables(cfg, state)

    ################ Retrain

    time_retra = time.time()

    update_iceflow_emulator(cfg, state, state.it)

    fieldin = [vars(state)[f] for f in cfg.processes.iceflow.emulator.fieldin]
    X = fieldin_to_X(cfg, fieldin)
    Y = state.iceflow_model(X)
    U, V = Y_to_UV(cfg, Y)

    if len(state.COST_EMULATOR) > 0:
        COST_Emulator = state.COST_EMULATOR[-1].numpy()
    else:
        COST_Emulator = np.NaN

#    if len(state.GRAD_NORM) > 0:
#        GRAD_Norm     = state.GRAD_NORM[-1].numpy()
#    else:
    GRAD_Norm = np.NaN

    time_retra -= time.time()
    time_retra *= -1

    ################ Analysis

    nb_it_solve = len(Cost_Glen)
    nb_it_emula = len(state.COST_EMULATOR)

    l1, l2 = computemisfit(state, state.thk, state.U - U, state.V - V)

    vol = np.sum(state.thk) * (state.dx**2) / 10**9

    state.diagno.append([state.t,state.it, l1, l2, COST_Glen, COST_Emulator, nb_it_solve, nb_it_emula, time_solve, time_retra, GRAD_Norm, vol])

    state.velsurf_mag_app = tf.linalg.norm(tf.stack([U[0,-1], V[0,-1]], axis=0), axis=0)
    state.velsurf_mag_exa = tf.linalg.norm(tf.stack([state.U[-1], state.V[-1]], axis=0), axis=0)

    if state.it % 100 == 0: 
        np.savetxt("errors_diagno.txt", np.stack(state.diagno), delimiter=",", fmt="%10.3f",
                header="time, it,l1,l2,COST_Glen,COST_Emulator,nb_it_solve,nb_it_emula,time_solve,time_retra, GRAD_Norm, vol",
                comments='')


def computemisfit(state, thk, U, V):
    ubar = tf.reduce_sum(state.vert_weight * U, axis=0)
    vbar = tf.reduce_sum(state.vert_weight * V, axis=0)

    VEL = tf.stack([ubar, vbar], axis=0)
    MA = tf.where(thk > 1, tf.ones_like(VEL), 0)
    # MA = tf.where(state.thk > 1, tf.ones_like(VEL), 0)

    nl1diff = tf.reduce_sum(MA * tf.abs(VEL)) / tf.reduce_sum(MA)
    nl2diff = tf.reduce_sum(MA * tf.abs(VEL) ** 2) / tf.reduce_sum(MA)

    return nl1diff.numpy(), np.sqrt(nl2diff)

def finalize_iceflow_diagnostic(cfg, state):
 
    pass
