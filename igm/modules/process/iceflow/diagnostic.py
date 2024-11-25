import numpy as np 
import tensorflow as tf 

from .utils import *
from .solve import *
from .emulate import *

import time

def initialize_iceflow_diagnostic(params,state):

    initialize_iceflow_emulator(params,state)
    
    initialize_iceflow_solver(params,state)

    state.diagno = []

    # state.UT = tf.Variable(
    #     tf.zeros((params.iflo_Nz, state.thk.shape[0], state.thk.shape[1]))
    # )
    # state.VT = tf.Variable(
    #     tf.zeros((params.iflo_Nz, state.thk.shape[0], state.thk.shape[1]))
    # )

def update_iceflow_diagnostic(params, state):

    ################ Solve

    time_solve = time.time()

    U, V, Cost_Glen = solve_iceflow(params, state, state.U, state.V)

    COST_Glen     = Cost_Glen[-1].numpy()

    time_solve -= time.time()
    time_solve *= -1
 
    state.U.assign(U)
    state.V.assign(V)
    
    update_2d_iceflow_variables(params, state)

    ################ Retrain

    time_retra = time.time()

    update_iceflow_emulator(params, state)

    fieldin = [vars(state)[f] for f in params.iflo_fieldin]
    X = fieldin_to_X(params, fieldin)
    Y = state.iceflow_model(X)
    U, V = Y_to_UV(params, Y)

    COST_Emulator = state.COST_EMULATOR[-1].numpy()

    time_retra -= time.time()
    time_retra *= -1

    ################ Analysis

    nb_it_solve = len(Cost_Glen)
    nb_it_emula = len(state.COST_EMULATOR)

    l1, l2 = computemisfit(state, state.thk, state.U - U, state.V - V)

    state.diagno.append([state.it, l1, l2, COST_Glen, COST_Emulator, nb_it_solve, nb_it_emula, time_solve, time_retra])

def computemisfit(state, thk, U, V):
    ubar = tf.reduce_sum(state.vert_weight * U, axis=0)
    vbar = tf.reduce_sum(state.vert_weight * V, axis=0)

    VEL = tf.stack([ubar, vbar], axis=0)
    MA = tf.where(thk > 1, tf.ones_like(VEL), 0)
    # MA = tf.where(state.thk > 1, tf.ones_like(VEL), 0)

    nl1diff = tf.reduce_sum(MA * tf.abs(VEL)) / tf.reduce_sum(MA)
    nl2diff = tf.reduce_sum(MA * tf.abs(VEL) ** 2) / tf.reduce_sum(MA)

    return nl1diff.numpy(), np.sqrt(nl2diff)

def finalize_iceflow_diagnostic(params, state):

    state.diagno = np.stack(state.diagno)
    np.savetxt("errors_diagno.txt", state.diagno, delimiter=",", fmt="%10.3f",
               header="it,l1,l2,COST_Glen,COST_Emulator,nb_it_solve,nb_it_emula,time_solve,time_retra")
