import numpy as np 
import tensorflow as tf 

from .utils import *
from .solve import *
from .emulate import *

def initialize_iceflow_diagnostic(params,state):

    initialize_iceflow_emulator(params,state)
    
    initialize_iceflow_solver(params,state)

    state.UT = tf.Variable(
        tf.zeros((params.iflo_Nz, state.thk.shape[0], state.thk.shape[1]))
    )
    state.VT = tf.Variable(
        tf.zeros((params.iflo_Nz, state.thk.shape[0], state.thk.shape[1]))
    )

def update_iceflow_diagnostic(params, state):
    
    if params.iflo_retrain_emulator_freq > 0:
        update_iceflow_emulator(params, state)
        COST_Emulator = state.COST_EMULATOR[-1].numpy()
    else:
        COST_Emulator = 0.0

    update_iceflow_emulated(params, state)

    if state.it % 10 == 0:
        UT, VT, Cost_Glen = solve_iceflow(params, state, state.UT, state.VT)
        state.UT.assign(UT)
        state.VT.assign(VT)
        COST_Glen = Cost_Glen[-1].numpy()

        print("nb solve iterations :", len(Cost_Glen))

        l1, l2 = computemisfit(state, state.thk, state.U - state.UT, state.V - state.VT)

        ERR = [state.t.numpy(), COST_Glen, COST_Emulator, l1, l2]

        print(ERR)

        with open("errors.txt", "ab") as f:
            np.savetxt(f, np.expand_dims(ERR, axis=0), delimiter=",", fmt="%5.5f")


def computemisfit(state, thk, U, V):
    ubar = tf.reduce_sum(state.vert_weight * U, axis=0)
    vbar = tf.reduce_sum(state.vert_weight * V, axis=0)

    VEL = tf.stack([ubar, vbar], axis=0)
    MA = tf.where(thk > 1, tf.ones_like(VEL), 0)
    # MA = tf.where(state.thk > 1, tf.ones_like(VEL), 0)

    nl1diff = tf.reduce_sum(MA * tf.abs(VEL)) / tf.reduce_sum(MA)
    nl2diff = tf.reduce_sum(MA * tf.abs(VEL) ** 2) / tf.reduce_sum(MA)

    return nl1diff.numpy(), np.sqrt(nl2diff)
