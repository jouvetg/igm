import numpy as np 
import tensorflow as tf 
import math

def initialize_iceflow_fields(params,state):

    # here we initialize variable parmaetrizing ice flow
    if not hasattr(state, "arrhenius"):
        if params.iflo_dim_arrhenius == 3:
            state.arrhenius = tf.Variable(
                tf.ones((params.iflo_Nz, state.thk.shape[0], state.thk.shape[1]))
                * params.iflo_init_arrhenius * params.iflo_enhancement_factor, trainable=False
            )
        else:
            state.arrhenius = tf.Variable(
                tf.ones_like(state.thk) * params.iflo_init_arrhenius * params.iflo_enhancement_factor, trainable=False
            )

    if not hasattr(state, "slidingco"):
        state.slidingco = tf.Variable(
            tf.ones_like(state.thk) * params.iflo_init_slidingco, trainable=False
        )

    # here we create a new velocity field
    if not hasattr(state, "U"):
        state.U = tf.Variable(
            tf.zeros((params.iflo_Nz, state.thk.shape[0], state.thk.shape[1])), trainable=False
        )
        state.V = tf.Variable(
            tf.zeros((params.iflo_Nz, state.thk.shape[0], state.thk.shape[1])), trainable=False
        )

def define_vertical_weight(params, state):
    """
    define_vertical_weight
    """

    zeta = np.arange(params.iflo_Nz + 1) / params.iflo_Nz
    weight = (zeta / params.iflo_vert_spacing) * (
        1.0 + (params.iflo_vert_spacing - 1.0) * zeta
    )
    weight = tf.Variable(weight[1:] - weight[:-1], dtype=tf.float32, trainable=False)
    state.vert_weight = tf.expand_dims(tf.expand_dims(weight, axis=-1), axis=-1)


def update_2d_iceflow_variables(params, state):
    state.uvelbase = state.U[0, :, :]
    state.vvelbase = state.V[0, :, :]
    state.ubar = tf.reduce_sum(state.U * state.vert_weight, axis=0)
    state.vbar = tf.reduce_sum(state.V * state.vert_weight, axis=0)
    state.uvelsurf = state.U[-1, :, :]
    state.vvelsurf = state.V[-1, :, :]

def compute_PAD(params,Nx,Ny):

    # In case of a U-net, must make sure the I/O size is multiple of 2**N
    if params.iflo_multiple_window_size > 0:
        NNy = params.iflo_multiple_window_size * math.ceil(
            Ny / params.iflo_multiple_window_size
        )
        NNx = params.iflo_multiple_window_size * math.ceil(
            Nx / params.iflo_multiple_window_size
        )
        return [[0, 0], [0, NNy - Ny], [0, NNx - Nx], [0, 0]]
    else:
        return [[0, 0], [0, 0], [0, 0], [0, 0]]
    
