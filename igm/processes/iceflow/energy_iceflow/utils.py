import numpy as np 
import tensorflow as tf 
 
@tf.function()
def compute_gradient_stag(s, dX, dY):
    """
    compute spatial gradient, outcome on stagerred grid
    """

    E = 2.0 * (s[:, :, 1:] - s[:, :, :-1]) / (dX[:, :, 1:] + dX[:, :, :-1])
    diffx = 0.5 * (E[:, 1:, :] + E[:, :-1, :])

    EE = 2.0 * (s[:, 1:, :] - s[:, :-1, :]) / (dY[:, 1:, :] + dY[:, :-1, :])
    diffy = 0.5 * (EE[:, :, 1:] + EE[:, :, :-1])

    return diffx, diffy
 

def stag2(B):
    return (B[:, 1:] + B[:, :-1]) / 2


def stag4(B):
    return (B[:, 1:, 1:] + B[:, 1:, :-1] + B[:, :-1, 1:] + B[:, :-1, :-1]) / 4


def stag4b(B):
    return (
        B[:, :, 1:, 1:] + B[:, :, 1:, :-1] + B[:, :, :-1, 1:] + B[:, :, :-1, :-1]
    ) / 4


def stag8(B):
    return (
        B[:, 1:, 1:, 1:]
        + B[:, 1:, 1:, :-1]
        + B[:, 1:, :-1, 1:]
        + B[:, 1:, :-1, :-1]
        + B[:, :-1, 1:, 1:]
        + B[:, :-1, 1:, :-1]
        + B[:, :-1, :-1, 1:]
        + B[:, :-1, :-1, :-1]
    ) / 8
 
def Y_to_UV(cfg, Y):
    N = cfg.processes.iceflow.iceflow.Nz

    U = tf.experimental.numpy.moveaxis(Y[:, :, :, :N], [-1], [1])
    V = tf.experimental.numpy.moveaxis(Y[:, :, :, N:], [-1], [1])

    return U, V

def UV_to_Y(cfg, U, V):
    UU = tf.experimental.numpy.moveaxis(U, [0], [-1])
    VV = tf.experimental.numpy.moveaxis(V, [0], [-1])
    RR = tf.expand_dims(
        tf.concat(
            [UU, VV],
            axis=-1,
        ),
        axis=0,
    )

    return RR

def fieldin_to_X(cfg, fieldin):
    X = []

    fieldin_dim = [0, 0, 1 * (cfg.processes.iceflow.iceflow.dim_arrhenius == 3), 0, 0]

    for f, s in zip(fieldin, fieldin_dim):
        if s == 0:
            X.append(tf.expand_dims(f, axis=-1))
        else:
            X.append(tf.experimental.numpy.moveaxis(f, [0], [-1]))

    return tf.expand_dims(tf.concat(X, axis=-1), axis=0)


def X_to_fieldin(cfg, X):
    i = 0

    fieldin_dim = [0, 0, 1 * (cfg.processes.iceflow.iceflow.dim_arrhenius == 3), 0, 0]

    fieldin = []

    for f, s in zip(cfg.processes.iceflow.iceflow.fieldin, fieldin_dim):
        if s == 0:
            fieldin.append(X[:, :, :, i])
            i += 1
        else:
            fieldin.append(
                tf.experimental.numpy.moveaxis(
                    X[:, :, :, i : i + cfg.processes.iceflow.iceflow.Nz], [-1], [1]
                )
            )
            i += cfg.processes.iceflow.iceflow.Nz

    return fieldin
