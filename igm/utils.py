#!/usr/bin/env python3

"""
Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
Published under the GNU GPL (Version 3), check at the LICENSE file
"""


import numpy as np
import os, sys, shutil
import matplotlib.pyplot as plt
import tensorflow as tf  
 
def str2bool(v):
    return v.lower() in ("true", "1") 

@tf.function()
def getmag( u, v):
    """
    return the norm of a 2D vector, e.g. to compute velbase_mag
    """
    return tf.norm(
        tf.concat([tf.expand_dims(u, axis=-1), tf.expand_dims(v, axis=-1)], axis=2),
        axis=2,
    )

@tf.function()
def compute_gradient_tf( s, dx, dy):
    """
    compute spatial 2D gradient of a given field
    """

    # EX = tf.concat([s[:, 0:1], 0.5 * (s[:, :-1] + s[:, 1:]), s[:, -1:]], 1)
    # diffx = (EX[:, 1:] - EX[:, :-1]) / dx

    # EY = tf.concat([s[0:1, :], 0.5 * (s[:-1, :] + s[1:, :]), s[-1:, :]], 0)
    # diffy = (EY[1:, :] - EY[:-1, :]) / dy
    
    EX = tf.concat([ 1.5*s[:,0:1] - 0.5*s[:,1:2], 0.5*s[:,:-1] + 0.5*s[:,1:], 1.5*s[:,-1:] - 0.5*s[:,-2:-1] ], 1)
    diffx = (EX[:, 1:] - EX[:, :-1]) / dx

    EY = tf.concat([ 1.5*s[0:1,:] - 0.5*s[1:2,:], 0.5*s[:-1,:] + 0.5*s[1:,:], 1.5*s[-1:,:] - 0.5*s[-2:-1,:] ], 0)
    diffy = (EY[1:, :] - EY[:-1, :]) / dy

    return diffx, diffy

@tf.function()
def compute_divflux( u, v, h, dx, dy):
    """
    #   upwind computation of the divergence of the flux : d(u h)/dx + d(v h)/dy
    #   First, u and v are computed on the staggered grid (i.e. cell edges)
    #   Second, one extend h horizontally by a cell layer on any bords (assuming same value)
    #   Third, one compute the flux on the staggered grid slecting upwind quantities
    #   Last, computing the divergence on the staggered grid yields values def on the original grid
    """

    ## Compute u and v on the staggered grid
    u = tf.concat(
        [u[:, 0:1], 0.5 * (u[:, :-1] + u[:, 1:]), u[:, -1:]], 1
    )  # has shape (ny,nx+1)
    v = tf.concat(
        [v[0:1, :], 0.5 * (v[:-1, :] + v[1:, :]), v[-1:, :]], 0
    )  # has shape (ny+1,nx)

    # Extend h with constant value at the domain boundaries
    Hx = tf.pad(h, [[0, 0], [1, 1]], "CONSTANT")  # has shape (ny,nx+2)
    Hy = tf.pad(h, [[1, 1], [0, 0]], "CONSTANT")  # has shape (ny+2,nx)

    ## Compute fluxes by selcting the upwind quantities
    Qx = u * tf.where(u > 0, Hx[:, :-1], Hx[:, 1:])  # has shape (ny,nx+1)
    Qy = v * tf.where(v > 0, Hy[:-1, :], Hy[1:, :])  # has shape (ny+1,nx)

    ## Computation of the divergence, final shape is (ny,nx)
    return (Qx[:, 1:] - Qx[:, :-1]) / dx + (Qy[1:, :] - Qy[:-1, :]) / dy
  
@tf.function()
def interp1d_tf(xs,ys,x):
     
    x = tf.clip_by_value(x,tf.reduce_min(xs),tf.reduce_max(xs))
    
    # determine the output data type
    ys = tf.convert_to_tensor(ys)
    dtype = ys.dtype
    
    # normalize data types
    ys = tf.cast(ys, tf.float64)
    xs = tf.cast(xs, tf.float64)
    x = tf.cast(x, tf.float64)

    # pad control points for extrapolation
    xs = tf.concat([[xs.dtype.min], xs, [xs.dtype.max]], axis=0)
    ys = tf.concat([ys[:1], ys, ys[-1:]], axis=0)

    # compute slopes, pad at the edges to flatten
    ms = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
    ms = tf.pad(ms[:-1], [(1, 1)])

    # solve for intercepts
    bs = ys - ms*xs

    # search for the line parameters at each input data point
    # create a grid of the inputs and piece breakpoints for thresholding
    # rely on argmax stopping on the first true when there are duplicates,
    # which gives us an index into the parameter vectors
    i = tf.math.argmax(xs[..., tf.newaxis, :] > x[..., tf.newaxis], axis=-1)
    m = tf.gather(ms, i, axis=-1)
    b = tf.gather(bs, i, axis=-1)

    # apply the linear mapping at each input data point
    y = m*x + b
    return tf.cast(tf.reshape(y, tf.shape(x)), dtype)