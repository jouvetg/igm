#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math
 
def run(cfg, state):

    state.x = tf.constant(np.linspace(-3950,3950,80).astype("float32"))
    state.y = tf.constant(np.linspace(-3950,3950,80).astype("float32"))

    nx = state.x.shape[0]
    ny = state.y.shape[0]
    
    state.usurf = tf.Variable(tf.zeros((ny,nx)))
    state.thk   = tf.Variable(tf.zeros((ny,nx)))
    state.topg  = tf.Variable(tf.zeros((ny,nx)))

    state.X, state.Y = tf.meshgrid(state.x, state.y)

    state.dx = state.x[1] - state.x[0]

    state.dX = tf.ones_like(state.X) * state.dx 
   



