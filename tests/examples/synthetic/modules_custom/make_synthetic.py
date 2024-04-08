#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file 

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from igm.modules.utils import complete_data

def params(parser):
    pass


def initialize(params, state):

    x = np.arange(0, 100) * 100  # make x-axis, lenght 10 km, resolution 100 m
    y = np.arange(0, 200) * 100  # make y-axis, lenght 20 km, resolution 100 m

    X, Y = np.meshgrid(x, y)

    topg = 1000 + 0.15 * Y + ((X - 5000) ** 2) / 50000  # define the bedrock topography
    thk = np.zeros_like(topg)

    state.x = tf.constant(x.astype("float32"))
    state.y = tf.constant(y.astype("float32"))

    state.topg = tf.Variable(topg.astype("float32"))
    state.thk = tf.Variable(thk.astype("float32"))

    complete_data(state)


def update(params, state):
    pass


def finalize(params, state):
    pass
    
