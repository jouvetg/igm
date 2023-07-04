#!/usr/bin/env python3

"""
Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
Published under the GNU GPL (Version 3), check at the LICENSE file
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from igm.modules.utils import complete_data

def params_make_synthetic(parser):
    pass


def init_make_synthetic(params, self):
    """
    Make a synthetic glacier bedrock
    """
    x = np.arange(0, 100) * 100  # make x-axis, lenght 10 km, resolution 100 m
    y = np.arange(0, 200) * 100  # make y-axis, lenght 20 km, resolution 100 m

    X, Y = np.meshgrid(x, y)

    topg = 1000 + 0.15 * Y + ((X - 5000) ** 2) / 50000  # define the bedrock topography
    thk = np.zeros_like(topg)

    self.x = tf.constant(x.astype("float32"))
    self.y = tf.constant(y.astype("float32"))

    self.topg = tf.Variable(topg.astype("float32"))
    self.thk = tf.Variable(thk.astype("float32"))

    complete_data(self)


def update_make_synthetic(params, self):
    pass