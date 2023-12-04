#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os, sys, shutil
import matplotlib.pyplot as plt
import tensorflow as tf
import igm
import time

from igm.modules.utils import *


def params(parser):
    parser.add_argument(
        "--rock_flow_speed",
        type=float,
        default=1,
        help="Speed of rock flow along the slope in m/y",
    )


def initialize(params, state):
    pass


def update(params, state):
    slopsurfx, slopsurfy = compute_gradient_tf(state.usurf, state.dx, state.dx)

    slop = getmag(slopsurfx, slopsurfy)

    dirx = -params.rock_flow_speed * tf.where(
        tf.not_equal(slop, 0), slopsurfx / slop, 1
    )
    diry = -params.rock_flow_speed * tf.where(
        tf.not_equal(slop, 0), slopsurfy / slop, 1
    )

    thkexp = tf.repeat(tf.expand_dims(state.thk, axis=0), state.U.shape[0], axis=0)

    state.U.assign(tf.where(thkexp > 0, state.U, dirx))
    state.V.assign(tf.where(thkexp > 0, state.V, diry))


def finalize(params, state):
    pass
