#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import tensorflow as tf

from igm.modules.utils import *

def params(parser):
    parser.add_argument(
        "--print_mem_info",
        type=str2bool,
        default=False,
        help="Additional info on memory usage",
    )

def initialize(params, state):
    if params.print_mem_info:
        print(
            "IGM %s :         Iterations   |         Time (y)     |     Time Step (y)   |   Ice Volume (km^3)   |   GPU memory (MB) "
        )
    else:
        print(
            "IGM %s :         Iterations   |         Time (y)     |     Time Step (y)   |   Ice Volume (km^3) "
        )


def update(params, state):
    """
    This serves to print key info on the fly during computation
    """
    if state.saveresult:

        if params.print_mem_info:
            gpu_info = tf.config.experimental.get_memory_info("GPU:0")
            print(
                "IGM %s :      %6.0f    |      %8.0f        |     %7.2f        |     %10.2f         |     %10.2f "
                % (
                    datetime.datetime.now().strftime("%H:%M:%S"),
                    state.it,
                    state.t,
                    state.dt_target,
                    np.sum(state.thk) * (state.dx**2) / 10**9,
                    gpu_info['current'] / 1024**2
                )
            )
        else:    
            print(
                "IGM %s :      %6.0f    |      %8.0f        |     %7.2f        |     %10.2f "
                % (
                    datetime.datetime.now().strftime("%H:%M:%S"),
                    state.it,
                    state.t,
                    state.dt_target,
                    np.sum(state.thk) * (state.dx**2) / 10**9
                )
            )

def finalize(params, state):
    pass
