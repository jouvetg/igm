#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os
import datetime
import matplotlib.pyplot as plt


def params(parser):
    pass


def initialize(params, state):
    print(
        "IGM %s :         Iterations   |         Time (y)     |     Time Step (y)   |   Ice Volume (km^3) "
    )


def update(params, state):
    """
    This serves to print key info on the fly during computation
    """
    if state.saveresult:
        print(
            "IGM %s :      %6.0f    |      %8.0f        |     %7.2f        |     %10.2f "
            % (
                datetime.datetime.now().strftime("%H:%M:%S"),
                state.it,
                state.t,
                state.dt_target,
                np.sum(state.thk) * (state.dx**2) / 10**9,
            )
        )


def finalize(params, state):
    pass
