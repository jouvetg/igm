#!/usr/bin/env python3

# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time


def params_gflex(parser):
    parser.add_argument(
        "--gflex_update_freq",
        type=float,
        default=100.0,
        help="Update gflex each X years (1)",
    )
    parser.add_argument(
        "--gflex_default_Te",
        type=float,
        default=50000,
        help="Default value for Te (Elastic thickness [m]) if not given as ncdf file",
    )


def initialize_gflex(params, state):
    import gflex

    state.tcomp_gflex = []
    state.tlast_gflex = tf.Variable(params.time_start, dtype=tf.float32)

    state.flex = gflex.F2D()

    state.flex.giafreq = params.gflex_update_freq
    state.flex.giatime = 1000
    state.flex.Quiet = False
    state.flex.Method = "FD"
    state.flex.PlateSolutionType = "vWC1994"
    state.flex.Solver = "direct"
    state.flex.g = 9.81
    state.flex.E = 100e9
    state.flex.nu = 0.25
    state.flex.rho_m = 3300.0
    state.flex.rho_fill = 1900
    state.flex.dx = state.dx.numpy()
    state.flex.dy = state.dx.numpy()
    state.flex.BC_W = "Periodic"
    state.flex.BC_E = "Periodic"
    state.flex.BC_S = "Periodic"
    state.flex.BC_N = "Periodic"

    if not hasattr(state, "Te"):
        state.flex.Te = np.ones_like(state.thk.numpy() * params.gflex_default_Te)


def update_gflex(params, state):
    import gflex

    if (state.t - state.tlast_gflex) >= params.gflex_update_freq:
        if hasattr(state, "logger"):
            state.logger.info("Update gflex at time : " + str(state.t.numpy()))

        state.tcomp_gflex.append(time.time())

        state.flex.qs = state.thk.numpy() * 917 * 9.81  # Populating this template
        state.flex.initialize()
        state.flex.run()
        state.flex.finalize()
        state.topg = state.topg + state.flex.w
        state.usurf = state.topg + state.thk
        # state.flex.output()

        state.tlast_gflex.assign(state.t)

        state.tcomp_gflex[-1] -= time.time()
        state.tcomp_gflex[-1] *= -1


def finalize_gflex(params, state):
    pass
