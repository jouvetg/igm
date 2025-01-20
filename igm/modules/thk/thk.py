#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import datetime, time
import tensorflow as tf

from igm.modules.utils import compute_divflux_slope_limiter
# from igm.tests.Bueler2005 import initialize as initialize_Bueler

def params(parser):
    parser.add_argument(
        "--thk_slope_type",
        type=str,
        default="superbee",
        help="Type of slope limiter for the ice thickness equation (godunov or superbee)",
    )
    parser.add_argument(
        "--thk_ratio_density",
        type=float,
        default=0.910,
        help="density of ice divided by density of water",
    )
    parser.add_argument(
        "--thk_default_sealevel",
        type=float,
        default=0.0,
        help="Default sea level if not provided by the user",
    )

def initialize(cfg, state):

    if not hasattr(state, "topg"):
        raise ValueError("The 'thk' module requires an initial topography ('state.topg') to be defined. Please define it through the preprocessing steps (not yet implemented)")
        
    # define the lower ice surface
    if hasattr(state, "sealevel"):
        # ! This is not clear which modules provides state.topg and allows us to use state.thk!!!
        state.lsurf = tf.maximum(state.topg,-cfg.modules.thk.ratio_density*state.thk + state.sealevel)
    else:
        state.lsurf = tf.maximum(state.topg,-cfg.modules.thk.ratio_density*state.thk + cfg.modules.thk.default_sealevel)

    # define the upper ice surface
    state.usurf = state.lsurf + state.thk

    state.tcomp_thk = []

def update(cfg, state):

    if state.it >= 0:
        if hasattr(state, "logger"):
            state.logger.info(
                "Ice thickness equation at time : " + str(state.t.numpy())
            )

        state.tcomp_thk.append(time.time())

        # compute the divergence of the flux
        state.divflux = compute_divflux_slope_limiter(
            state.ubar, state.vbar, state.thk, state.dx, state.dx, state.dt, slope_type=cfg.modules.thk.slope_type
        )

        # if not smb model is given, set smb to zero
        if not hasattr(state, "smb"):
            state.smb = tf.zeros_like(state.thk)

        # Forward Euler with projection to keep ice thickness non-negative
        state.thk = tf.maximum(state.thk + state.dt * (state.smb - state.divflux), 0)

        # define the lower ice surface
        if hasattr(state, "sealevel"):
            state.lsurf = tf.maximum(state.topg,-cfg.modules.thk.ratio_density*state.thk + state.sealevel)
        else:
            state.lsurf = tf.maximum(state.topg,-cfg.modules.thk.ratio_density*state.thk + cfg.modules.thk.default_sealevel)

        # define the upper ice surface
        state.usurf = state.lsurf + state.thk

        state.tcomp_thk[-1] -= time.time()
        state.tcomp_thk[-1] *= -1


def finalize(params, state):
    pass
