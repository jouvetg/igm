#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import datetime, time
import tensorflow as tf

from igm.modules.utils import compute_divflux


def params_thk(parser):
    pass

def initialize_thk(params, state):
    state.tcomp_thk = []


def update_thk(params, state):
    
    if state.it>=0:

        if hasattr(state,'logger'):
            state.logger.info("Ice thickness equation at time : " + str(state.t.numpy()))

        state.tcomp_thk.append(time.time())

        # compute the divergence of the flux
        state.divflux = compute_divflux(state.ubar, state.vbar, state.thk, state.dx, state.dx)

        # if not smb model is given, set smb to zero
        if not hasattr(state, 'smb'):
            state.smb = tf.zeros_like(state.thk)

        # Forward Euler with projection to keep ice thickness non-negative    
        state.thk = tf.maximum(state.thk + state.dt * (state.smb - state.divflux), 0) 

        # TODO: replace 0.9 by physical constant, and add SL value
        # define the lower ice surface
        # state.lsurf = tf.maximum(state.topg,-0.9*state.thk)
        # lower surface is not implmented yet
        state.lsurf = state.topg

        # define the upper ice surface
        state.usurf = state.lsurf + state.thk

        state.tcomp_thk[-1] -= time.time()
        state.tcomp_thk[-1] *= -1


def finalize_thk(params, state):
    pass
