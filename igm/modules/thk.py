#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
This IGM module solves the mass conservation of ice to update the thickness
from ice flow and surface mass balance. The mass conservation equation
is solved using an explicit first-order upwind finite-volume scheme
on a regular 2D grid with constant cell spacing in any direction.
The discretization and the approximation of the flux divergence is
described [here](https://github.com/jouvetg/igm/blob/main/fig/transp-igm.jpg).
With this scheme mass of ice is allowed to move from cell to cell
(where thickness and velocities are defined) from edge-defined fluxes
(inferred from depth-averaged velocities, and ice thickness in upwind direction).
The resulting scheme is mass conservative and parallelizable (because fully explicit).
However, it is subject to a CFL condition. This means that the time step
(defined in update_t_dt()) is controlled by parameter params.cfl,
which is the maximum number of cells crossed in one iteration
(this parameter cannot exceed one).

==============================================================================

Input  : state.ubar, state.vbar, state.thk, state.dx, 
Output : state.thk, state.usurf, state.lsurf
"""

import datetime, time
import tensorflow as tf

from igm.modules.utils import compute_divflux


def params_thk(state):
    pass

def init_thk(params, state):
    state.tcomp_thk = []


def update_thk(params, state):

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
    state.lsurf = tf.maximum(state.topg,-0.9*state.thk)

    # define the upper ice surface
    state.usurf = state.lsurf + state.thk

    state.tcomp_thk[-1] -= time.time()
    state.tcomp_thk[-1] *= -1


def final_thk(params, state):
    pass
