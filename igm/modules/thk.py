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

Input  : self.ubar, self.vbar, self.thk, self.dx, 
Output : self.thk, self.usurf, self.lsurf
"""

import datetime, time
import tensorflow as tf

from igm.modules.utils import compute_divflux


def params_thk(self):
    pass


def init_thk(params, self):
    self.tcomp["thk"] = []


def update_thk(params, self):

    self.logger.info("Ice thickness equation at time : " + str(self.t.numpy()))

    self.tcomp["thk"].append(time.time())

    # compute the divergence of the flux
    self.divflux = compute_divflux(self.ubar, self.vbar, self.thk, self.dx, self.dx)

    # Forward Euler with projection to keep ice thickness non-negative
    self.thk = tf.maximum(self.thk + self.dt * (self.smb - self.divflux), 0)

    # TODO: replace 0.9 by physical constant, and add SL value
    # define the lower ice surface
    self.lsurf = tf.maximum(self.topg,-0.9*self.thk)

    # define the upper ice surface
    self.usurf = self.lsurf + self.thk

    self.tcomp["thk"][-1] -= time.time()
    self.tcomp["thk"][-1] *= -1
