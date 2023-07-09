#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
This IGM modules compute time step dt (computed to satisfy the CFL condition),
updated time t, and a boolean telling whether results must be saved or not.
For stability reasons of the transport scheme for the ice thickness evolution,
the time step must respect a CFL condition, controlled by parameter params.cfl,
which is the maximum number of cells crossed in one iteration
(this parameter cannot exceed one).

==============================================================================

Input  : self.ubar, self.vbar, self.dx 
Output : self.dt, self.t, self.it, self.saveresult 
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime, time
import tensorflow as tf


def params_time_step(parser):
    parser.add_argument(
        "--tstart",
        type=float,
        default=2000.0,
        help="Start modelling time (default 2000)",
    )
    parser.add_argument(
        "--tend", type=float, default=2100.0, help="End modelling time (default: 2100)"
    )
    parser.add_argument(
        "--tsave", type=float, default=10, help="Save result each X years (default: 10)"
    )
    parser.add_argument(
        "--cfl",
        type=float,
        default=0.3,
        help="CFL number for the stability of the mass conservation scheme, \
        it must be below 1 (Default: 0.3)",
    )
    parser.add_argument(
        "--dtmax",
        type=float,
        default=10.0,
        help="Maximum time step allowed, used only with slow ice (default: 10.0)",
    )


def init_time_step(params, self):
    self.tcomp["time_step"] = []

    # Initialize the time with starting time
    self.t = tf.Variable(float(params.tstart))

    self.it = 0

    self.dt = tf.Variable(float(params.dtmax))

    self.dt_target = tf.Variable(float(params.dtmax))

    self.tsave = np.ndarray.tolist(
        np.arange(params.tstart, params.tend, params.tsave)
    ) + [params.tend]

    self.tsave = tf.constant(self.tsave)

    self.itsave = 0

    self.saveresult = True


def update_time_step(params, self):

    self.logger.info(
        "Update DT from the CFL condition at time : " + str(self.t.numpy())
    )

    self.tcomp["time_step"].append(time.time())

    # compute maximum ice velocitiy magnitude
    velomax = max(
        tf.math.reduce_max(tf.math.abs(self.ubar)),
        tf.math.reduce_max(tf.math.abs(self.vbar)),
    )

    # dt_target account for both cfl and dt_max
    if velomax > 0:
        self.dt_target = min(params.cfl * self.dx / velomax, params.dtmax)
    else:
        self.dt_target = params.dtmax

    self.dt = self.dt_target

    # modify dt such that times of requested savings are reached exactly
    if self.tsave[self.itsave + 1] <= self.t + self.dt:
        self.dt = self.tsave[self.itsave + 1] - self.t
        self.saveresult = True
        self.itsave += 1
    else:
        self.saveresult = False

    self.t.assign(self.t + self.dt)

    self.it += 1

    self.tcomp["time_step"][-1] -= time.time()
    self.tcomp["time_step"][-1] *= -1


def final_time_step(params, self):
    pass
