#!/usr/bin/env python3

"""
Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
Published under the GNU GPL (Version 3), check at the LICENSE file
"""

import numpy as np
import os, sys, shutil
import time
import tensorflow as tf

from igm.modules.utils import *


def params_topg_glacial_erosion(parser):
    parser.add_argument(
        "--erosion_cst",
        type=float,
        default=2.7 * 10 ** (-7),
        help="Erosion multiplicative factor, here taken from Herman, F. et al. \
              Erosion by an Alpine glacier. Science 350, 193–195 (2015)",
    )
    parser.add_argument(
        "--erosion_exp",
        type=float,
        default=2,
        help="Erosion exponent factor, here taken from Herman, F. et al. \
               Erosion by an Alpine glacier. Science 350, 193–195 (2015)",
    )
    parser.add_argument(
        "--erosion_update_freq",
        type=float,
        default=1,
        help="Update the erosion only each X years (Default: 100)",
    )


def init_topg_glacial_erosion(params, self):
    """ """

    self.tcomp["topg_glacial_erosion"] = []
    self.tlast_erosion = tf.Variable(params.tstart)


def update_topg_glacial_erosion(params, self):
    """
    This function implements change in basal topography (due to glacial erosion
    The bedrock is updated (each glacier.config.erosion_update_freq years) assuming the erosion
    rate to be proportional (parameter glacier.config.erosion_cst) to a power (parameter glacier.config.erosion_exp)
    of the sliding velocity magnitude. By default, we use the parameters from Herman,
    F. et al. Erosion by an Alpine glacier. Science 350, 193–195 (2015).

    """

    if (self.t - self.tlast_erosion) >= params.erosion_update_freq:

        self.logger.info("update topg_glacial_erosion at time : " + str(self.t.numpy()))

        self.tcomp["topg_glacial_erosion"].append(time.time())

        velbase_mag = getmag(self.U[0, 0], self.U[1, 0])

        # apply erosion law, erosion rate is proportional to a power of basal sliding speed
        dtopgdt = params.erosion_cst * (velbase_mag**params.erosion_exp)

        self.topg = self.topg - (self.t - self.tlast_erosion) * dtopgdt

        self.usurf = self.topg + self.thk

        self.tlast_erosion.assign(self.t)

        self.tcomp["topg_glacial_erosion"][-1] -= time.time()
        self.tcomp["topg_glacial_erosion"][-1] *= -1
