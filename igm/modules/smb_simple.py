#!/usr/bin/env python3

"""
Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
Published under the GNU GPL (Version 3), check at the LICENSE file
"""

import numpy as np
import os, sys, shutil
import time
import tensorflow as tf
from igm.modules.utils import interp1d_tf


def params_smb_simple(parser):
    parser.add_argument(
        "--mb_update_freq",
        type=float,
        default=1,
        help="Update the mass balance each X years (1)",
    )
    parser.add_argument(
        "--mb_simple_file",
        type=str,
        default="mb_simple_param.txt",
        help="Name of the imput file for the simple mass balance model",
    )


def init_smb_simple(params, self):
    """
    Read parameters of simple mass balance from an ascii file
    """
    self.smbpar = np.loadtxt(
        os.path.join(params.working_dir, params.mb_simple_file),
        skiprows=1,
        dtype=np.float32,
    )

    self.tcomp["smb_simple"] = []
    self.tlast_mb = tf.Variable(-1.0e5000)


def update_smb_simple(params, self):
    """
    Surface Mass balance implementation 'simple' parametrized by ELA, ablation
    and accumulation gradients, and max acuumulation from a given file mb_simple_file

    Inputs fields
    ----------
    self.usurf

    Output fields
    ----------
    self.smb

    """

    # update smb each X years
    if (self.t - self.tlast_mb) >= params.mb_update_freq:
        self.logger.info("Construct mass balance at time : " + str(self.t.numpy()))

        self.tcomp["smb_simple"].append(time.time())

        # get the smb parameters at given time t
        gradabl = interp1d_tf(self.smbpar[:, 0], self.smbpar[:, 1], self.t)
        gradacc = interp1d_tf(self.smbpar[:, 0], self.smbpar[:, 2], self.t)
        ela = interp1d_tf(self.smbpar[:, 0], self.smbpar[:, 3], self.t)
        maxacc = interp1d_tf(self.smbpar[:, 0], self.smbpar[:, 4], self.t)

        # compute smb from glacier surface elevation and parameters
        self.smb = self.usurf - ela
        self.smb *= tf.where(tf.less(self.smb, 0), gradabl, gradacc)
        self.smb = tf.clip_by_value(self.smb, -100, maxacc)

        # if an icemask exists, then force negative smb aside to prevent leaks
        if hasattr(self, "icemask"):
            self.smb = tf.where(self.icemask > 0.5, self.smb, -10)

        self.tlast_mb.assign(self.t)

        self.tcomp["smb_simple"][-1] -= time.time()
        self.tcomp["smb_simple"][-1] *= -1
