#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os, sys, shutil
import time
import tensorflow as tf
from igm.modules.utils import interp1d_tf


def params(parser):
    parser.add_argument(
        "--smb_simple_update_freq",
        type=float,
        default=1,
        help="Update the mass balance each X years (1)",
    )
    parser.add_argument(
        "--smb_simple_file",
        type=str,
        default="smb_simple_param.txt",
        help="Name of the imput file for the simple mass balance model (time, gradabl, gradacc, ela, accmax)",
    )
    parser.add_argument(
        "--smb_simple_array",
        type=list,
        default=[],
        help="Time dependent parameters for simple mass balance model (time, gradabl, gradacc, ela, accmax)",
    )


def initialize(params, state):
    if params.smb_simple_array == []:
        state.smbpar = np.loadtxt(
            params.smb_simple_file,
            skiprows=1,
            dtype=np.float32,
        )
    else:
        state.smbpar = np.array(params.smb_simple_array[1:]).astype(np.float32)

    state.tcomp_smb_simple = []
    state.tlast_mb = tf.Variable(-1.0e5000)


def update(params, state):
    # update smb each X years
    if (state.t - state.tlast_mb) >= params.smb_simple_update_freq:
        if hasattr(state, "logger"):
            state.logger.info(
                "Construct mass balance at time : " + str(state.t.numpy())
            )

        state.tcomp_smb_simple.append(time.time())

        # get the smb parameters at given time t
        gradabl = interp1d_tf(state.smbpar[:, 0], state.smbpar[:, 1], state.t)
        gradacc = interp1d_tf(state.smbpar[:, 0], state.smbpar[:, 2], state.t)
        ela = interp1d_tf(state.smbpar[:, 0], state.smbpar[:, 3], state.t)
        maxacc = interp1d_tf(state.smbpar[:, 0], state.smbpar[:, 4], state.t)

        # compute smb from glacier surface elevation and parameters
        state.smb = state.usurf - ela
        state.smb *= tf.where(tf.less(state.smb, 0), gradabl, gradacc)
        state.smb = tf.clip_by_value(state.smb, -100, maxacc)

        # if an icemask exists, then force negative smb aside to prevent leaks
        if hasattr(state, "icemask"):
            state.smb = tf.where(
                (state.smb < 0) | (state.icemask > 0.5), state.smb, -10
            )

        state.tlast_mb.assign(state.t)

        state.tcomp_smb_simple[-1] -= time.time()
        state.tcomp_smb_simple[-1] *= -1


def finalize(params, state):
    pass
