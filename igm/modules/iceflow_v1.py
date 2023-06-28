#!/usr/bin/env python3

"""
Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
Published under the GNU GPL (Version 3), check at the LICENSE file
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import datetime, time
import math
import tensorflow as tf

from igm.modules.utils import *


def params_iceflow_v1(parser):
    parser.add_argument(
        "--init_strflowctrl",
        type=float,
        default=78,
        help="Initial strflowctrl (default 78)",
    )
    parser.add_argument(
        "--init_slidingco",
        type=float,
        default=0,
        help="Initial sliding coeeficient slidingco (default: 0)",
    )
    parser.add_argument(
        "--init_arrhenius",
        type=float,
        default=78,
        help="Initial arrhenius factor arrhenuis (default: 78)",
    )
    parser.add_argument(
        "--iceflow_model_lib_path",
        type=str,
        default="../../model-lib/f15_cfsflow_GJ_22_a",
        help="Directory path of the deep-learning ice flow model",
    )
    parser.add_argument(
        "--multiple_window_size",
        type=int,
        default=0,
        help="If a U-net, this force window size a multiple of 2**N (default: 0)",
    )
    parser.add_argument(
        "--force_max_velbar",
        type=float,
        default=0,
        help="This permits to artif. upper-bound velocities, active if > 0 (default: 0)",
    )

    # params = self.parser.parse_args()


def init_iceflow_v1(params, self):
    """
    set-up the iceflow emulator
    """

    self.tcomp["iceflow"] = []

    # here we initialize variable parmaetrizing ice flow
    if not hasattr(self, "strflowctrl"):
        self.strflowctrl = tf.Variable(tf.ones_like(self.thk) * params.init_strflowctrl)

    if not hasattr(self, "arrhenius"):
        self.arrhenius = tf.Variable(tf.ones_like(self.thk) * params.init_arrhenius)

    if not hasattr(self, "slidingco"):
        self.slidingco = tf.Variable(tf.ones_like(self.thk) * params.init_slidingco)

    dirpath = os.path.join(params.iceflow_model_lib_path, str(int(self.dx)))

    if not os.path.isdir(dirpath):
        dirpath = params.iceflow_model_lib_path

    # fieldin, fieldout, fieldbounds contains name of I/O variables, and bounds for scaling
    fieldin, fieldout, fieldbounds = _read_fields_and_bounds(self, dirpath)

    self.iceflow_mapping = {}
    self.iceflow_mapping["fieldin"] = fieldin
    self.iceflow_mapping["fieldout"] = fieldout
    self.iceflow_fieldbounds = fieldbounds

    self.iceflow_model = tf.keras.models.load_model(os.path.join(dirpath, "model.h5"))

    # print(self.iceflow_model.summary())

    Ny = self.thk.shape[0]
    Nx = self.thk.shape[1]

    # In case of a U-net, must make sure the I/O size is multiple of 2**N
    if params.multiple_window_size > 0:
        NNy = params.multiple_window_size * math.ceil(Ny / params.multiple_window_size)
        NNx = params.multiple_window_size * math.ceil(Nx / params.multiple_window_size)
        self.PAD = [[0, NNy - Ny], [0, NNx - Nx]]
    else:
        self.PAD = [[0, 0], [0, 0]]


def update_iceflow_v1(params, self):
    """
    Ice flow dynamics are modeled using Artificial Neural Networks trained
    from physical models.

    You may find trained and ready-to-use ice flow emulators in the folder
    `model-lib/T_M_I_Y_V/R/`, where 'T_M_I_Y_V' defines the emulator, and
    R defines the spatial resolution. Make sure that the resolution of the
    picked emulator is available in the database. Results produced with IGM
    will strongly rely on the chosen emulator. Make sure that you use the
    emulator within the hull of its training dataset (e.g., do not model
    an ice sheet with an emulator trained with mountain glaciers) to ensure
    reliability (or fidelity w.r.t to the instructor model) -- the emulator
    is probably much better at interpolating than at extrapolating.
    Information on the training dataset is provided in a dedicated README
    coming along with the emulator.

    At the time of writing, I recommend using *f15_cfsflow_GJ_22_a*, which
    takes ice thickness, top surface slopes, the sliding coefficient c
    ('slidingco'), and Arrhenuis factor A ('arrhenius'), and return basal,
    vertical-average and surface x- and y- velocity components as depicted
    on the following graph:

    ![](https://github.com/jouvetg/igm/blob/main/fig/mapping-f15.png)

    I have trained *f15_cfsflow_GJ_22_a* using a large dataset of modeled
    glaciers (based on a Stokes-based CfsFlow ice flow solver) and varying
    sliding coefficient c, and Arrhenius factor A into a 2D space.

    For now, only the emulator trained by CfsFlow and PISM is available
    with different resolutions. Consider training your own with the
    [Deep Learning Emulator](https://github.com/jouvetg/dle) if none of
    these emulators fill your need.
    """

    self.logger.info("Update ICEFLOW at time : " + str(self.t.numpy()))

    self.tcomp["iceflow"].append(time.time())

    # update gradients of the surface (slopes)
    self.slopsurfx, self.slopsurfy = compute_gradient_tf(self.usurf, self.dx, self.dx)

    # Define the input of the NN, include scaling
    X = tf.expand_dims(
        tf.stack(
            [
                tf.pad(vars(self)[f], self.PAD, "CONSTANT")
                / self.iceflow_fieldbounds[f]
                for f in self.iceflow_mapping["fieldin"]
            ],
            axis=-1,
        ),
        axis=0,
    )

    # Get the ice flow after applying the NN
    Y = self.iceflow_model(X)

    # Appplying scaling, and update variables
    Ny, Nx = self.thk.shape
    for kk, f in enumerate(self.iceflow_mapping["fieldout"]):
        vars(self)[f] = (
            tf.where(self.thk > 0, Y[0, :Ny, :Nx, kk], 0) * self.iceflow_fieldbounds[f]
        )

    # If requested, the speeds are artifically upper-bounded
    if params.force_max_velbar > 0:
        self.velbar_mag = self.getmag(self.ubar, self.vbar)

        self.ubar = tf.where(
            self.velbar_mag >= params.force_max_velbar,
            params.force_max_velbar * (self.ubar / self.velbar_mag),
            self.ubar,
        )
        self.vbar = tf.where(
            self.velbar_mag >= params.force_max_velbar,
            params.force_max_velbar * (self.vbar / self.velbar_mag),
            self.vbar,
        )

    self.tcomp["iceflow"][-1] -= time.time()
    self.tcomp["iceflow"][-1] *= -1


def _read_fields_and_bounds(self, path):
    """
    get fields (input and outputs) from given file
    """

    fieldbounds = {}
    fieldin = []
    fieldout = []

    fid = open(os.path.join(path, "fieldin.dat"), "r")
    for fileline in fid:
        part = fileline.split()
        fieldin.append(part[0])
        fieldbounds[part[0]] = float(part[1])
    fid.close()

    fid = open(os.path.join(path, "fieldout.dat"), "r")
    for fileline in fid:
        part = fileline.split()
        fieldout.append(part[0])
        fieldbounds[part[0]] = float(part[1])
    fid.close()

    return fieldin, fieldout, fieldbounds
