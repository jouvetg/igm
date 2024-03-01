#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os, sys, shutil
import matplotlib.pyplot as plt
import matplotlib
import datetime, time
import tensorflow as tf

from igm.modules.utils import *


def params(parser):
    parser.add_argument(
        "--plt2d_editor",
        type=str,
        default="vs",
        help="Optimized for VS code (vs) or spyder (sp) for live plot",
    )
    parser.add_argument(
        "--plt2d_live",
        type=str2bool,
        default=False,
        help="Display plots live the results during computation instead of making png",
    )


def initialize(params, state):
    state.extent = [np.min(state.x), np.max(state.x), np.min(state.y), np.max(state.y)]

    if params.plt2d_editor == "vs":
        plt.ion()  # enable interactive mode

    state.fig = plt.figure(dpi=200)
    state.ax = state.fig.add_subplot(1, 1, 1)
    state.ax.axis("off")
    state.ax.set_aspect("equal")

    os.system(
        "echo rm temperature*.png >> clean.sh"
    )


def update(params, state):
    if state.saveresult:
        abs = tf.tile(tf.expand_dims(state.X, 0), (params.iflo_Nz, 1, 1)).numpy()
        ele = np.expand_dims(state.thk, 0) - state.depth.numpy()
        tem = (state.T - 273.15).numpy()
        ome = state.omega.numpy()

        X = np.ndarray.flatten(abs[:, 100, 100:]) / 25
        Z = np.ndarray.flatten(ele[:, 100, 100:])
        T = np.ndarray.flatten(tem[:, 100, 100:])
        O = np.ndarray.flatten(ome[:, 100, 100:])

        im = state.ax.scatter(
            X, Z, c=T, s=0.2, vmin=-10, vmax=0, cmap=matplotlib.cm.jet
        )
        #        im = state.ax.scatter( X, Z, c=O, s=0.2, vmin=0, vmax=0.03, cmap=matplotlib.cm.viridis )

        #        XX = state.X[100,100:].numpy()
        #        YY = state.basalMeltRate[100,100:].numpy()
        #        im = state.ax.plot( XX, YY ,'-k') ; state.already_set_cbar = True

        state.ax.set_title("YEAR : " + str(state.t.numpy()), size=15)

        if not hasattr(state, "already_set_cbar"):
            state.cbar = plt.colorbar(im, label="temperature")
            state.already_set_cbar = True

        if params.plt2d_live:
            if params.plt2d_editor == "vs":
                state.fig.canvas.draw()  # re-drawing the figure
                state.fig.canvas.flush_events()  # to flush the GUI events
            else:
                from IPython.display import display, clear_output

                clear_output(wait=True)
                display(state.fig)
        else:
            plt.savefig(
                os.path.join(
                    params.working_dir,
                    "temperature-" + str(state.t.numpy()).zfill(4) + ".png",
                ),
                bbox_inches="tight",
                pad_inches=0.2,
            )


def finalize(params, state):
    pass
