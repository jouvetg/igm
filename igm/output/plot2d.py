#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os, sys, shutil
import matplotlib.pyplot as plt
import matplotlib
import datetime, time
import tensorflow as tf

from igm.modules.utils import *


# def params(parser):
#     parser.add_argument(
#         "--plt2d_editor",
#         type=str,
#         default="vs",
#         help="Optimized for VS code (vs) or spyder (sp) for live plot",
#     )
#     parser.add_argument(
#         "--plt2d_live",
#         type=str2bool,
#         default=False,
#         help="Display plots live the results during computation instead of making png",
#     )
#     parser.add_argument(
#         "--plt2d_particles",
#         type=str2bool,
#         default=True,
#         help="Display particles is True, does not display if False",
#     )
#     parser.add_argument(
#         "--plt2d_var",
#         type=str,
#         default="velbar_mag",
#         help="Name of the variable to plot",
#     )
#     parser.add_argument(
#         "--plt2d_var_max",
#         type=float,
#         default=1000,
#         help="Maximum value of the varplot variable used to adjust the scaling of the colorbar",
#     )


def initialize(cfg, state):
    state.extent = [np.min(state.x), np.max(state.x), np.min(state.y), np.max(state.y)]

    if cfg.output.plot2d.editor == "vs":
        plt.ion()  # enable interactive mode

    state.tcomp_plot2d = []

    state.fig = plt.figure(dpi=200)
    state.ax = state.fig.add_subplot(1, 1, 1)
    state.ax.axis("off")
    state.ax.set_aspect("equal")

def run(cfg, state):
    if state.saveresult:
        state.tcomp_plot2d.append(time.time())

        if cfg.output.plot2d.var == "velbar_mag":
            state.velbar_mag = getmag(state.ubar, state.vbar)

        im0 = state.ax.imshow(
            state.topg,
            origin="lower",
            cmap='binary', # matplotlib.cm.terrain,
            extent=state.extent
#            alpha=0.65,
        )
 
        if cfg.output.plot2d.var=="velbar_mag":
            im = state.ax.imshow(
                np.where(state.thk > 0, vars(state)[cfg.output.plot2d.var], np.nan),
                origin="lower",
                cmap="turbo",
                extent=state.extent, 
                norm=matplotlib.colors.LogNorm(vmin=1, vmax=cfg.output.plot2d.var_max)
            )
        else:
            im = state.ax.imshow(
                np.where(state.thk > 0, vars(state)[cfg.output.plot2d.var], np.nan),
                origin="lower",
                cmap='jet',
                vmin=0,
                vmax=cfg.output.plot2d.var_max,
                extent=state.extent,
            )
        if cfg.output.plot2d.particles:
            if hasattr(state, "particle_x"):
                if hasattr(state, "ip"):
                    state.ip.set_visible(False)
                r = 1
                state.ip = state.ax.scatter(
                    x = state.particle_x[::r] + state.x[0],
                    y = state.particle_y[::r] + state.y[0],
                    c = 1 - state.particle_r[::r].numpy(), #or r ?
                    vmin=0,
                    vmax=1,
                    s=0.5,
                    cmap="RdBu",
                )
        state.ax.set_title("YEAR : " + str(state.t.numpy()), size=15)

        if not hasattr(state, "already_set_cbar"):
            state.cbar = plt.colorbar(im, label=cfg.output.plot2d.var)
            state.already_set_cbar = True

        if cfg.output.plot2d.live:
            if cfg.output.plot2d.editor == "vs":
                state.fig.canvas.draw()  # re-drawing the figure
                state.fig.canvas.flush_events()  # to flush the GUI events
            else:
                from IPython.display import display, clear_output

                clear_output(wait=True)
                display(state.fig)
        else:
            plt.savefig(
                cfg.output.plot2d.var + "-" + str(state.t.numpy()).zfill(4) + ".png",
                bbox_inches="tight",
                pad_inches=0.2,
            )

        state.tcomp_plot2d[-1] -= time.time()
        state.tcomp_plot2d[-1] *= -1


def finalize(params, state):
    pass
