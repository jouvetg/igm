#!/usr/bin/env python3

"""
Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
Published under the GNU GPL (Version 3), check at the LICENSE file
"""


import numpy as np
import os, sys, shutil
import matplotlib.pyplot as plt
import matplotlib
import datetime, time
import tensorflow as tf

from igm.modules.utils import *

def params_write_plot2d(parser):
    parser.add_argument(
        "--editor_plot2d",
        type=str,
        default="vs",
        help="optimized for VS code (vs) or spyder (sp) for live plot",
    )
    parser.add_argument(
        "--plot_live",
        type=str2bool,
        default=False,
        help="Display plots live the results during computation (Default: False)",
    )
    parser.add_argument(
        "--plot_particles",
        type=str2bool,
        default=True,
        help="Display particles (Default: True)",
    )
    parser.add_argument(
        "--varplot",
        type=str,
        default="velbar_mag",
        help="variable to plot",
    )
    parser.add_argument(
        "--varplot_max",
        type=float,
        default=250,
        help="maximum value of the varplot variable used to adjust the scaling of the colorbar",
    )


def init_write_plot2d(params, self):
    self.extent = [np.min(self.x), np.max(self.x), np.min(self.y), np.max(self.y)]

    if params.editor_plot2d == "vs":
        # enable interactive mode
        plt.ion()

    self.tcomp["write_plot2d"] = []

    self.fig = plt.figure(dpi=200)
    self.ax = self.fig.add_subplot(1, 1, 1)
    self.ax.axis("off")
    self.ax.set_aspect("equal")

    os.system(
        "echo rm "
        + os.path.join(params.working_dir, params.varplot + "*.png")
        + " >> clean.sh"
    )


def update_write_plot2d(params, self):
    """
    Plot some variables (e.g. thickness, velocity, mass balance) over time
    """

    if self.saveresult:
 
        self.tcomp["write_plot2d"].append(time.time())

        if params.varplot == "velbar_mag":
            self.velbar_mag = getmag(self.ubar, self.vbar)

        cmap = matplotlib.cm.viridis
        cmap.set_bad(color='white')

        im = self.ax.imshow(
            np.ma.masked_where(self.thk==0, vars(self)[params.varplot]),
            origin="lower",
            cmap=cmap,
            vmin=0,
            vmax=params.varplot_max,
            extent=self.extent,
        )
        if params.plot_particles:
            if hasattr(self, "xpos"):
                if hasattr(self, "ip"):
                    self.ip.set_visible(False)
                r = 1
                self.ip = self.ax.scatter(
                    x=self.xpos[::r],
                    y=self.ypos[::r],
                    c=1 - self.rhpos[::r].numpy(),
                    vmin=0,
                    vmax=1,
                    s=0.5,
                    cmap="RdBu",
                )
        self.ax.set_title("YEAR : " + str(self.t.numpy()), size=15)

        if not hasattr(self,'already_set_cbar'):
                self.cbar = plt.colorbar(im, label=params.varplot)
                self.already_set_cbar = True

        if params.plot_live:
            if params.editor_plot2d == "vs":
                self.fig.canvas.draw()         # re-drawing the figure
                self.fig.canvas.flush_events() # to flush the GUI events
            else:
                from IPython.display import display, clear_output
                clear_output(wait=True)
                display(self.fig)
        else:
            plt.savefig(
                os.path.join(
                    params.working_dir,
                    params.varplot + "-" + str(self.t.numpy()).zfill(4) + ".png",
                ),
                bbox_inches="tight",
                pad_inches=0.2,
            )

        self.tcomp["write_plot2d"][-1] -= time.time()
        self.tcomp["write_plot2d"][-1] *= -1
