#!/usr/bin/env python3

"""
Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
Published under the GNU GPL (Version 3), check at the LICENSE file
"""


import numpy as np
import os, sys, shutil
import matplotlib.pyplot as plt
import datetime, time
import tensorflow as tf  

from igm.utils import *
 
def params_plot_vs(parser):
    
    parser.add_argument(
        "--plot_live",
        type=str2bool,
        default=False,
        help="Display plots live the results during computation (Default: False)",
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
        default=500,
        help="maximum value of the varplot variable used to adjust the scaling of the colorbar",
    )
    
def init_plot_vs(params,self):
    
    self.extent = [np.min(self.x),np.max(self.x),np.min(self.y),np.max(self.y)]
    
    os.system('echo rm '+ os.path.join(params.working_dir, params.varplot+"*.png") + ' >> clean.sh')

def update_plot_vs(params,self):
    """
    Plot some variables (e.g. thickness, velocity, mass balance) over time
    """

    if self.saveresult: 

        if params.varplot == "velbar_mag":
            self.velbar_mag = getmag(self.ubar, self.vbar)

        firstime = False
        if not hasattr(self, "already_called_update_plot"):
            self.already_called_update_plot = True
            self.tcomp["plot_vs"] = []
            firstime = True 

        self.tcomp["plot_vs"].append(time.time())
 
        if firstime:

            # enable interactive mode
            plt.ion()

            self.fig = plt.figure(dpi=200)
            self.ax = self.fig.add_subplot(1, 1, 1)
            self.ax.axis("off")
            im = self.ax.imshow(
                tf.where(self.thk>1,vars(self)[params.varplot],np.nan),
                origin="lower",
                cmap="viridis",
                vmin=0,
                vmax=params.varplot_max,
                extent=self.extent
            ) 
            self.ip = self.ax.scatter(
                x=[self.x[0]],
                y=[self.y[0]],
                c=[1],
                vmin=0,
                vmax=1,
                s=0.5,
                cmap="RdBu",
            )
            self.ax.set_title("YEAR : " + str(self.t.numpy()), size=15)
            self.cbar = plt.colorbar(im,label=params.varplot)
 
        else:
            im = self.ax.imshow(
                tf.where(self.thk>1,vars(self)[params.varplot],np.nan),
                origin="lower",
                cmap="viridis",
                vmin=0,
                vmax=params.varplot_max,
                extent=self.extent
            )
            if hasattr(self,'xpos'):
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

        if params.plot_live:
            # re-drawing the figure
            self.fig.canvas.draw()
            # to flush the GUI events
            self.fig.canvas.flush_events()

        else:
            plt.savefig(
                os.path.join(
                    params.working_dir,
                    params.varplot
                    + "-"
                    + str(self.t.numpy()).zfill(4)
                    + ".png",
                ),
                bbox_inches="tight",
                pad_inches=0.2,
            )
            
        self.tcomp["plot_vs"][-1] -= time.time()
        self.tcomp["plot_vs"][-1] *= -1
