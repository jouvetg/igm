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

from igm.modules.utils import *

def params_plot3d_vs(parser):
    parser.add_argument(
        "--plot_live",
        type=str2bool,
        default=False,
        help="Display plots live the results during computation (Default: False)",
    )

def init_plot3d_vs(params, state):
    state.tcomp["plot3d_vs"] = []

def update_plot3d_vs(params, state):
    """
    3d Plot using mayavi library
    """
    from mayavi import mlab

    if state.saveresult:

        state.tcomp["plot3d_vs"].append(time.time())

        state.velbar_mag = getmag(state.ubar, state.vbar)

        if not hasattr(state, "already_called_update_plot3d"):

            #surf = mlab.surf(state.x,state.y,np.transpose(tf.where(state.thk<1,np.nan,state.usurf)), color=state.velbar_mag, warp_scale=1)  
            #base = mlab.surf(state.x,state.y,np.transpose(state.topg), color=(1,1,1), warp_scale=1) 

            state.already_called_update_plot3d = True
 
            mlab.figure(bgcolor=(0.16, 0.28, 0.46))

            X=tf.where(state.thk<1,np.nan,state.X)
            Y=tf.where(state.thk<1,np.nan,state.Y)
            Z=tf.where(state.thk<1,np.nan,state.usurf)
            C=tf.where(state.thk<1,np.nan,state.velbar_mag)
 
            state.plot3d_base = mlab.mesh(state.X,state.Y,state.topg,  colormap='terrain',opacity=0.75)
            state.plot3d_surf = mlab.mesh(X,Y,Z, scalars=C, colormap='jet')

        else:
 
            state.plot3d_surf.mlab_source.z       = tf.where(state.thk<1,np.nan,state.usurf).numpy()
            state.plot3d_surf.mlab_source.scalars = tf.where(state.thk<1,np.nan,state.velbar_mag).numpy()
  

#        if params.plot_live:
#            mlab.draw() # figure=state.fig
#            mlab.savefig( os.path.join( params.working_dir, "3d-" + str(state.t.numpy()).zfill(4) + ".png" ) )
#            time.sleep(1)
#            mlab.show()


        state.tcomp["plot3d_vs"][-1] -= time.time()
        state.tcomp["plot3d_vs"][-1] *= -1


#############################

 
#from mayavi import mlab

#plt.ion()

#mlab.figure(size=(1024, 2048), bgcolor=(0.16, 0.28, 0.46))
 
#surf = mlab.mesh(state.X,state.Y,state.topg,scalars=state.topg)

#surf.mlab_source.z = state.usurf.numpy()
#surf.mlab_source.z = state.topg.numpy()
#surf.mlab_source.scalars = state.thk.numpy()
