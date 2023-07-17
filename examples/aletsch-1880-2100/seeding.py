# Import the most important libraries
import numpy as np
import tensorflow as tf
import igm
from netCDF4 import Dataset
import os

def init_particles(params, state):
    state.tlast_seeding = -1.0e5000
    state.tcomp_particles = []

    # initialize trajectories
    state.xpos = tf.Variable([])
    state.ypos = tf.Variable([])
    state.zpos = tf.Variable([])
    state.rhpos = tf.Variable([])
    state.wpos = tf.Variable([])  # this is to give a weight to the particle
    state.tpos = tf.Variable([])
    state.englt = tf.Variable([])

    # build the gridseed
    state.gridseed = np.zeros_like(state.thk) == 1
    rr = int(1.0 / params.density_seeding)
    state.gridseed[::rr, ::rr] = True
    
    nc = Dataset( os.path.join(params.working_dir, 'seeding.nc'), "r" ) 
    state.seeding = np.squeeze( nc.variables["seeding"] ).astype("float32") 
    nc.close()

def seeding_particles(params, state):
    # here we seed where i) thickness is higher than 1 m
    #                    ii) the seeding field of geology.nc is active
    #                    iii) on the gridseed (which permit to control the seeding density)
    I = (state.thk>1)&(state.seeding>0.5)&state.gridseed  # here you may redefine how you want to seed particles
    state.nxpos  = state.X[I]                # x position of the particle
    state.nypos  = state.Y[I]                # y position of the particle
    state.nzpos  = state.usurf[I]            # z position of the particle
    state.nrhpos = tf.ones_like(state.X[I])  # relative position in the ice column
    state.nwpos  = tf.ones_like(state.X[I])  # this is the weight of the particle
    state.ntpos  = tf.ones_like(state.X[I]) * state.t
    state.nenglt = tf.zeros_like(state.X[I])

igm.seeding_particles = seeding_particles
igm.init_particles    = init_particles

