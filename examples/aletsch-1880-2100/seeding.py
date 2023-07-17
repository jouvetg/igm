# Import the most important libraries
import numpy as np
import tensorflow as tf
import igm

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

