#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file 

"""

Track observation of Aletsch, compare to existing dem

==============================================================================

Input: ---
Output: ----
"""

# Import the most important libraries
import numpy as np
import os, sys, shutil
import matplotlib.pyplot as plt
import tensorflow as tf
import igm
import time


def params_track_usurf_obs(parser):
    pass

def init_track_usurf_obs(params,state):
 
    # load the surface toporgaphy available at given year
    state.usurf = vars(state)['surf_'+str(int(params.tstart))]
    state.thk   = state.usurf-state.topg

def update_track_usurf_obs(params,state):

    if state.t in [1880,1926,1957,1980,1999,2009,2017]:

        diff = (state.usurf-vars(state)['surf_'+str(int(state.t))]).numpy()
        diff = diff[state.thk>1]
        mean  = np.mean(diff)
        std   = np.std(diff)
        vol   = np.sum(state.thk) * (state.dx ** 2) / 10 ** 9
        print(" Check modelled vs observed surface at time : %8.0f ; Mean discr. : %8.2f  ;  Std : %8.2f |  Ice volume : %8.2f " \
                % (state.t, mean, std, vol) )


def  final_track_usurf_obs(params,state):
    pass

# make sure to make these function new attributes of the igm module
igm.params_track_usurf_obs  = params_track_usurf_obs  
igm.init_track_usurf_obs    = init_track_usurf_obs  
igm.update_track_usurf_obs  = update_track_usurf_obs
igm.final_track_usurf_obs   = final_track_usurf_obs
    
