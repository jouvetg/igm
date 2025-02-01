#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
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
import time
from netCDF4 import Dataset


def params(parser):
    pass

def initialize(cfg,state):
    
    nc = Dataset(os.path.join("data", 'past_surf.nc'), "r" )
#    for y in [1880,1926,1957,1980,1999,2009,2017]:
#        vars(state)['surf_'+str(y)] = np.squeeze( nc.variables['surf_'+str(y)] ).astype("float32") 
    for v in nc.variables:
        if v not in ['x','y']:
            vars(state)[v] = tf.Variable(np.squeeze( nc.variables[v] ).astype("float32"))
    nc.close()

 
    # load the surface toporgaphy available at given year
    if "time" not in cfg.modules:
        raise ValueError("The 'time' module is required for the 'clim_aletsch' module.")
    
    state.usurf = vars(state)['surf_'+str(int(cfg.modules.time.time_start))]
    state.thk   = state.usurf -state.topg

def update(cfg,state):

    if state.t in [1880,1926,1957,1980,1999,2009,2017]:

        diff = (state.usurf-vars(state)['surf_'+str(int(state.t))]).numpy()
        diff = diff[state.thk>1]
        mean  = np.mean(diff)
        std   = np.std(diff)
        vol   = np.sum(state.thk) * (state.dx ** 2) / 10 ** 9
        print(" Check modelled vs observed surface at time : %8.0f ; Mean discr. : %8.2f  ;  Std : %8.2f |  Ice volume : %8.2f " \
                % (state.t, mean, std, vol) )


def finalize(cfg,state):
    pass
