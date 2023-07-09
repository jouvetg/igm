#!/usr/bin/env python3

import sys
sys.path.append("/home/gjouvet/IGM/igm2-public/")

# Import the most important libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import igm

from mysmb import *
 
# Define in order the model components step to be updated
modules = [
           "prepare_data",
           "optimize",
#           "load_ncdf_data",
           "mysmb", 
           "iceflow", 
           "vertical_iceflow",
           "time_step",
#           "particles",
           "thk", 
           "write_ncdf_ex", 
#           "write_particles",
#           "write_plot2d", 
           "print_info",
           "print_all_comp_info",
           "anim3d_from_ncdf_ex"
          ]

# Collect and parse all the parameters of all model components
parser = igm.params_core()
for module in modules:
    getattr(igm, "params_" + module)(parser)
params = parser.parse_args()

# Override parameters
params.tstart = 2000.0
params.tend   = 2100.0
params.tsave  = 2
params.plot_live = True
params.RGI = 'RGI60-11.01450' 
params.observation = True

# Define a state class/dictionnary that contains all the data
state = igm.State()
igm.init_state(params, state)

# Place the computation on your device GPU ('/GPU:0') or CPU ('/CPU:0')
with tf.device("/GPU:0"):

    # Initialize all the model components in turn
    for module in modules:
        getattr(igm, "init_" + module)(params, state)

    # Time loop, perform the simulation until reaching the defined end time
    while state.t < params.tend:
        
        # Update each model components in turn
        for module in modules:
#            if (not (module=='particles_v2'))|(state.t>2300):
            getattr(igm, "update_" + module)(params, state)
            
    # Finalize each module in turn
    for module in modules:
        getattr(igm, "final_" + module)(params, state)

