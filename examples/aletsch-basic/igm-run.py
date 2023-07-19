#!/usr/bin/env python3
 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import igm

 
# Define in order the model components step to be updated
modules = [
#    "prepare_data",
    "load_ncdf_data",
#    "optimize",
    "smb_simple",
    "iceflow",
#    "particles",
    "time_step",
    "thk",
#    "topg_glacial_erosion",
    "write_ncdf_ex",
    "write_ncdf_ts",
#   "write_plot2d",
#   "write_particles",
    "print_info",
   "print_all_comp_info",
#   "anim3d_from_ncdf_ex"
]

# Collect and parse all the parameters of all model components
parser = igm.params_core()
for module in modules:
    getattr(igm, "params_" + module)(parser)
params = parser.parse_args()

# Override parameters
# params.RGI = 'RGI60-11.01450'
# params.observation = True
params.tstart = 100.0
params.tend = 200.0
params.tsave = 10 
params.plot_live = True
# params.logging_level     = 'INFO'

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
            getattr(igm, "update_" + module)(params, state)
            
    # Finalize each module in turn
    for module in modules:
        getattr(igm, "final_" + module)(params, state)
