#!/usr/bin/env python3

import sys
sys.path.append("/home/gjouvet/IGM/igm2-public/")

# Import the most important libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import igm
 
# Define in order the model components step to be updated
modules = [
#    "prepare_data",
    "load_ncdf_data",
    "optimize_v2",
    "smb_simple",
    "iceflow_v2",
#    "particles_v2",
    "time_step",
    "thk",
#    "topg_glacial_erosion",
    "write_ncdf_ex",
    "write_ncdf_ts",
    "write_plot2d",
#    "write_particles",
    "print_info",
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
if "iceflow_v1" in modules:
    params.iceflow_model_lib_path = "../../emulators/f15_cfsflow_GJ_22_a"
else:
    params.iceflow_model_lib_path = "../../emulators/f21_pinnbp_GJ_23_a"
params.plot_live = True
params.varplot_max = 250
# params.logging_file      = ''
# params.logging_level     = 'INFO'

# Define a state class/dictionnary that contains all the data
state = igm.State(params)

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

# Provide computational statistic of the run
igm.update_print_all_comp_info(params, state)

