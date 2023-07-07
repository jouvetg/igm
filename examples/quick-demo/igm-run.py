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
#    "prepare_data",
    "load_ncdf_data",
    "mysmb",
    "iceflow_v2",
    "time_step",
    "thk",
    "write_ncdf_ex",
    "write_ncdf_ts",
    "write_plot2d",
    "print_info",
]

# Collect and parse all the parameters of all model components
parser = igm.params_core()
for module in modules:
    getattr(igm, "params_" + module)(parser)
params = parser.parse_args()

# Override parameters
params.tstart = 2000.0
params.tend   = 2100.0
params.tsave  = 5
params.plot_live = True
params.RGI = 'RGI60-01.00709'
# params.logging_file      = ''
# params.logging_level     = 'INFO'

# Define a state class/dictionnary that contains all the data
state = igm.State()
igm.init_state(params, state)

# Place the computation on your device GPU ('/GPU:0') or CPU ('/CPU:0')
with tf.device("/GPU:0"):
    
    # Initialize all the model components in turn
    for module in modules:
        getattr(igm, "init_" + module)(params, state)

    # give a mean ela that is suitable for this glacier
    params.meanela = np.quantile(state.usurf[state.thk>10],0.3)

    # Time loop, perform the simulation until reaching the defined end time
    while state.t < params.tend:

        # Update in turn each model components
        for module in modules:
            getattr(igm, "update_" + module)(params, state)

# Provide computational statistic of the run
igm.modules.utils.print_all_comp_info(params, state)

igm.modules.utils.anim_3d_from_ncdf_ex(params)
