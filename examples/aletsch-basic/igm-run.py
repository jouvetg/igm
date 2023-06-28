#!/usr/bin/env python3

import sys

sys.path.append("/home/gjouvet/IGM/igm2-public/")

# Import the most important libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import igm

# Define in order the model components step to be updated
steps = [
    #          'prepare_data',
    "load_ncdf_data",
    "smb_simple",
    "iceflow_v1",
    "time_step",
    "thk",
    "ncdf_ex",
    "ncdf_ts",
    "plot_vs",
    "print_info",
]

# def update_smb(params,state):
#     state.smb  = state.usurf - 2000
#     state.smb *= tf.where(tf.less(state.smb, 0), 0.006, 0.009)
#     state.smb  = tf.clip_by_value(state.smb, -100, 2)
#     if hasattr(state, "icemask"):
#         state.smb = tf.where(state.icemask > 0.5, state.smb, -10)

# igm.update_smb = update_smb

# Collect and parse all the parameters of all model components
parser = igm.params_core()
for step in [s for s in steps if hasattr(igm, "params_" + s)]:
    getattr(igm, "params_" + step)(parser)
params = parser.parse_args()

# Override parameters
params.tstart = 100.0
params.tend = 200.0
params.tsave = 10
params.iceflow_model_lib_path = "../../model-lib/f15_cfsflow_GJ_22_a"
params.plot_live = True
params.varplot_max = 250
# params.logging_file      = ''
# params.logging_level     = 'INFO'

# Define a state class/dictionnary that contains all the data
state = igm.State(params)

# Place the computation on your device GPU ('/GPU:0') or CPU ('/CPU:0')
with tf.device("/GPU:0"):
    # Initialize all the model components in turn
    for step in [s for s in steps if hasattr(igm, "init_" + s)]:
        getattr(igm, "init_" + step)(params, state)

    # Time loop, perform the simulation until reaching the defined end time
    while state.t < params.tend:
        # Update in turn each model components
        for step in [s for s in steps if hasattr(igm, "update_" + s)]:
            getattr(igm, "update_" + step)(params, state)

# Provide computational statistic of the run
igm.update_print_all_comp_info(params, state)
