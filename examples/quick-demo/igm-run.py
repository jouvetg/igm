#!/usr/bin/env python3

import sys
sys.path.append("/home/gjouvet/IGM/igm2-public/")

# Import the most important libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import igm
import math
 
# Define in order the model components step to be updated
modules = [
    "prepare_data",
    "load_ncdf_data",
    "mysmb",
    "iceflow_v1",
    "time_step",
    "thk",
    "write_ncdf_ex",
    "write_ncdf_ts",
#    "write_plot2d",
    "print_info",
]

## add custumized smb function
def params_mysmb(parser):  
    parser.add_argument("--meanela", type=float, default=3000 )

def init_mysmb(params,state):
    pass

def update_mysmb(params,state):
    # perturabe the ELA with sinusional signal 
    ELA = ( params.meanela + 750*math.sin((state.t/50)*math.pi) )
    # compute smb linear with elevation with 2 acc & abl gradients
    state.smb  = state.usurf - ELA
    state.smb *= tf.where(state.smb<0, 0.005, 0.009)
    # cap smb by 2 m/y 
    state.smb  = tf.clip_by_value(state.smb, -100, 2)
    # make sure the smb is not positive outside of the mask to prevent overflow
    state.smb  = tf.where((state.smb<0)|(state.icemask>0.5),state.smb,-10)

# make sure to make these function new attributes of the igm module
igm.params_mysmb = params_mysmb  
igm.init_mysmb   = init_mysmb  
igm.update_mysmb = update_mysmb

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
params.varplot_max = 250
params.RGI = 'RGI60-01.00709'
# params.logging_file      = ''
# params.logging_level     = 'INFO'

# Define a state class/dictionnary that contains all the data
state = igm.State(params)

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
igm.update_print_all_comp_info(params, state)

from igm.modules.utils import anim3d_from_netcdf
igm.anim3d_from_netcdf()
