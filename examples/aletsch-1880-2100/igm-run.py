#!/usr/bin/env python3

import sys
sys.path.append("/home/gjouvet/IGM2/igm2-public/")

# Import the most important libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import igm

from clim_aletsch import *
from smb_accmelt import *
from seeding import *
from track_usurf_obs import *
 
# Define in order the model components step to be updated
modules = [
           "load_ncdf_data",
           "optimize",
           "track_usurf_obs",
           "clim_aletsch",
           "smb_accmelt", 
           "flow_dt_thk", 
           "rockflow",
           "vertical_iceflow",
           "particles",
           "write_ncdf_ex", 
           "write_particles",
           "write_plot2d", 
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
params.tstart = 1880.0
params.tend   = 2020.0
params.tsave  = 5
params.plot_live    = True

# params.editor_plot2d = 'sp'

params.init_slidingco  = 5000

params.weight_accumulation   = 1.0
params.weight_ablation       = 1.25

params.opti_nbitmax          = 500
params.opti_control          = ["thk"] 
params.opti_cost             = ["velsurf", "thk", "icemask"] 
params.opti_convexity_weight = 0
params.opti_regu_param_thk   = 10

params.frequency_seeding     = 500
params.tracking_method       = 'simple'
params.density_seeding       = 1

# params.geology_file = 'geology-optimized.nc' # this permits to skip the "optimize" step

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
            if (not (module=='particles'))|(state.t>1900):
                getattr(igm, "update_" + module)(params, state)
            
    # Finalize each module in turn
    for module in modules:
        getattr(igm, "final_" + module)(params, state)
 