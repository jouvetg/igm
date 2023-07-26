#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import igm

from clim_aletsch import *
from smb_accmelt import *
from seeding import *
from track_usurf_obs import *

# Select one OPTION btw the first, keep the MANDATORY ones, un/comment OPTIONAL modules
modules_preproc =   [ 
            "load_ncdf_data",
#            "optimize", 
            "track_usurf_obs"
                    ]

modules_physics =   [
            "clim_aletsch",
            "smb_accmelt",
            "flow_dt_thk",          # MANDATORY : update the ice thickness solving mass cons.
            "rockflow",
            "vertical_iceflow",     # OPTIONAL  : retrieve vertical ice flow from horiz.
#            "particles",            # OPTIONAL  : seed and update particle trajectories
                   ]

modules_postproc = [
#           "write_particles",      # OPTIONAL  : write particle trajectories to a csv file
            "write_ncdf_ex",        # OPTIONAL  : write 2d state data to netcdf files
            "write_plot2d",         # OPTIONAL  : write 2d state plots to png files
            "print_info",           # OPTIONAL  : print basic live-info about the model state
            "print_all_comp_info",  # OPTIONAL  : report information about computation time
#           "anim3d_from_ncdf_ex"  # OPTIONAL  : make a nice 3D animation of glacier evolution
                   ]


modules = modules_preproc + modules_physics + modules_postproc

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

params.geology_file = 'geology.nc' # this permits to skip the "optimize" step optimized

# Define a state class/dictionnary that contains all the data
state = igm.State()

# igm.print_params(params) # uncomment this if you wish to record used params
# igm.add_logger(params, state, logging_file="") # uncoment if you wish to log

# Place the computation on your device GPU ('/GPU:0') or CPU ('/CPU:0')
with tf.device("/GPU:0"):

    # Initialize all the model components in turn
    for module in modules:
        getattr(igm, "init_" + module)(params, state)

    if not modules_physics==[]:
        # Time loop, perform the simulation until reaching the defined end time
        while state.t < params.tend:
            # Update each model components in turn
            for module in modules:
                if (not (module=='particles'))|(state.t>1900):
                    getattr(igm, "update_" + module)(params, state)
            
    # Finalize each module in turn
    for module in modules:
        getattr(igm, "final_" + module)(params, state)
 
