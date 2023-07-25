#!/usr/bin/env python3
 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import igm

from mysmb import *
 
# Select one OPTION btw the first, keep the MANDATORY ones, un/comment OPTIONAL modules
modules = [
           "prepare_data",          # OPTION 1  : download and prepare the data with OGGM
#           "load_ncdf_data",        # OPTION 2  : read 2d data from netcdf files
#           "load_tif_data",        # OPTION 3  : read 2d data from tif files
#           "make_synthetic",       # OPTION 4  : make a synthetic glacier with ideal geom.
#           "optimize",             # OPTIONAL  : optimize unobservable variables from obs.
            "mysmb",                # OPTIONAL  : custom surface mass balance model
            "flow_dt_thk",          # MANDATORY : update the ice thickness solving mass cons.
            "rockflow",
            "vertical_iceflow",     # OPTIONAL  : retrieve vertical ice flow from horiz.
#           "particles",            # OPTIONAL  : seed and update particle trajectories
#           "write_particles",      # OPTIONAL  : write particle trajectories to a csv file
#           "topg_glacier_erosion", # OPTIONAL  : update the bedrock elevation from erosion
            "write_ncdf_ex",        # OPTIONAL  : write 2d state data to netcdf files
#           "write_ncdf_ts",        # OPTIONAL  : write the result in tif files
#           "write_ncdf_ts",        # OPTIONAL  : write time serie data to netcdf files
#           "write_plot2d",         # OPTIONAL  : write 2d state plots to png files
            "print_info",           # OPTIONAL  : print basic live-info about the model state
            "print_all_comp_info",  # OPTIONAL  : report information about computation time
            "anim3d_from_ncdf_ex"   # OPTIONAL  : make a nice 3D animation of glacier evolution
#           "anim_mp4_from_ncdf_ex" # OPTIONAL  : make a animated mp4 movie of glacier evolution
          ]

# Collect and parse all the parameters of all model components
parser = igm.params_core()
for module in modules:
    getattr(igm, "params_" + module)(parser)
params = parser.parse_args(args=[])

# Override parameters
params.RGI = 'RGI60-11.03646'
params.tstart = 2000.0
params.tend   = 2100.0
params.tsave  = 5
params.plot_live = True
params.observation = False

# parameters for using the 'optimize' module, params.observation must be True
params.opti_nbitmax = 500
params.opti_control = ["thk"] # , "usurf" # "slidingco"
params.opti_cost    = ["velsurf", "thk", "icemask"] # ,"usurf" "divfluxfcz",
params.opti_regu_param_thk        = 5
# params.opti_step_size           = 1
params.opti_regu_param_slidingco  = 10   # weight for the regul. of slidingco
params.opti_convexity_weight      = 0

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
