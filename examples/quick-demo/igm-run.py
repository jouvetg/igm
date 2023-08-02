#!/usr/bin/env python3
 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import igm

from mysmb import *
 
# Select one OPTION btw the first, keep the MANDATORY ones, un/comment OPTIONAL modules
modules_preproc =  [
            "prepare_data",         # OPTION 1  : download and prepare the data with OGGM
#            "load_ncdf_data",       # OPTION 2  : read 2d data from netcdf files
#           "load_tif_data",        # OPTION 3  : read 2d data from tif files
#           "make_synthetic",       # OPTION 4  : make a synthetic glacier with ideal geom.
#            "optimize",             # OPTIONAL  : optimize unobservable variables from obs.
                   ]

modules_process =  [
            "mysmb",                # OPTIONAL  : custom surface mass balance model
            "flow_dt_thk",          # MANDATORY : update the ice thickness solving mass cons.
            "vertical_iceflow",     # OPTIONAL  : retrieve vertical ice flow from horiz.
            "particles",            # OPTIONAL  : seed and update particle trajectories
                   ]

modules_postproc = [
#           "write_particles",      # OPTIONAL  : write particle trajectories to a csv file
            "write_ncdf_ex",        # OPTIONAL  : write 2d state data to netcdf files
#           "write_tif_ex",         # OPTIONAL  : write the result in tif files
#           "write_ncdf_ts",        # OPTIONAL  : write time serie data to netcdf files
            "write_plot2d",         # OPTIONAL  : write 2d state plots to png files
            "print_info",           # OPTIONAL  : print basic live-info about the model state
            "print_all_comp_info",  # OPTIONAL  : report information about computation time
#            "anim3d_from_ncdf_ex"  # OPTIONAL  : make a nice 3D animation of glacier evolution
                   ]

modules = modules_preproc + modules_process + modules_postproc

# Collect and parse all the parameters of all model components
parser = igm.params_core()
for module in modules:
    getattr(igm, "params_" + module)(parser)
params = parser.parse_args(args=[])

# Override parameters
params.RGI         = 'RGI60-11.01238'
params.tstart      = 2000.0
params.tend        = 2050.0
params.tsave       = 5
params.plot_live   = True
params.include_glathida = True

# Define a state class/dictionnary that contains all the data
state = igm.State()

# igm.print_params(params) # uncomment this if you wish to record used params
# igm.add_logger(params, state, logging_file="") # uncoment if you wish to log

# Place the computation on your device GPU ('/GPU:0') or CPU ('/CPU:0')
with tf.device("/GPU:0"):

    # Initialize all the model components in turn
    for module in modules:
        getattr(igm, "init_" + module)(params, state)

    # Time loop, perform the simulation until reaching the defined end time
    if not modules_process==[]:
        while state.t < params.tend:
            # Update each model components in turn
            for module in modules:
                getattr(igm, "update_" + module)(params, state)
            
    # Finalize each module in turn
    for module in modules:
        getattr(igm, "final_" + module)(params, state)
