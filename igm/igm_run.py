#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file 
 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import igm

def main():
    
    
    # Collect defaults, overide from json file, and parse all core parameters 
    parser = igm.params_core()
    igm.overide_from_json_file(parser,check_if_params_exist=False)
    params, __ = parser.parse_known_args() # args=[] add this for jupyter notebook

    # get the list of all modules in order
    modules = params.modules_preproc + params.modules_process + params.modules_postproc

    # load custom modules from file (must be called my_module_name.py) to igm
    for module in modules:
        igm.load_custom_module(params, module)

    # get the list of all dependent modules, which parameters must be called too
    dependent_modules = igm.find_dependent_modules(modules)
    
    # Collect defaults, overide from json file, and parse all specific module parameters 
    for module in modules + dependent_modules:
        getattr(igm, "params_" + module)(parser)
    igm.overide_from_json_file(parser,check_if_params_exist=True)
    params = parser.parse_args() # args=[] add this for jupyter notebook

    # print definive parameters in a file for record
    if params.print_params:
        igm.print_params(params)
    
    # Define a state class/dictionnary that contains all the data
    state = igm.State()

    # if logging is activated, add a logger to the state
    if params.logging:
        igm.add_logger(params, state)
        state.logger.info("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Place the computation on your device GPU ('/GPU:0') or CPU ('/CPU:0')
    with tf.device("/GPU:0"):

        # Initialize all the model components in turn
        for module in modules:
            getattr(igm, "initialize_" + module)(params, state)

        # Time loop, perform the simulation until reaching the defined end time
        if hasattr(state, "t"):
            while state.t < params.time_end:
                # Update each model components in turn
                for module in modules:
                    getattr(igm, "update_" + module)(params, state)
                
        # Finalize each module in turn
        for module in modules:
            getattr(igm, "finalize_" + module)(params, state)

if __name__ == '__main__':
    main()
