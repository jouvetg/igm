#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
 Quick notes about the code below:
 
 The goal of this module is to compute the ice flow velocity field
 using a deep-learning emulator of the Blatter-Pattyn model.
  
 The aim of this module is
   - to initialize the ice flow and its emulator in init_iceflow
   - to update the ice flow and its emulator in update_iceflow

In update_iceflow, we compute/update with function _update_iceflow_emulated,
and retraine the iceflow emaultor in function _update_iceflow_emulator

- in _update_iceflow_emulated, we baiscially gather together all input fields
of the emulator and stack all in a single tensor X, then we compute the output
with Y = iceflow_model(X), and finally we split Y into U and V

- in _update_iceflow_emulator, we retrain the emulator. For that purpose, we
iteratively (usually we do only one iteration) compute the output of the emulator,
compute the energy associated with the state of the emulator, and compute the
gradient of the energy with respect to the emulator parameters. Then we update
the emulator parameters with the gradient descent method (Adam optimizer).
Because this step may be memory consuming, we split the computation in several
patches of size cfg.processes.iceflow.iceflow.retrain_emulator_framesizemax. This permits to
retrain the emulator on large size arrays.

Alternatively, one can solve the Blatter-Pattyn model using a solver using 
function _update_iceflow_solved. Doing so is not very different to retrain the
emulator as we minmize the same energy, however, with different controls,
namely directly the velocity field U and V instead of the emulator parameters.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import datetime, time
import math
import tensorflow as tf

from igm.processes.utils import *

from .params_pretraining import *
from .params_optimize import *
from .params_iceflow import *
from .emulate import *
from .solve import *
from .diagnostic import *
from .utils import *
from .optimize import *
from .pretraining import *

# def params(parser):
#     pass

#     params_pretraining(parser)
#     params_optimize(parser)
#     params_iceflow(parser)

def initialize(cfg, state):

    # This makes it so that if the user included the optimize module, this intializer will not be called again.
    # This is due to the fact that the optimize module calls the initialize (and params) function of the iceflow module.
    if hasattr(state, "optimize_initializer_called"):
        return

    if cfg.processes.iceflow.iceflow.run_pretraining:
        pretraining(cfg, state)

    # deinfe the fields of the ice flow such a U, V, but also sliding coefficient, arrhenius, ectt
    initialize_iceflow_fields(cfg, state)

    if cfg.processes.iceflow.iceflow.type == "emulated":
        # define the emulator, and the optimizer
        initialize_iceflow_emulator(cfg, state)
    elif cfg.processes.iceflow.iceflow.type == "solved":
        # define the solver, and the optimizer
        initialize_iceflow_solver(cfg, state)    
    elif cfg.processes.iceflow.iceflow.type == "diagnostic":
        # define the second velocity field
        initialize_iceflow_diagnostic(cfg,state)

    # create the vertica discretization
    define_vertical_weight(cfg, state)
    
    # padding is necessary when using U-net emulator
    state.PAD = compute_PAD(cfg, state.thk.shape[1],state.thk.shape[0])
    
    if not cfg.processes.iceflow.iceflow.type == "solved":
        update_iceflow_emulated(cfg, state)
        
        
    # Currently it is not supported to have the two working simulatanoutly
    assert (cfg.processes.iceflow.iceflow.exclude_borders==0) | (cfg.processes.iceflow.iceflow.multiple_window_size==0)

    if cfg.processes.iceflow.iceflow.run_data_assimilation:
        state.it = -1
        update_iceflow_emulator(cfg, state)
        optimize(cfg, state)
        

def update(cfg, state):

    if hasattr(state, "logger"):
        state.logger.info("Update ICEFLOW at time : " + str(state.t.numpy()))

    if cfg.processes.iceflow.iceflow.type == "emulated":
        if cfg.processes.iceflow.iceflow.retrain_emulator_freq > 0:
            update_iceflow_emulator(cfg, state)

        update_iceflow_emulated(cfg, state)

    elif cfg.processes.iceflow.iflo_type == "solved":
        update_iceflow_solved(cfg, state)

    elif cfg.processes.iceflow.iflo_type == "diagnostic":
        update_iceflow_diagnostic(cfg, state)

def finalize(cfg, state):

    if cfg.processes.iceflow.iceflow.save_model:
        save_iceflow_model(cfg, state)
   
 
  