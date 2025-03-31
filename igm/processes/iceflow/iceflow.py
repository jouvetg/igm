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

from .emulate.emulate import *
from .solve.solve import *
from .diagnostic.diagnostic import *
from .utils import * 

def initialize(cfg, state):

    # This makes sure this function is only called once
    if hasattr(state, "was_initialize_iceflow_already_called"):
        return

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
 
    # This makes sure this function is only called once
    state.was_initialize_iceflow_already_called = True

def update(cfg, state):

    if hasattr(state, "logger"):
        state.logger.info("Update ICEFLOW at time : " + str(state.t.numpy()))

    if cfg.processes.iceflow.iceflow.type == "emulated":
        if cfg.processes.iceflow.iceflow.retrain_emulator_freq > 0:
            update_iceflow_emulator(cfg, state, state.it)

        update_iceflow_emulated(cfg, state)

    elif cfg.processes.iceflow.iceflow.type == "solved":
        update_iceflow_solved(cfg, state)

    elif cfg.processes.iceflow.iceflow.type == "diagnostic":
        update_iceflow_diagnostic(cfg, state)

def finalize(cfg, state):

    if cfg.processes.iceflow.iceflow.save_model:
        save_iceflow_model(cfg, state)
   
 
  