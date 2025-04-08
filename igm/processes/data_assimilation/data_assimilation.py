#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

from .utils import compute_rms_std_optimization
from .optimize_initialize import optimize_initialize
from .optimize_update import optimize_update
from .outputs.output_ncdf import update_ncdf_optimize, output_ncdf_optimize_final
from .outputs.prints import print_costs, save_rms_std
from .outputs.plots import update_plot_inversion, plot_cost_functions

from igm.processes.iceflow.emulate.emulate import update_iceflow_emulator
from igm.processes.iceflow import initialize as iceflow_initialize

def initialize(cfg, state):

    iceflow_initialize(cfg, state) # initialize the iceflow model

    optimize_initialize(cfg, state)

    update_iceflow_emulator(cfg, state, 0) # initialize the emulator
  
    # iterate over the optimization process
    for i in range(cfg.processes.data_assimilation.optimization.nbitmax+1):

        cost = {}

        # one step of data assimilation
        optimize_update(cfg, state, cost, i)

        compute_rms_std_optimization(state, i)
            
        # one step of retraning the iceflow emulator
        if cfg.processes.data_assimilation.optimization.retrain_iceflow_model:
            update_iceflow_emulator(cfg, state, i+1) 
            cost["glen"] = state.COST_EMULATOR[-1]
            
        print_costs(cfg, state, cost, i)

        if i % cfg.processes.data_assimilation.output.freq == 0:
            if cfg.processes.data_assimilation.output.plot2d:
                update_plot_inversion(cfg, state, i)
            if cfg.processes.data_assimilation.output.save_iterat_in_ncdf:
                update_ncdf_optimize(cfg, state, i)

            # stopping criterion: stop if the cost no longer decrease
            # if i>cfg.processes.data_assimilation.optimization.nbitmin:
            #     cost = [c[0] for c in costs]
            #     if np.mean(cost[-10:])>np.mean(cost[-20:-10]):
            #         break;  

    # now that the ice thickness is optimized, we can fix the bed once for all! (ONLY FOR GROUNDED ICE)
    state.topg = state.usurf - state.thk

    if not cfg.processes.data_assimilation.output.save_result_in_ncdf=="":
        output_ncdf_optimize_final(cfg, state)

    plot_cost_functions() # ! Bug right now with plotting values... (extra headers)

    save_rms_std(cfg, state) 

def update(cfg, state):
    pass

def finalize(cfg, state):
    pass