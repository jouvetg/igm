#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

from .utils import compute_rms_std_optimization
from .initialize import initialize_optimize
from .update import optimize_update
from .outputs.output_ncdf import update_ncdf_optimize, output_ncdf_optimize_final
from .outputs.prints import print_costs, save_rms_std
from .outputs.plots import update_plot_inversion, plot_cost_functions

from ..emulate.emulate import update_iceflow_emulator

def optimize(cfg, state):

    initialize_optimize(cfg, state)
  
    # iterate over the optimization process
    for i in range(cfg.processes.iceflow.optimize.nbitmax+1):

        cost = {}

        # one step of data assimilation
        optimize_update(cfg, state, cost, i)

        compute_rms_std_optimization(state, i)
            
        # one step of retraning the iceflow emulator
        if cfg.processes.iceflow.optimize.retrain_iceflow_model:
            update_iceflow_emulator(cfg, state, i+1) 
            cost["glen"] = state.COST_EMULATOR[-1]
            
        print_costs(cfg, state, cost, i)

        if i % cfg.processes.iceflow.optimize.output_freq == 0:
            if cfg.processes.iceflow.optimize.plot2d:
                update_plot_inversion(cfg, state, i)
            if cfg.processes.iceflow.optimize.save_iterat_in_ncdf:
                update_ncdf_optimize(cfg, state, i)

            # stopping criterion: stop if the cost no longer decrease
            # if i>cfg.processes.iceflow.optimize_nbitmin:
            #     cost = [c[0] for c in costs]
            #     if np.mean(cost[-10:])>np.mean(cost[-20:-10]):
            #         break;  

    # now that the ice thickness is optimized, we can fix the bed once for all! (ONLY FOR GROUNDED ICE)
    state.topg = state.usurf - state.thk

    if not cfg.processes.iceflow.optimize.save_result_in_ncdf=="":
        output_ncdf_optimize_final(cfg, state)

    plot_cost_functions() # ! Bug right now with plotting values... (extra headers)

    save_rms_std(cfg, state)

    # Flag so we can check if initialize was already called
    state.optimize_initializer_called = True
 
