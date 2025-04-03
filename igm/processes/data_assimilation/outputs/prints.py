#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import tensorflow as tf
import datetime
import os

def print_costs(cfg, state, cost, i):

    vol = ( np.sum(state.thk) * (state.dx**2) / 10**9 ).numpy()
    # mean_slidingco = tf.math.reduce_mean(state.slidingco[state.icemaskobs > 0.5])

    f = open('costs.dat','a')

    def bound(x):
        return min(x, 9999999)

    keys = list(cost.keys()) 
    if i == 0:
        L = [f"{key:>8}" for key in ["it","vol"]] + [f"{key:>12}" for key in keys]
        print("Costs:     " + "   ".join(L))
        print("   ".join([f"{key:>12}" for key in keys]),file=f)

    if i % cfg.processes.data_assimilation.output.freq == 0:
        L = [datetime.datetime.now().strftime("%H:%M:%S"),f"{i:0>{8}}",f"{vol:>8.4f}"] \
          + [f"{bound(cost[key].numpy()):>12.4f}" for key in keys]
        print("   ".join(L))

    print("   ".join([f"{bound(cost[key].numpy()):>12.4f}" for key in keys]),file=f)

def save_rms_std(cfg, state):

    np.savetxt(
        "rms_std.dat",
        np.stack(
            [
                state.rmsthk,
                state.stdthk,
                state.rmsvel,
                state.stdvel,
                state.rmsdiv,
                state.stddiv,
                state.rmsusurf,
                state.stdusurf,
            ],
            axis=-1,
        ),
        fmt="%.10f",
        header="        rmsthk      stdthk       rmsvel       stdvel       rmsdiv       stddiv       rmsusurf       stdusurf",
    )
