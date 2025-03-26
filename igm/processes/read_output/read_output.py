#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os 
import tensorflow as tf
from netCDF4 import Dataset
 
from igm.processes.utils import *
 
def initialize(cfg, state):
 
    nc = Dataset(cfg.processes.read_output.input_file, "r")

    state.x    = np.squeeze(nc.variables["x"]).astype("float32")
    state.y    = np.squeeze(nc.variables["y"]).astype("float32")
    state.time = np.squeeze(nc.variables["time"]).astype("float32")

    state.list_var = [l for l in nc.variables if not l in ["x", "y", "z", "time"]]

    for var in state.list_var:
        vars(state)[var+'_'+'t'] = np.squeeze(nc.variables[var]).astype("float32")
 
    # crop if requested
    if cfg.processes.read_output.crop:
        i0 = max(0, int((cfg.processes.read_output.xmin - x[0]) / (x[1] - x[0])))
        i1 = min(int((cfg.processes.read_output.xmax - x[0]) / (x[1] - x[0])), x.shape[0] - 1)
        i1 = max(i0 + 1, i1)
        j0 = max(0, int((cfg.processes.read_output.ymin - y[0]) / (y[1] - y[0])))
        j1 = min(int((cfg.processes.read_output.ymax - y[0]) / (y[1] - y[0])), y.shape[0] - 1)
        j1 = max(j0 + 1, j1) 
        for var in state.list_var:
            vars(state)[var+'_'+'t'] = vars(state)[var+'_'+'t'][j0:j1, i0:i1]
        y = y[j0:j1]
        x = x[i0:i1]
        
    for var in ["x", "y"]:
        vars(state)[var] = tf.constant(vars(state)[var].astype("float32"))

    nc.close()
    
    cfg.processes.time.start = state.time[0]
    cfg.processes.time.end   = state.time[-1]
     
    state.t = tf.Variable(float(cfg.processes.time.start))
    state.it = 0
    state.saveresult = True

def update(cfg, state):
    
    state.t.assign(state.time[state.it])
     
    for var in state.list_var:
        vars(state)[var] = tf.Variable(vars(state)[var+'_'+'t'][state.it]) 

    state.it += 1

def finalize(cfg, state):
    pass
