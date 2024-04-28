import igm
import os
import tensorflow as tf
import numpy as np
import pytest
 
state = igm.State()
parser = igm.params_core()

modules_dict = igm.get_modules_list("./params.json")
 
modules = igm.load_modules(modules_dict)
 
params = igm.setup_igm_params(parser, modules)
 
params, __ = parser.parse_known_args() 
 
with tf.device(f"/GPU:{params.gpu_id}"):
    igm.run_intializers(modules, params, state)
    igm.run_processes(modules, params, state)
    igm.run_finalizers(modules, params, state)
    
vol = np.sum(state.thk) * (state.dx**2) / 10**9

assert (vol<2.9)&(vol>.75)
    
for f in ['optimize.nc','convergence.png','rms_std.dat','costs.dat','clean.sh']:
    if os.path.exists(f):
        os.remove(f)