import igm
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

assert (vol<11.5)&(vol>11.0)

modules = [A for A in state.__dict__.keys() if 'tcomp_' in A]

state.tcomp_all = [ np.sum([np.sum(getattr(state,m)) for m in modules]) ]

print(" Computational time "+ str(int(state.tcomp_all[0]))+" sec ")
     