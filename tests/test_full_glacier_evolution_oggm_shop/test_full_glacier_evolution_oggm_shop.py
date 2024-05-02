import igm
import tensorflow as tf
import numpy as np
import pytest

def test_full_glacier_evolution_oggm_shop():
    
    param_file = "./test_full_glacier_evolution_oggm_shop/params.json"
    
    state = igm.State()
    parser = igm.params_core()

    modules_dict = igm.get_modules_list(param_file)

    modules = igm.load_modules(modules_dict)
    
######### The fowllowing should be rather params = igm.setup_igm_params(parser, modules)
 
    for module in modules:
        module.params(parser)
    params,_ = parser.parse_known_args()
    params = igm.load_user_defined_params( param_file=param_file, params_dict=vars(params) )
    parser.set_defaults(**params) 
    
#################################
    
    params, __ = parser.parse_known_args() 
    
    with tf.device(f"/GPU:{params.gpu_id}"):
        igm.run_intializers(modules, params, state)
        igm.run_processes(modules, params, state)
        igm.run_finalizers(modules, params, state)
        
    vol = np.sum(state.thk) * (state.dx**2) / 10**9

    assert (vol<2.9)&(vol>.75)

    modules = [A for A in state.__dict__.keys() if 'tcomp_' in A]

    state.tcomp_all = [ np.sum([np.sum(getattr(state,m)) for m in modules]) ]

    print(" Computational time "+ str(int(state.tcomp_all[0]))+" sec ")
        
        