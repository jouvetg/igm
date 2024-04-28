import igm
import tensorflow as tf
import pytest

def test_vert_flow():
    
    state = igm.State()
    modules_dict = {'modules_preproc': [], 'modules_process': ["iceflow","vert_flow"], 'modules_postproc': []}
    
    modules = igm.load_modules(modules_dict)

    parser = igm.params_core()

    for module in modules:
        module.params(parser)

    params, __ = parser.parse_known_args()

    Nz,Ny,Nx = 10,40,30

    state.thk   = tf.Variable(tf.ones((Ny,Nx))*200)
    state.topg  = tf.Variable(tf.zeros((Ny,Nx)))
    state.usurf = state.thk + state.topg
    state.dX    = tf.Variable(tf.ones((Ny,Nx))*100) 
    state.dx    = 100
    state.it    = -1
    
    for module in modules:
        module.initialize(params, state)

    for module in modules:
        module.update(params, state)

    for module in modules:
        module.finalize(params, state)
        
    assert (tf.reduce_mean(state.W).numpy()<10*10)
