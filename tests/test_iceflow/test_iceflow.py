import igm
import tensorflow as tf
import pytest
 
def test_iceflow():
    
    state = igm.State()
    modules_dict = {'modules_preproc': [], 'modules_process': ['iceflow'], 'modules_postproc': []}

    print(modules_dict)
    
    imported_modules = igm.load_modules(modules_dict)

    print(imported_modules)

    module = imported_modules[0] 

    parser = igm.params_core()

    module.params(parser)

    params, __ = parser.parse_known_args()

    params.iflo_network = 'cnn'

    Ny,Nx = 40,30

    state.thk   = tf.Variable(tf.zeros((Ny,Nx)))
    state.usurf = tf.Variable(tf.zeros((Ny,Nx)))
    state.dX    = tf.Variable(tf.ones((Ny,Nx))*100)
    state.it    = -1
    
    module.initialize(params, state)

    module.update(params, state)

    module.finalize(params, state)

    assert (tf.reduce_mean(state.ubar).numpy()<10*10)
