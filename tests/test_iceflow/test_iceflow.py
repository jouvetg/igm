import igm, os
import tensorflow as tf
import pytest
 
def test_iceflow():
    
    state = igm.State()
    
    cfg = igm.EmptyClass()  
    cfg.processes  = igm.load_yaml_as_cfg(os.path.join("conf","processes","iceflow.yaml"))

    Ny,Nx = 40,30

    state.thk   = tf.Variable(tf.zeros((Ny,Nx)))
    state.usurf = tf.Variable(tf.zeros((Ny,Nx)))
    state.dX    = tf.Variable(tf.ones((Ny,Nx))*100)
    state.it    = -1
    
    igm.processes.iceflow.initialize(cfg, state)

    igm.processes.iceflow.update(cfg, state)

    igm.processes.iceflow.finalize(cfg, state)

    assert (tf.reduce_mean(state.ubar).numpy()<10*10)
