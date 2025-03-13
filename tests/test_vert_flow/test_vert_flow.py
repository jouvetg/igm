import igm, os
import tensorflow as tf
import pytest

def test_vert_flow():
    
    state = igm.State()

    cfg = igm.EmptyClass()  
    cfg.processes = igm.EmptyClass()  
    cfg.processes.iceflow  = igm.load_yaml_as_cfg(os.path.join("conf","processes","iceflow.yaml")).iceflow
    cfg.processes.vert_flow = igm.load_yaml_as_cfg(os.path.join("conf","processes","vert_flow.yaml")).vert_flow
 
    Nz,Ny,Nx = 10,40,30

    state.thk   = tf.Variable(tf.ones((Ny,Nx))*200)
    state.topg  = tf.Variable(tf.zeros((Ny,Nx)))
    state.usurf = state.thk + state.topg
    state.dX    = tf.Variable(tf.ones((Ny,Nx))*100) 
    state.dx    = 100
    state.it    = -1
    
    igm.processes.iceflow.initialize(cfg, state)
    igm.processes.vert_flow.initialize(cfg, state)

    igm.processes.iceflow.update(cfg, state)
    igm.processes.vert_flow.update(cfg, state)

    igm.processes.iceflow.finalize(cfg, state)
    igm.processes.vert_flow.finalize(cfg, state)
     
    assert (tf.reduce_mean(state.W).numpy()<10*10)
