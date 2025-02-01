# Import the most important libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math

 ## add custumized smb function
def params(parser):  
    parser.add_argument("--meanela", type=float, default=3000 )

def initialize(cfg,state):
    state.meanela = np.quantile(state.usurf[state.thk>10],0.2)
    # cfg.modules.mysmb.meanela = np.quantile(state.usurf[state.thk>10],0.2) # cfg does not seem to work here.. which is okay as maybe we want to keep the config separate...

def update(cfg,state):
    # perturabe the ELA with sinusional signal 
    ELA = (state.meanela - 750*math.sin((state.t/100)*math.pi) )
    # compute smb linear with elevation with 2 acc & abl gradients
    state.smb  = state.usurf - ELA
    state.smb *= tf.where(state.smb<0, 0.005, 0.009)
    # cap smb by 2 m/y 
    state.smb  = tf.clip_by_value(state.smb, -100, 2)
    # make sure the smb is not positive outside of the mask to prevent overflow
    if hasattr(state, "icemask"):
        state.smb  = tf.where((state.smb<0)|(state.icemask>0.5),state.smb,-10)
    
def finalize(cfg,state):
    pass
