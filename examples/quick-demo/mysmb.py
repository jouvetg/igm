# Import the most important libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import igm
import math

 ## add custumized smb function
def params_mysmb(parser):  
    parser.add_argument("--meanela", type=float, default=3000 )

def init_mysmb(params,state):
    params.meanela = np.quantile(state.usurf[state.thk>10],0.3)

def update_mysmb(params,state):
    # perturabe the ELA with sinusional signal 
    ELA = ( params.meanela + 750*math.sin((state.t/100)*math.pi) )
    # compute smb linear with elevation with 2 acc & abl gradients
    state.smb  = state.usurf - ELA
    state.smb *= tf.where(state.smb<0, 0.005, 0.009)
    # cap smb by 2 m/y 
    state.smb  = tf.clip_by_value(state.smb, -100, 2)
    # make sure the smb is not positive outside of the mask to prevent overflow
    state.smb  = tf.where((state.smb<0)|(state.icemask>0.5),state.smb,-10)
    
def final_mysmb(params,state):
    pass

# make sure to make these function new attributes of the igm module
igm.params_mysmb  = params_mysmb  
igm.init_mysmb    = init_mysmb  
igm.update_mysmb  = update_mysmb
igm.final_mysmb   = final_mysmb