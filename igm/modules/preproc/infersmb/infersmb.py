# Import the most important libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math
 
from igm.modules.process.clim_oggm import *
from igm.modules.process.smb_oggm import *

from igm.modules.process.clim_oggm import params as params_clim_oggm
from igm.modules.process.smb_oggm import params as params_smb_oggm
 
from igm.modules.process.clim_oggm import initialize as initialize_clim_oggm
from igm.modules.process.smb_oggm import initialize as initialize_smb_oggm

from igm.modules.process.clim_oggm import update as update_clim_oggm
from igm.modules.process.smb_oggm import update as update_smb_oggm

import igm
 
def params(parser):  
    params_clim_oggm(parser)
    params_smb_oggm(parser)
 
def initialize(params,state):
    initialize_clim_oggm(params,state)
    initialize_smb_oggm(params,state)

    state.t = tf.Variable(2000.0) 
    
    smb = []
    
    while state.t < 2021:

        update_clim_oggm(params,state)
        update_smb_oggm(params,state)
        smb.append(state.smb)
        state.t = state.t + 1
    
    smb = tf.stack(smb)

    state.smb = tf.reduce_mean(smb,axis=0)
    
    state.smb = tf.where(state.icemaskobs > 0.5, state.smb, 0.0)
    
    del state.t
 

def update(params,state):
    pass
     
def finalize(params,state):
    pass
