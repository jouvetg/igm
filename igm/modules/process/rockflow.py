#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file 

import numpy as np
import os, sys, shutil
import matplotlib.pyplot as plt
import tensorflow as tf
import igm
import time

from igm.modules.utils import *

def params_rockflow(parser):
    parser.add_argument(
        "--speed_rockflow",
        type=float,
        default=1,
        help="speed rock flow",
    )

def init_rockflow(params,state):
    pass

def update_rockflow(params,state):

    slopsurfx, slopsurfy = compute_gradient_tf( state.usurf, state.dx, state.dx )
    
    slop = getmag(slopsurfx, slopsurfy)
    
    dirx = -params.speed_rockflow * tf.where(tf.not_equal(slop,0), slopsurfx/slop,1)
    diry = -params.speed_rockflow * tf.where(tf.not_equal(slop,0), slopsurfy/slop,1)

    thkexp = tf.repeat( tf.expand_dims(state.thk, axis=0), state.U.shape[1], axis=0)
    
    state.U[0].assign( tf.where( thkexp > 0, state.U[0], dirx ) )
    state.U[1].assign( tf.where( thkexp > 0, state.U[1], diry ) )    

def  final_rockflow(params,state):
    pass

# make sure to make these function new attributes of the igm module
igm.params_rockflow  = params_rockflow  
igm.init_rockflow    = init_rockflow  
igm.update_rockflow  = update_rockflow
igm.final_rockflow   = final_rockflow
    
