#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf 

def misfit_icemask(cfg,state):    
    return 10**10 * tf.math.reduce_mean( tf.where(state.icemaskobs > 0.5, 0.0, state.thk**2) )
