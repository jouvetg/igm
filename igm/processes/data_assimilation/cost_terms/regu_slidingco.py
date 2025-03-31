#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
import numpy as np

def regu_slidingco(cfg,state):

#    if not hasattr(state, "flowdirx"):
    dadx = (state.slidingco[:, 1:] - state.slidingco[:, :-1])/state.dx
    dady = (state.slidingco[1:, :] - state.slidingco[:-1, :])/state.dx

    if cfg.processes.data_assimilation.sole_mask:                
        dadx = tf.where( (state.icemaskobs[:, 1:] == 1) & (state.icemaskobs[:, :-1] == 1) , dadx, 0.0)
        dady = tf.where( (state.icemaskobs[1:, :] == 1) & (state.icemaskobs[:-1, :] == 1) , dady, 0.0)

    if cfg.processes.data_assimilation.smooth_anisotropy_factor_sl == 1:
        if cfg.processes.data_assimilation.fix_opti_normalization_issue:
            REGU_S = (cfg.processes.data_assimilation.regu_param_slidingco) * 0.5 * (
                tf.math.reduce_mean(dadx**2) + tf.math.reduce_mean(dady**2)
            )
        else:
            REGU_S = (cfg.processes.data_assimilation.regu_param_slidingco) * (
                tf.nn.l2_loss(dadx) + tf.nn.l2_loss(dady)
            )
    else:
        dadx = (state.slidingco[:, 1:] - state.slidingco[:, :-1])/state.dx
        dadx = (dadx[1:, :] + dadx[:-1, :]) / 2.0
        dady = (state.slidingco[1:, :] - state.slidingco[:-1, :])/state.dx
        dady = (dady[:, 1:] + dady[:, :-1]) / 2.0
 
        if cfg.processes.data_assimilation.sole_mask:
            MASK = (state.icemaskobs[1:, 1:] > 0.5) & (state.icemaskobs[1:, :-1] > 0.5) & (state.icemaskobs[:-1, 1:] > 0.5) & (state.icemaskobs[:-1, :-1] > 0.5)
            dadx = tf.where( MASK, dadx, 0.0)
            dady = tf.where( MASK, dady, 0.0)
 
        if cfg.processes.data_assimilation.fix_opti_normalization_issue:
            REGU_S = (cfg.processes.data_assimilation.regu_param_slidingco) * 0.5 * (
                (1.0/np.sqrt(cfg.processes.data_assimilation.smooth_anisotropy_factor_sl))
                * tf.math.reduce_mean((dadx * state.flowdirx + dady * state.flowdiry)**2)
                + np.sqrt(cfg.processes.data_assimilation.smooth_anisotropy_factor_sl)
                * tf.math.reduce_mean((dadx * state.flowdiry - dady * state.flowdirx)**2)
            )
        else:
            REGU_S = (cfg.processes.data_assimilation.regu_param_slidingco) * (
                (1.0/np.sqrt(cfg.processes.data_assimilation.smooth_anisotropy_factor_sl))
                * tf.nn.l2_loss((dadx * state.flowdirx + dady * state.flowdiry))
                + np.sqrt(cfg.processes.data_assimilation.smooth_anisotropy_factor_sl)
                * tf.nn.l2_loss((dadx * state.flowdiry - dady * state.flowdirx)) )
 
    if not cfg.processes.data_assimilation.log_slidingco:
        REGU_S = REGU_S + 10**10 * tf.math.reduce_mean( tf.where(state.slidingco >= 0, 0.0, state.slidingco**2) ) 
        # this last line serve to enforce non-negative slidingco
 
    return REGU_S
