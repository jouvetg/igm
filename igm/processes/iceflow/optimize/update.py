#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf

from ..energy_iceflow.energy_iceflow import *
from ..utils import *
from ..emulate.emulate import *

from .cost_terms.misfit_thk import misfit_thk
from .cost_terms.misfit_usurf import misfit_usurf
from .cost_terms.misfit_velsurf import misfit_velsurf
from .cost_terms.cost_divfluxfcz import cost_divfluxfcz
from .cost_terms.cost_divfluxobs import cost_divfluxobs
from .cost_terms.cost_vol import cost_vol
from .cost_terms.regu_thk import regu_thk
from .cost_terms.regu_slidingco import regu_slidingco
from .cost_terms.regu_arrhenius import regu_arrhenius

from .utils import compute_flow_direction_for_anisotropic_smoothing

def optimize_update(cfg, state, cost, i):

    sc = {}
    sc["thk"] = cfg.processes.iceflow.optimize.scaling_thk
    sc["usurf"] = cfg.processes.iceflow.optimize.scaling_usurf
    sc["slidingco"] = cfg.processes.iceflow.optimize.scaling_slidingco
    sc["arrhenius"] = cfg.processes.iceflow.optimize.scaling_arrhenius

    if i==0:

        for f in cfg.processes.iceflow.optimize.control:
            vars(state)[f+'_sc'] = tf.Variable(vars(state)[f] / sc[f])
            if cfg.processes.iceflow.optimize.log_slidingco & (f == "slidingco"):
                # vars(state)[f+'_sc'] = tf.Variable(( tf.math.log(vars(state)[f]) / tf.math.log(10.0) ) / sc[f]) 
                vars(state)[f+'_sc'] = tf.Variable( tf.sqrt(vars(state)[f] / sc[f]) ) 
            else:
                vars(state)[f+'_sc'] = tf.Variable(vars(state)[f] / sc[f]) 

    with tf.GradientTape() as t:

        if cfg.processes.iceflow.optimize.step_size_decay < 1:
            state.optimizer.lr = cfg.processes.iceflow.optimize.step_size * (cfg.processes.iceflow.optimize.step_size_decay ** (i / 100))

        # is necessary to remember all operation to derive the gradients w.r.t. control variables
        for f in cfg.processes.iceflow.optimize.control:
            t.watch(vars(state)[f+'_sc'])

        for f in cfg.processes.iceflow.optimize.control:
            if cfg.processes.iceflow.optimize.log_slidingco & (f == "slidingco"):
#                    vars(state)[f] = (10**(vars(state)[f+'_sc'] * sc[f]))
                vars(state)[f] =  (vars(state)[f+'_sc']**2) * sc[f]
            else:
                vars(state)[f] = vars(state)[f+'_sc'] * sc[f]

        update_iceflow_emulated(cfg, state)

        if not cfg.processes.iceflow.optimize.smooth_anisotropy_factor == 1:
            compute_flow_direction_for_anisotropic_smoothing(state)
                
        # misfit between surface velocity
        if "velsurf" in cfg.processes.iceflow.optimize.cost:
            cost["velsurf"] = misfit_velsurf(cfg,state)

        # misfit between ice thickness profiles
        if "thk" in cfg.processes.iceflow.optimize.cost:
            cost["thk"] = misfit_thk(cfg.processes.iceflow.optimize.thkobs_std, state)

        # misfit between divergence of flux
        if ("divfluxfcz" in cfg.processes.iceflow.optimize.cost):
            cost["divflux"] = cost_divfluxfcz(cfg, state, i)
        elif ("divfluxobs" in cfg.processes.iceflow.optimize.cost):
            cost["divflux"] = cost_divfluxobs(cfg, state, i)

        # misfit between top ice surfaces
        if "usurf" in cfg.processes.iceflow.optimize.cost:
            cost["usurf"] = misfit_usurf(cfg.processes.iceflow.optimize.usurfobs_std, state) 

        # force zero thikness outisde the mask
        if "icemask" in cfg.processes.iceflow.optimize.cost:
            cost["icemask"] = 10**10 * tf.math.reduce_mean( tf.where(state.icemaskobs > 0.5, 0.0, state.thk**2) )

        # Here one enforces non-negative ice thickness, and possibly zero-thickness in user-defined ice-free areas.
        if "thk" in cfg.processes.iceflow.optimize.control:
            cost["thk_positive"] = 10**10 * tf.math.reduce_mean( tf.where(state.thk >= 0, 0.0, state.thk**2) )
            
        if cfg.processes.iceflow.optimize.infer_params:
            cost["volume"] = cost_vol(cfg, state)

        # Here one adds a regularization terms for the bed toporgraphy to the cost function
        if "thk" in cfg.processes.iceflow.optimize.control:
            cost["thk_regu"] = regu_thk(cfg, state)

        # Here one adds a regularization terms for slidingco to the cost function
        if "slidingco" in cfg.processes.iceflow.optimize.control:
            cost["slid_regu"] = regu_slidingco(cfg, state)

        # Here one adds a regularization terms for arrhenius to the cost function
        if "arrhenius" in cfg.processes.iceflow.optimize.control:
            cost["arrh_regu"] = regu_arrhenius(cfg, state) 

        cost_total = tf.reduce_sum(tf.convert_to_tensor(list(cost.values())))

        #################

        var_to_opti = [ ]
        for f in cfg.processes.iceflow.optimize.control:
            var_to_opti.append(vars(state)[f+'_sc'])

        # Compute gradient of COST w.r.t. X
        grads = tf.Variable(t.gradient(cost_total, var_to_opti))

        # this serve to restict the optimization of controls to the mask
        if cfg.processes.iceflow.optimize.sole_mask:
            for ii in range(grads.shape[0]):
                if not "slidingco" == cfg.processes.iceflow.optimize.control[ii]:
                    grads[ii].assign(tf.where((state.icemaskobs > 0.5), grads[ii], 0))
                else:
                    grads[ii].assign(tf.where((state.icemaskobs == 1), grads[ii], 0))
        else:
            for ii in range(grads.shape[0]):
                if not "slidingco" == cfg.processes.iceflow.optimize.control[ii]:
                    grads[ii].assign(tf.where((state.icemaskobs > 0.5), grads[ii], 0))

        # One step of descent -> this will update input variable X
        state.optimizer.apply_gradients(
            zip([grads[i] for i in range(grads.shape[0])], var_to_opti)
        )

        ###################

        for f in cfg.processes.iceflow.optimize.control:
            if cfg.processes.iceflow.optimize.log_slidingco & (f == "slidingco"):
                # vars(state)[f] = (10**(vars(state)[f+'_sc'] * sc[f]))
                vars(state)[f] =  (vars(state)[f+'_sc']**2) * sc[f]
            else:
                vars(state)[f] = vars(state)[f+'_sc'] * sc[f]

        # get back optimized variables in the pool of state.variables
        if "thk" in cfg.processes.iceflow.optimize.control:
            state.thk = tf.where(state.icemaskobs > 0.5, state.thk, 0)
#                state.thk = tf.where(state.thk < 0.01, 0, state.thk)
        if "slidingco" in cfg.processes.iceflow.optimize.control:
            state.slidingco = tf.where(state.slidingco < 0, 0, state.slidingco)


        state.divflux = compute_divflux(
            state.ubar, state.vbar, state.thk, state.dx, state.dx, 
            method=cfg.processes.iceflow.optimize.divflux_method
        )

        #state.divflux = tf.where(ACT, state.divflux, 0.0)
 