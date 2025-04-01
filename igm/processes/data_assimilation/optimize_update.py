#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import tensorflow as tf
 
from igm.processes.iceflow.emulate.emulate import update_iceflow_emulated
from igm.processes.utils import compute_divflux
from .cost_terms.misfit_thk import misfit_thk
from .cost_terms.misfit_usurf import misfit_usurf
from .cost_terms.misfit_velsurf import misfit_velsurf
from .cost_terms.misfit_icemask import misfit_icemask
from .cost_terms.cost_divfluxfcz import cost_divfluxfcz
from .cost_terms.cost_divfluxobs import cost_divfluxobs
from .cost_terms.cost_vol import cost_vol
from .cost_terms.regu_thk import regu_thk
from .cost_terms.regu_slidingco import regu_slidingco
from .cost_terms.regu_arrhenius import regu_arrhenius

from .utils import compute_flow_direction_for_anisotropic_smoothing

def optimize_update(cfg, state, cost, i):

    sc = {}
    sc["thk"] = cfg.processes.data_assimilation.scaling_thk
    sc["usurf"] = cfg.processes.data_assimilation.scaling_usurf
    sc["slidingco"] = cfg.processes.data_assimilation.scaling_slidingco
    sc["arrhenius"] = cfg.processes.data_assimilation.scaling_arrhenius

    if i==0:

        for f in cfg.processes.data_assimilation.control:
            vars(state)[f+'_sc'] = tf.Variable(vars(state)[f] / sc[f])
            if cfg.processes.data_assimilation.log_slidingco & (f == "slidingco"):
                # vars(state)[f+'_sc'] = tf.Variable(( tf.math.log(vars(state)[f]) / tf.math.log(10.0) ) / sc[f]) 
                vars(state)[f+'_sc'] = tf.Variable( tf.sqrt(vars(state)[f] / sc[f]) ) 
            else:
                vars(state)[f+'_sc'] = tf.Variable(vars(state)[f] / sc[f]) 

    with tf.GradientTape() as t:

        if cfg.processes.data_assimilation.step_size_decay < 1:
            state.optimizer.lr = cfg.processes.data_assimilation.step_size * (cfg.processes.data_assimilation.step_size_decay ** (i / 100))

        # is necessary to remember all operation to derive the gradients w.r.t. control variables
        for f in cfg.processes.data_assimilation.control:
            t.watch(vars(state)[f+'_sc'])

        for f in cfg.processes.data_assimilation.control:
            if cfg.processes.data_assimilation.log_slidingco & (f == "slidingco"):
#                    vars(state)[f] = (10**(vars(state)[f+'_sc'] * sc[f]))
                vars(state)[f] =  (vars(state)[f+'_sc']**2) * sc[f]
            else:
                vars(state)[f] = vars(state)[f+'_sc'] * sc[f]

        update_iceflow_emulated(cfg, state)

        if not cfg.processes.data_assimilation.smooth_anisotropy_factor == 1:
            compute_flow_direction_for_anisotropic_smoothing(state)
                
        # misfit between surface velocity
        if "velsurf" in cfg.processes.data_assimilation.cost:
            cost["velsurf"] = misfit_velsurf(cfg,state)

        # misfit between ice thickness profiles
        if "thk" in cfg.processes.data_assimilation.cost:
            cost["thk"] = misfit_thk(cfg, state)

        # misfit between divergence of flux
        if ("divfluxfcz" in cfg.processes.data_assimilation.cost):
            cost["divflux"] = cost_divfluxfcz(cfg, state, i)
        elif ("divfluxobs" in cfg.processes.data_assimilation.cost):
            cost["divflux"] = cost_divfluxobs(cfg, state, i)

        # misfit between top ice surfaces
        if "usurf" in cfg.processes.data_assimilation.cost:
            cost["usurf"] = misfit_usurf(cfg, state) 

        # force zero thikness outisde the mask
        if "icemask" in cfg.processes.data_assimilation.cost:
            cost["icemask"] = misfit_icemask(cfg, state)

        # Here one enforces non-negative ice thickness
        if "thk" in cfg.processes.data_assimilation.control:
            cost["thk_positive"] = \
            10**10 * tf.math.reduce_mean( tf.where(state.thk >= 0, 0.0, state.thk**2) )

        # Here one enforces non-negative slidinco
        if ("slidingco" in cfg.processes.data_assimilation.control) & \
           (not cfg.processes.data_assimilation.log_slidingco):
            cost["slidingco_positive"] =  \
            10**10 * tf.math.reduce_mean( tf.where(state.slidingco >= 0, 0.0, state.slidingco**2) ) 

        # Here one enforces non-negative arrhenius
        if ("arrhenius" in cfg.processes.data_assimilation.control):
            cost["arrhenius_positive"] =  \
            10**10 * tf.math.reduce_mean( tf.where(state.arrhenius >= 0, 0.0, state.arrhenius**2) ) 
            
        if cfg.processes.data_assimilation.infer_params:
            cost["volume"] = cost_vol(cfg, state)

        # Here one adds a regularization terms for the bed toporgraphy to the cost function
        if "thk" in cfg.processes.data_assimilation.control:
            cost["thk_regu"] = regu_thk(cfg, state)

        # Here one adds a regularization terms for slidingco to the cost function
        if "slidingco" in cfg.processes.data_assimilation.control:
            cost["slid_regu"] = regu_slidingco(cfg, state)

        # Here one adds a regularization terms for arrhenius to the cost function
        if "arrhenius" in cfg.processes.data_assimilation.control:
            cost["arrh_regu"] = regu_arrhenius(cfg, state) 

        cost_total = tf.reduce_sum(tf.convert_to_tensor(list(cost.values())))

        #################

        var_to_opti = [ ]
        for f in cfg.processes.data_assimilation.control:
            var_to_opti.append(vars(state)[f+'_sc'])

        # Compute gradient of COST w.r.t. X
        grads = tf.Variable(t.gradient(cost_total, var_to_opti))

        # this serve to restict the optimization of controls to the mask
        if cfg.processes.data_assimilation.sole_mask:
            for ii in range(grads.shape[0]):
                if not "slidingco" == cfg.processes.data_assimilation.control[ii]:
                    grads[ii].assign(tf.where((state.icemaskobs > 0.5), grads[ii], 0))
                else:
                    grads[ii].assign(tf.where((state.icemaskobs == 1), grads[ii], 0))
        else:
            for ii in range(grads.shape[0]):
                if not "slidingco" == cfg.processes.data_assimilation.control[ii]:
                    grads[ii].assign(tf.where((state.icemaskobs > 0.5), grads[ii], 0))

        # One step of descent -> this will update input variable X
        state.optimizer.apply_gradients(
            zip([grads[i] for i in range(grads.shape[0])], var_to_opti)
        )

        ###################

        for f in cfg.processes.data_assimilation.control:
            if cfg.processes.data_assimilation.log_slidingco & (f == "slidingco"):
                # vars(state)[f] = (10**(vars(state)[f+'_sc'] * sc[f]))
                vars(state)[f] =  (vars(state)[f+'_sc']**2) * sc[f]
            else:
                vars(state)[f] = vars(state)[f+'_sc'] * sc[f]

        # get back optimized variables in the pool of state.variables
        if "thk" in cfg.processes.data_assimilation.control:
            state.thk = tf.where(state.icemaskobs > 0.5, state.thk, 0)

        if "slidingco" in cfg.processes.data_assimilation.control:
            state.slidingco = tf.where(state.slidingco < 0, 0, state.slidingco)

        if "arrhenius" in cfg.processes.data_assimilation.control:
            state.arrhenius = tf.where(state.arrhenius < 0, 0, state.arrhenius)

        state.divflux = compute_divflux(
            state.ubar, state.vbar, state.thk, state.dx, state.dx, 
            method=cfg.processes.data_assimilation.divflux_method
        )

        #state.divflux = tf.where(ACT, state.divflux, 0.0)
 