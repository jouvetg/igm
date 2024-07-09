#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os, copy
import matplotlib.pyplot as plt 
import datetime, time
import math
import tensorflow as tf
from scipy import stats 

from igm.modules.utils import * 
from .energy_iceflow import *
from .utils import *
from .emulate import *
from .optimize_outputs import *
from .optimize_params_cook import *
 
def optimize(params, state):

    ###### PERFORM CHECKS PRIOR OPTIMIZATIONS

    # make sure this condition is satisfied
    assert ("usurf" in params.opti_cost) == ("usurf" in params.opti_control)

    # make sure that there are lease some profiles in thkobs
    if tf.reduce_all(tf.math.is_nan(state.thkobs)):
        if "thk" in params.opti_cost:
            params.opti_cost.remove("thk")

    ###### PREPARE DATA PRIOR OPTIMIZATIONS
 
    if "divfluxobs" in params.opti_cost:
        if not hasattr(state, "divfluxobs"):
            state.divfluxobs = state.smb - state.dhdt

    if hasattr(state, "thkinit"):
        state.thk = state.thkinit
    else:
        state.thk = tf.zeros_like(state.thk)

    if params.opti_init_zero_thk:
        state.thk = state.thk*0.0
        
    # this is a density matrix that will be used to weight the cost function
    if params.opti_uniformize_thkobs:
        state.dens_thkobs = create_density_matrix(state.thkobs, kernel_size=5)
        state.dens_thkobs = tf.where(state.dens_thkobs>0, 1.0/state.dens_thkobs, 0.0)
        state.dens_thkobs = tf.where(tf.math.is_nan(state.thkobs),0.0,state.dens_thkobs)
        state.dens_thkobs = state.dens_thkobs / tf.reduce_mean(state.dens_thkobs[state.dens_thkobs>0])
    else:
        state.dens_thkobs = tf.ones_like(state.thkobs)
        
    # force zero slidingco in the floating areas
    state.slidingco = tf.where( state.icemaskobs == 2, 0.0, state.slidingco)
    
    # this will infer values for slidingco and convexity weight based on the ice velocity and an empirical relationship from test glaciers with thickness profiles
    if params.opti_infer_params:
        #Because OGGM will index icemask from 0
        dummy = infer_params_cook(state, params)
        if tf.reduce_max(state.icemask).numpy() < 1:
            return
    
    if (int(tf.__version__.split(".")[1]) <= 10) | (int(tf.__version__.split(".")[1]) >= 16) :
        optimizer = tf.keras.optimizers.Adam(learning_rate=params.opti_step_size)
        opti_retrain = tf.keras.optimizers.Adam(
            learning_rate=params.iflo_retrain_emulator_lr
        )
    else:
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=params.opti_step_size)
        opti_retrain = tf.keras.optimizers.legacy.Adam(
            learning_rate=params.iflo_retrain_emulator_lr
        )
 
    state.costs = []

    state.tcomp_optimize = []

    # this thing is outdated with using iflo_new_friction_param default as we use scaling of one.
    sc = {}
    sc["thk"] = params.opti_scaling_thk
    sc["usurf"] = params.opti_scaling_usurf
    sc["slidingco"] = params.opti_scaling_slidingco
    sc["arrhenius"] = params.opti_scaling_arrhenius
    
    Ny, Nx = state.thk.shape

    for f in params.opti_control:
        vars()[f] = tf.Variable(vars(state)[f] / sc[f])

    # main loop
    for i in range(params.opti_nbitmax):
        with tf.GradientTape() as t, tf.GradientTape() as s:
            state.tcomp_optimize.append(time.time())
            
            if params.opti_step_size_decay < 1:
                optimizer.lr = params.opti_step_size * (params.opti_step_size_decay ** (i / 100))

            # is necessary to remember all operation to derive the gradients w.r.t. control variables
            for f in params.opti_control:
                t.watch(vars()[f])

            for f in params.opti_control:
                vars(state)[f] = vars()[f] * sc[f]

            fieldin = [vars(state)[f] for f in params.iflo_fieldin]

            X = fieldin_to_X(params, fieldin)

            # evalutae th ice flow emulator                
            if params.iflo_multiple_window_size==0:
                Y = state.iceflow_model(X)
            else:
                Y = state.iceflow_model(tf.pad(X, state.PAD, "CONSTANT"))[:, :Ny, :Nx, :]

            U, V = Y_to_UV(params, Y)

            U = U[0]
            V = V[0]
           
            # this is strange, but it having state.U instead of U, slidingco is not more optimized ....
            state.uvelbase = U[0, :, :]
            state.vvelbase = V[0, :, :]
            state.ubar = tf.reduce_sum(U * state.vert_weight, axis=0)
            state.vbar = tf.reduce_sum(V * state.vert_weight, axis=0)
            state.uvelsurf = U[-1, :, :]
            state.vvelsurf = V[-1, :, :]
 
            if not params.opti_smooth_anisotropy_factor == 1:
                _compute_flow_direction_for_anisotropic_smoothing(state)
                 
            # misfit between surface velocity
            if "velsurf" in params.opti_cost:
                COST_U = misfit_velsurf(params, state)
            else:
                COST_U = tf.Variable(0.0)

            # misfit between ice thickness profiles
            if "thk" in params.opti_cost:
                COST_H = misfit_thk(params, state)
            else:
                COST_H = tf.Variable(0.0)

            # misfit divergence of the flux
            if ("divfluxobs" in params.opti_cost) | ("divfluxfcz" in params.opti_cost) | ("divfluxpen" in params.opti_cost):
                COST_D = cost_divflux(params, state, i)
            else:
                COST_D = tf.Variable(0.0)

            # misfit between top ice surfaces
            if "usurf" in params.opti_cost:
                COST_S = misfit_usurf(params, state)
            else:
                COST_S = tf.Variable(0.0)

            # force zero thikness outisde the mask
            if "icemask" in params.opti_cost:
                COST_O = 10**10 * tf.math.reduce_mean( tf.where(state.icemaskobs > 0.5, 0.0, state.thk**2) )
            else:
                COST_O = tf.Variable(0.0)

            # Here one enforces non-negative ice thickness, and possibly zero-thickness in user-defined ice-free areas.
            if "thk" in params.opti_control:
                COST_HPO = 10**10 * tf.math.reduce_mean( tf.where(state.thk >= 0, 0.0, state.thk**2) )
            else:
                COST_HPO = tf.Variable(0.0)
                
            if params.opti_infer_params:
                COST_VOL = cost_vol(params, state)
                #COST_VOL = tf.Variable(0.0)
            else:
                COST_VOL = tf.Variable(0.0)
    
            # Here one adds a regularization terms for the bed toporgraphy to the cost function
            if "thk" in params.opti_control:
                REGU_H = regu_thk(params, state)
            else:
                REGU_H = tf.Variable(0.0)

            # Here one adds a regularization terms for slidingco to the cost function
            if "slidingco" in params.opti_control:
                REGU_S = regu_slidingco(params, state)
            else:
                REGU_S = tf.Variable(0.0)

            # Here one adds a regularization terms for arrhenius to the cost function
            if "arrhenius" in params.opti_control:
                REGU_A = regu_arrhenius(params, state)
            else:
                REGU_A = tf.Variable(0.0)
 
            # sum all component into the main cost function
            COST = COST_U + COST_H + COST_D + COST_S + COST_O + COST_HPO + COST_VOL + REGU_H + REGU_S + REGU_A

            # Here one allow retraining of the ice flow emaultor
            if params.opti_retrain_iceflow_model:
                C_shear, C_slid, C_grav, C_float = iceflow_energy_XY(params, X, Y)

                COST_GLEN = tf.reduce_mean(C_shear) + tf.reduce_mean(C_slid) \
                          + tf.reduce_mean(C_grav)  + tf.reduce_mean(C_float)
                
                grads = s.gradient(COST_GLEN, state.iceflow_model.trainable_variables)

                opti_retrain.apply_gradients(
                    zip(grads, state.iceflow_model.trainable_variables)
                )
            else:
                COST_GLEN = tf.Variable(0.0)

            print_costs(params, state, COST_U, COST_H, COST_D, COST_S, REGU_H, REGU_S, COST_GLEN, i)


            state.costs.append(
                [
                    COST_U.numpy(),
                    COST_H.numpy(),
                    COST_D.numpy(),
                    COST_S.numpy(),
                    REGU_H.numpy(),
                    REGU_S.numpy(),
                    COST_GLEN.numpy(),
                    COST_VOL.numpy()
                ]
            )

            #################

            var_to_opti = []
            for f in params.opti_control:
                var_to_opti.append(vars()[f])

            # Compute gradient of COST w.r.t. X
            grads = tf.Variable(t.gradient(COST, var_to_opti))

            # this serve to restict the optimization of controls to the mask
            if params.sole_mask:
                for ii in range(grads.shape[0]):
                    if not "slidingco" == params.opti_control[ii]:
                        grads[ii].assign(tf.where((state.icemaskobs > 0.5), grads[ii], 0))
                    else:
                        grads[ii].assign(tf.where((state.icemaskobs == 1), grads[ii], 0))
            else:
                for ii in range(grads.shape[0]):
                    if not "slidingco" == params.opti_control[ii]:
                        grads[ii].assign(tf.where((state.icemaskobs > 0.5), grads[ii], 0))

            # One step of descent -> this will update input variable X
            optimizer.apply_gradients(
                zip([grads[i] for i in range(grads.shape[0])], var_to_opti)
            )

            ###################

            # get back optimized variables in the pool of state.variables
            if "thk" in params.opti_control:
                state.thk = tf.where(state.icemaskobs > 0.5, state.thk, 0)
#                state.thk = tf.where(state.thk < 0.01, 0, state.thk)

            state.divflux = compute_divflux(
                state.ubar, state.vbar, state.thk, state.dx, state.dx, method=params.opti_divflux_method
            )

            #state.divflux = tf.where(ACT, state.divflux, 0.0)

            _compute_rms_std_optimization(state, i)

            state.tcomp_optimize[-1] -= time.time()
            state.tcomp_optimize[-1] *= -1

            if i % params.opti_output_freq == 0:
                if params.opti_plot2d:
                    update_plot_inversion(params, state, i)
                if params.opti_save_iterat_in_ncdf:
                    update_ncdf_optimize(params, state, i)

            # stopping criterion: stop if the cost no longer decrease
            # if i>params.opti_nbitmin:
            #     cost = [c[0] for c in costs]
            #     if np.mean(cost[-10:])>np.mean(cost[-20:-10]):
            #         break;

    for f in params.opti_control:
        vars(state)[f] = vars()[f] * sc[f]

    # now that the ice thickness is optimized, we can fix the bed once for all! (ONLY FOR GROUNDED ICE)
    state.topg = state.usurf - state.thk

    if not params.opti_save_result_in_ncdf=="":
        output_ncdf_optimize_final(params, state)

    plot_cost_functions(params, state, state.costs)

    plt.close("all")

    save_costs(params, state)

    save_rms_std(params, state)

    # Flag so we can check if initialize was already called
    state.optimize_initializer_called = True
 
####################################

def misfit_velsurf(params,state):

    velsurf    = tf.stack([state.uvelsurf,    state.vvelsurf],    axis=-1) 
    velsurfobs = tf.stack([state.uvelsurfobs, state.vvelsurfobs], axis=-1)

    ACT = ~tf.math.is_nan(velsurfobs)

    cost = 0.5 * tf.reduce_mean(
           ( (velsurfobs[ACT] - velsurf[ACT]) / params.opti_velsurfobs_std  )** 2
    )

    if params.opti_include_low_speed_term:

        # This terms penalize the cost function when the velocity is low
        # Reference : Inversion of basal friction in Antarctica using exact and incompleteadjoints of a higher-order model
        # M. Morlighem, H. Seroussi, E. Larour, and E. Rignot, JGR, 2013
        cost += 0.5 * 100 * tf.reduce_mean(
            tf.math.log( (tf.norm(velsurf[ACT],axis=-1)+1) / (tf.norm(velsurfobs[ACT],axis=-1)+1) )** 2
        )

    return cost


def misfit_thk(params,state):

    ACT = ~tf.math.is_nan(state.thkobs)

    return 0.5 * tf.reduce_mean( state.dens_thkobs[ACT] * 
        ((state.thkobs[ACT] - state.thk[ACT]) / params.opti_thkobs_std) ** 2
    )

def cost_divflux(params,state,i):

    divflux = compute_divflux(
        state.ubar, state.vbar, state.thk, state.dx, state.dx, method=params.opti_divflux_method
    )

    # divflux = tf.where(ACT, divflux, 0.0)

    if "divfluxfcz" in params.opti_cost:
        ACT = state.icemaskobs > 0.5
        if i % 10 == 0:
            # his does not need to be comptued any iteration as this is expensive
            state.res = stats.linregress(
                state.usurf[ACT], divflux[ACT]
            )  # this is a linear regression (usually that's enough)
        # or you may go for polynomial fit (more gl, but may leads to errors)
        #  weights = np.polyfit(state.usurf[ACT],divflux[ACT], 2)
        divfluxtar = tf.where(
            ACT, state.res.intercept + state.res.slope * state.usurf, 0.0
        )
    #   divfluxtar = tf.where(ACT, np.poly1d(weights)(state.usurf) , 0.0 )
     
    if ("divfluxfcz" in params.opti_cost):
        ACT = state.icemaskobs > 0.5
        COST_D = 0.5 * tf.reduce_mean(
            ((divfluxtar[ACT] - divflux[ACT]) / params.opti_divfluxobs_std) ** 2
        )
        
    if ("divfluxobs" in params.opti_cost):
        divfluxtar = state.divfluxobs
        ACT = ~tf.math.is_nan(divfluxtar)
        COST_D = 0.5 * tf.reduce_mean(
            ((divfluxtar[ACT] - divflux[ACT]) / params.opti_divfluxobs_std) ** 2
        )

    if ("divfluxpen" in params.opti_cost):
        dddx = (divflux[:, 1:] - divflux[:, :-1])/state.dx
        dddy = (divflux[1:, :] - divflux[:-1, :])/state.dx
        COST_D = (params.opti_regu_param_div) * ( tf.nn.l2_loss(dddx) + tf.nn.l2_loss(dddy) )

    if params.opti_force_zero_sum_divflux:
            ACT = state.icemaskobs > 0.5
            COST_D += 0.5 * 1000 * tf.reduce_mean(divflux[ACT] / params.opti_divfluxobs_std) ** 2

    return COST_D

def misfit_usurf(params,state):

    ACT = state.icemaskobs > 0.5

    return 0.5 * tf.reduce_mean(
        (
            (state.usurf[ACT] - state.usurfobs[ACT])
            / params.opti_usurfobs_std
        )
        ** 2
    )

def cost_vol(params,state):

    ACT = state.icemaskobs > 0.5
    
    num_basins = int(tf.reduce_max(state.icemaskobs).numpy())
    ModVols = tf.experimental.numpy.copy(state.icemaskobs)
    
    for j in range(1,num_basins+1):
        ModVols = tf.where(ModVols==j,(tf.reduce_sum(tf.where(state.icemask==j,state.thk,0.0))*state.dx**2)/1e9,ModVols)

    cost = 0.5 * tf.reduce_mean(
           ( (state.volumes[ACT] - ModVols[ACT]) / state.volume_weights[ACT]  )** 2
    )
    return cost

def regu_thk(params,state):

    areaicemask = tf.reduce_sum(tf.where(state.icemask>0.5,1.0,0.0))*state.dx**2

    # here we had factor 8*np.pi*0.04, which is equal to 1
    if params.opti_infer_params:
        gamma = tf.zeros_like(state.thk)
        gamma = state.convexity_weights * areaicemask**(params.opti_convexity_power-2.0)
    else:
        gamma = params.opti_convexity_weight * areaicemask**(params.opti_convexity_power-2.0)

    if params.opti_to_regularize == 'topg':
        field = state.usurf - state.thk
    elif params.opti_to_regularize == 'thk':
        field = state.thk

    if params.opti_smooth_anisotropy_factor == 1:
        dbdx = (field[:, 1:] - field[:, :-1])/state.dx
        dbdy = (field[1:, :] - field[:-1, :])/state.dx

        if params.sole_mask:
            dbdx = tf.where( (state.icemaskobs[:, 1:] > 0.5) & (state.icemaskobs[:, :-1] > 0.5) , dbdx, 0.0)
            dbdy = tf.where( (state.icemaskobs[1:, :] > 0.5) & (state.icemaskobs[:-1, :] > 0.5) , dbdy, 0.0)

        REGU_H = (params.opti_regu_param_thk) * (
            tf.nn.l2_loss(dbdx) + tf.nn.l2_loss(dbdy)
            - gamma * tf.math.reduce_sum(state.thk)
        )
    else:
        dbdx = (field[:, 1:] - field[:, :-1])/state.dx
        dbdx = (dbdx[1:, :] + dbdx[:-1, :]) / 2.0
        dbdy = (field[1:, :] - field[:-1, :])/state.dx
        dbdy = (dbdy[:, 1:] + dbdy[:, :-1]) / 2.0

        if params.sole_mask:
            MASK = (state.icemaskobs[1:, 1:] > 0.5) & (state.icemaskobs[1:, :-1] > 0.5) & (state.icemaskobs[:-1, 1:] > 0.5) & (state.icemaskobs[:-1, :-1] > 0.5)
            dbdx = tf.where( MASK, dbdx, 0.0)
            dbdy = tf.where( MASK, dbdy, 0.0)

        if params.opti_infer_params:
            REGU_H = (params.opti_regu_param_thk) * (
                (1.0/np.sqrt(params.opti_smooth_anisotropy_factor))
                * tf.nn.l2_loss((dbdx * state.flowdirx + dbdy * state.flowdiry))
                + np.sqrt(params.opti_smooth_anisotropy_factor)
                * tf.nn.l2_loss((dbdx * state.flowdiry - dbdy * state.flowdirx))
                - tf.math.reduce_sum(gamma*state.thk)
            )
        else:
            REGU_H = (params.opti_regu_param_thk) * (
                (1.0/np.sqrt(params.opti_smooth_anisotropy_factor))
                * tf.nn.l2_loss((dbdx * state.flowdirx + dbdy * state.flowdiry))
                + np.sqrt(params.opti_smooth_anisotropy_factor)
                * tf.nn.l2_loss((dbdx * state.flowdiry - dbdy * state.flowdirx))
                - gamma * tf.math.reduce_sum(state.thk)
            )

    return REGU_H

def regu_slidingco(params,state):

#    if not hasattr(state, "flowdirx"):
    dadx = (state.slidingco[:, 1:] - state.slidingco[:, :-1])/state.dx
    dady = (state.slidingco[1:, :] - state.slidingco[:-1, :])/state.dx

    if params.sole_mask:                
        dadx = tf.where( (state.icemaskobs[:, 1:] == 1) & (state.icemaskobs[:, :-1] == 1) , dadx, 0.0)
        dady = tf.where( (state.icemaskobs[1:, :] == 1) & (state.icemaskobs[:-1, :] == 1) , dady, 0.0)

    REGU_S = (params.opti_regu_param_slidingco) * (
        tf.nn.l2_loss(dadx) + tf.nn.l2_loss(dady)
    )
    + 10**10 * tf.math.reduce_mean( tf.where(state.slidingco >= 0, 0.0, state.slidingco**2) ) 
    # this last line serve to enforce non-negative slidingco

#               else:
        # dadx = (state.slidingco[:, 1:] - state.slidingco[:, :-1])/state.dx
        # dadx = (dadx[1:, :] + dadx[:-1, :]) / 2.0
        # dady = (state.slidingco[1:, :] - state.slidingco[:-1, :])/state.dx
        # dady = (dady[:, 1:] + dady[:, :-1]) / 2.0

        # if params.sole_mask:
        #     MASK = (state.icemaskobs[1:, 1:] > 0.5) & (state.icemaskobs[1:, :-1] > 0.5) & (state.icemaskobs[:-1, 1:] > 0.5) & (state.icemaskobs[:-1, :-1] > 0.5)
        #     dadx = tf.where( MASK, dadx, 0.0)
        #     dady = tf.where( MASK, dady, 0.0)

        # REGU_S = (params.opti_regu_param_slidingco) * (
        #     (1.0/np.sqrt(params.opti_smooth_anisotropy_factor))
        #     * tf.nn.l2_loss((dadx * state.flowdirx + dady * state.flowdiry))
        #     + np.sqrt(params.opti_smooth_anisotropy_factor)
        #     * tf.nn.l2_loss((dadx * state.flowdiry - dady * state.flowdirx))

    return REGU_S

def regu_arrhenius(params,state):

#    if not hasattr(state, "flowdirx"):
    dadx = (state.arrhenius[:, 1:] - state.arrhenius[:, :-1])/state.dx
    dady = (state.arrhenius[1:, :] - state.arrhenius[:-1, :])/state.dx

    if params.sole_mask:                
        dadx = tf.where( (state.icemaskobs[:, 1:] == 1) & (state.icemaskobs[:, :-1] == 1) , dadx, 0.0)
        dady = tf.where( (state.icemaskobs[1:, :] == 1) & (state.icemaskobs[:-1, :] == 1) , dady, 0.0)
    
    REGU_S = (params.opti_regu_param_arrhenius) * (
        tf.nn.l2_loss(dadx) + tf.nn.l2_loss(dady)
    )
    + 10**10 * tf.math.reduce_mean( tf.where(state.arrhenius >= 0, 0.0, state.arrhenius**2) ) 
    # this last line serve to enforce non-negative arrhenius 
        
    return REGU_S

##################################

def print_costs(params, state, COST_U, COST_H, COST_D, COST_S, REGU_H, REGU_S, COST_GLEN, i):

    vol = np.sum(state.thk) * (state.dx**2) / 10**9

    mean_slidingco = tf.math.reduce_mean(state.slidingco[state.icemaskobs > 0.5])

    if i == 0:
        print(
            "                   Step  |  ICE_VOL |  COST_U  |  COST_H  |  COST_D  |  COST_S  |   REGU_H |   REGU_S | COST_GLEN | MEAN_SLIDCO   "
        )

    if i % params.opti_output_freq == 0:
        print(
            "OPTI %s :   %6.0f |   %6.2f |   %6.2f |   %6.2f |   %6.2f |   %6.2f |   %6.2f |   %6.2f |   %6.2f |   %6.4f |"
            % (
                datetime.datetime.now().strftime("%H:%M:%S"),
                i,
                vol,
                COST_U.numpy(),
                COST_H.numpy(),
                COST_D.numpy(),
                COST_S.numpy(),
                REGU_H.numpy(),
                REGU_S.numpy(),
                COST_GLEN.numpy(),
                mean_slidingco.numpy()
            )
        )

def save_costs(params, state):

    np.savetxt(
        "costs.dat",
        np.stack(state.costs),
        fmt="%.10f",
        header="        COST_U        COST_H      COST_D       COST_S       REGU_H       REGU_S          HPO           COSTGLEN ",
    )


    os.system(
        "echo rm " + "costs.dat" + " >> clean.sh"
    )
    
def save_rms_std(params, state):

    np.savetxt(
        "rms_std.dat",
        np.stack(
            [
                state.rmsthk,
                state.stdthk,
                state.rmsvel,
                state.stdvel,
                state.rmsdiv,
                state.stddiv,
                state.rmsusurf,
                state.stdusurf,
            ],
            axis=-1,
        ),
        fmt="%.10f",
        header="        rmsthk      stdthk       rmsvel       stdvel       rmsdiv       stddiv       rmsusurf       stdusurf",
    )

    os.system(
        "echo rm " + "rms_std.dat" + " >> clean.sh"
    )

def create_density_matrix(data, kernel_size):
    # Convert data to binary mask (1 for valid data, 0 for NaN)
    binary_mask = tf.where(tf.math.is_nan(data), tf.zeros_like(data), tf.ones_like(data))

    # Create a kernel for convolution (all ones)
    kernel = tf.ones((kernel_size, kernel_size, 1, 1), dtype=binary_mask.dtype)

    # Apply convolution to count valid data points in the neighborhood
    density = tf.nn.conv2d(tf.expand_dims(tf.expand_dims(binary_mask, 0), -1), 
                           kernel, strides=[1, 1, 1, 1], padding='SAME')

    # Remove the extra dimensions added for convolution
    density = tf.squeeze(density)

    return density

def _compute_rms_std_optimization(state, i):
    I = state.icemaskobs > 0.5

    if i == 0:
        state.rmsthk = []
        state.stdthk = []
        state.rmsvel = []
        state.stdvel = []
        state.rmsusurf = []
        state.stdusurf = []
        state.rmsdiv = []
        state.stddiv = []

    if hasattr(state, "thkobs"):
        ACT = ~tf.math.is_nan(state.thkobs)
        if np.sum(ACT) == 0:
            state.rmsthk.append(0)
            state.stdthk.append(0)
        else:
            state.rmsthk.append(np.nanmean(state.thk[ACT] - state.thkobs[ACT]))
            state.stdthk.append(np.nanstd(state.thk[ACT] - state.thkobs[ACT]))

    else:
        state.rmsthk.append(0)
        state.stdthk.append(0)

    if hasattr(state, "uvelsurfobs"):
        velsurf_mag = getmag(state.uvelsurf, state.vvelsurf).numpy()
        velsurfobs_mag = getmag(state.uvelsurfobs, state.vvelsurfobs).numpy()
        ACT = ~np.isnan(velsurfobs_mag)

        state.rmsvel.append(
            np.mean(velsurf_mag[(I & ACT).numpy()] - velsurfobs_mag[(I & ACT).numpy()])
        )
        state.stdvel.append(
            np.std(velsurf_mag[(I & ACT).numpy()] - velsurfobs_mag[(I & ACT).numpy()])
        )
    else:
        state.rmsvel.append(0)
        state.stdvel.append(0)

    if hasattr(state, "divfluxobs"):
        state.rmsdiv.append(np.mean(state.divfluxobs[I] - state.divflux[I]))
        state.stddiv.append(np.std(state.divfluxobs[I] - state.divflux[I]))
    else:
        state.rmsdiv.append(0)
        state.stddiv.append(0)

    if hasattr(state, "usurfobs"):
        state.rmsusurf.append(np.mean(state.usurf[I] - state.usurfobs[I]))
        state.stdusurf.append(np.std(state.usurf[I] - state.usurfobs[I]))
    else:
        state.rmsusurf.append(0)
        state.stdusurf.append(0)


def _compute_flow_direction_for_anisotropic_smoothing(state):
    uvelsurf = tf.where(tf.math.is_nan(state.uvelsurf), 0.0, state.uvelsurf)
    vvelsurf = tf.where(tf.math.is_nan(state.vvelsurf), 0.0, state.vvelsurf)

    state.flowdirx = (
        uvelsurf[1:, 1:] + uvelsurf[:-1, 1:] + uvelsurf[1:, :-1] + uvelsurf[:-1, :-1]
    ) / 4.0
    state.flowdiry = (
        vvelsurf[1:, 1:] + vvelsurf[:-1, 1:] + vvelsurf[1:, :-1] + vvelsurf[:-1, :-1]
    ) / 4.0

    from scipy.ndimage import gaussian_filter

    state.flowdirx = gaussian_filter(state.flowdirx, 3, mode="constant")
    state.flowdiry = gaussian_filter(state.flowdiry, 3, mode="constant")

    # Same as gaussian filter above but for tensorflow is (NOT TESTED)
    # import tensorflow_addons as tfa
    # state.flowdirx = ( tfa.image.gaussian_filter2d( state.flowdirx , sigma=3, filter_shape=100, padding="CONSTANT") )

    state.flowdirx /= getmag(state.flowdirx, state.flowdiry)
    state.flowdiry /= getmag(state.flowdirx, state.flowdiry)

    state.flowdirx = tf.where(tf.math.is_nan(state.flowdirx), 0.0, state.flowdirx)
    state.flowdiry = tf.where(tf.math.is_nan(state.flowdiry), 0.0, state.flowdiry)
    
    # state.flowdirx = tf.zeros_like(state.flowdirx)
    # state.flowdiry = tf.ones_like(state.flowdiry)

    # this is to plot the observed flow directions
    # fig, axs = plt.subplots(1, 1, figsize=(8,16))
    # plt.quiver(state.flowdirx,state.flowdiry)
    # axs.axis("equal")