#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np
import os, copy
import matplotlib.pyplot as plt
import matplotlib
import datetime, time
import math
import tensorflow as tf
from scipy import stats
from netCDF4 import Dataset

from igm.modules.utils import *
from igm.modules.process.iceflow import initialize as initialize_iceflow
from igm.modules.process.iceflow import params as params_iceflow

from igm.modules.process.iceflow.iceflow import (
    fieldin_to_X,
    update_2d_iceflow_variables,
    iceflow_energy_XY,
    _update_iceflow_emulator,
    Y_to_UV,
    save_iceflow_model
)


def params(parser):
    # dependency on iceflow parameters...
    params_iceflow(parser)

    parser.add_argument(
        "--opti_vars_to_save",
        type=list,
        default=[
            "usurf",
            "thk",
            "slidingco",
            "velsurf_mag",
            "velsurfobs_mag",
            "divflux",
            "icemask",
        ],
        help="List of variables to be recorded in the ncdef file",
    )
    parser.add_argument(
        "--opti_init_zero_thk",
        type=str2bool,
        default="False",
        help="Initialize the optimization with zero ice thickness",
    )
    parser.add_argument(
        "--opti_regu_param_thk",
        type=float,
        default=10.0,
        help="Regularization weight for the ice thickness in the optimization",
    )
    parser.add_argument(
        "--opti_regu_param_slidingco",
        type=float,
        default=1,
        help="Regularization weight for the slidingco field in the optimization",
    )
    parser.add_argument(
        "--opti_regu_param_arrhenius",
        type=float,
        default=10.0,
        help="Regularization weight for the arrhenius field in the optimization",
    )
    parser.add_argument(
        "--opti_regu_param_div",
        type=float,
        default=1,
        help="Regularization weight for the divrgence field in the optimization",
    )
    parser.add_argument(
        "--opti_smooth_anisotropy_factor",
        type=float,
        default=0.2,
        help="Smooth anisotropy factor for the ice thickness regularization in the optimization",
    )
    parser.add_argument(
        "--opti_convexity_weight",
        type=float,
        default=0.002,
        help="Convexity weight for the ice thickness regularization in the optimization",
    )
    parser.add_argument(
        "--opti_convexity_power",
        type=float,
        default=1.3,
        help="Power b in the area-volume scaling V ~ a * A^b taking fom 'An estimate of global glacier volume', A. Grinste, TC, 2013",
    )
    parser.add_argument(
        "--opti_usurfobs_std",
        type=float,
        default=2.0,
        help="Confidence/STD of the top ice surface as input data for the optimization",
    )
    parser.add_argument(
        "--opti_velsurfobs_std",
        type=float,
        default=1.0,
        help="Confidence/STD of the surface ice velocities as input data for the optimization (if 0, velsurfobs_std field must be given)",
    )
    parser.add_argument(
        "--opti_thkobs_std",
        type=float,
        default=3.0,
        help="Confidence/STD of the ice thickness profiles (unless given)",
    )
    parser.add_argument(
        "--opti_divfluxobs_std",
        type=float,
        default=1.0,
        help="Confidence/STD of the flux divergence as input data for the optimization (if 0, divfluxobs_std field must be given)",
    )
    parser.add_argument(
        "--opti_divflux_method",
        type=str,
        default="upwind",
        help="Compute the divergence of the flux using the upwind or centered method",
    )
    parser.add_argument(
        "--opti_force_zero_sum_divflux",
        type=str2bool,
        default="False",
        help="Add a penalty to the cost function to force the sum of the divergence of the flux to be zero",
    )
    parser.add_argument(
        "--opti_scaling_thk",
        type=float,
        default=2.0,
        help="Scaling factor for the ice thickness in the optimization, serve to adjust step-size of each controls relative to each other",
    )
    parser.add_argument(
        "--opti_scaling_usurf",
        type=float,
        default=0.5,
        help="Scaling factor for the ice thickness in the optimization, serve to adjust step-size of each controls relative to each other",
    )
    parser.add_argument(
        "--opti_scaling_slidingco",
        type=float,
        default=0.0001,
        help="Scaling factor for the slidingco in the optimization, serve to adjust step-size of each controls relative to each other",
    )
    parser.add_argument(
        "--opti_scaling_arrhenius",
        type=float,
        default=0.1,
        help="Scaling factor for the Arrhenius in the optimization, serve to adjust step-size of each controls relative to each other",
    )
    parser.add_argument(
        "--opti_control",
        type=list,
        default=["thk"],  # "slidingco", "usurf"
        help="List of optimized variables for the optimization",
    )
    parser.add_argument(
        "--opti_cost",
        type=list,
        default=["velsurf", "thk", "icemask"],  # "divfluxfcz", ,"usurf"
        help="List of cost components for the optimization",
    )
    parser.add_argument(
        "--opti_nbitmin",
        type=int,
        default=50,
        help="Min iterations for the optimization",
    )
    parser.add_argument(
        "--opti_nbitmax",
        type=int,
        default=500,
        help="Max iterations for the optimization",
    )
    parser.add_argument(
        "--opti_step_size",
        type=float,
        default=1,
        help="Step size for the optimization",
    )
    parser.add_argument(
        "--opti_step_size_decay",
        type=float,
        default=0.9,
        help="Decay step size parameter for the optimization",
    )
    parser.add_argument(
        "--opti_output_freq",
        type=int,
        default=50,
        help="Frequency of the output for the optimization",
    )
    parser.add_argument(
        "--opti_save_result_in_ncdf",
        type=str,
        default="geology-optimized.nc",
        help="Geology input file",
    )
    parser.add_argument(
        "--opti_plot2d_live",
        type=str2bool,
        default=True,
        help="plot2d_live_inversion",
    )
    parser.add_argument(
        "--opti_plot2d",
        type=str2bool,
        default=True,
        help="plot 2d inversion",
    )
    parser.add_argument(
        "--opti_save_iterat_in_ncdf",
        type=str2bool,
        default=True,
        help="write_ncdf_optimize",
    )
    parser.add_argument(
        "--opti_editor_plot2d",
        type=str,
        default="vs",
        help="optimized for VS code (vs) or spyder (sp) for live plot",
    )
    parser.add_argument(
        "--opti_uniformize_thkobs",
        type=str2bool,
        default=True,
        help="uniformize the density of thkobs",
    )
    parser.add_argument(
        "--sole_mask",
        type=str2bool,
        default=False,
        help="sole_mask",
    )
    parser.add_argument(
        "--opti_retrain_iceflow_model",
        type=str2bool,
        default=True,
        help="Retrain the iceflow model simulatounously ?",
    )
    parser.add_argument(
       "--opti_to_regularize",
       type=str,
       default='topg',
       help="Field to regularize : topg or thk",
   )
    parser.add_argument(
       "--opti_include_low_speed_term",
       type=str2bool,
       default=False,
       help="opti_include_low_speed_term",
   ) 
    parser.add_argument(
        "--opti_infer_params",
        type=str2bool,
        default=False,
        help="infer slidingco and convexity weight from velocity observations",
    )
    parser.add_argument(
        "--opti_tidewater_glacier",
        type=str2bool,
        default=False,
        help="Is the glacier you're trying to infer parameters for a tidewater type?",
    )
    parser.add_argument(
        "--opti_vol_std",
        type=float,
        default=1000.0,
        help="Confidence/STD of the volume estimates from volume-area scaling",
    )

def initialize(params, state):
    """
    This function does the data assimilation (inverse modelling) to optimize thk, slidingco ans usurf from data
    """

    try:
        initialize_iceflow(params, state)
        
        state.it = -1
        
        _update_iceflow_emulator(params, state)
    except:
        return

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
    
    _optimize(params, state)
 
def _optimize(params, state):
    
    # this will infer values for slidingco and convexity weight based on the ice velocity and an empirical relationship from test glaciers with thickness profiles
    if params.opti_infer_params:
        #Because OGGM will index icemask from 0
        dummy = _infer_params(state, params)
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
                    _update_plot_inversion(params, state, i)
                if params.opti_save_iterat_in_ncdf:
                    _update_ncdf_optimize(params, state, i)

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
        _output_ncdf_optimize_final(params, state)

    _plot_cost_functions(params, state, state.costs)

    plt.close("all")

    save_costs(params, state)

    save_rms_std(params, state)

    # Flag so we can check if initialize was already called
    state.optimize_initializer_called = True


def update(params, state):
    pass


def finalize(params, state):
    if params.iflo_save_model:
        save_iceflow_model(params, state) 

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


def _update_ncdf_optimize(params, state, it):
    """
    Initialize and write the ncdf optimze file
    """

    if hasattr(state, "logger"):
        state.logger.info("Initialize  and write NCDF output Files")
        
    if "velbase_mag" in params.opti_vars_to_save:
        state.velbase_mag = getmag(state.uvelbase, state.vvelbase)

    if "velsurf_mag" in params.opti_vars_to_save:
        state.velsurf_mag = getmag(state.uvelsurf, state.vvelsurf)

    if "velsurfobs_mag" in params.opti_vars_to_save:
        state.velsurfobs_mag = getmag(state.uvelsurfobs, state.vvelsurfobs)
    
    if "sliding_ratio" in params.opti_vars_to_save:
        state.sliding_ratio = tf.where(state.velsurf_mag > 10, state.velbase_mag / state.velsurf_mag, np.nan)

    if it == 0:
        nc = Dataset(
            "optimize.nc",
            "w",
            format="NETCDF4",
        )

        nc.createDimension("iterations", None)
        E = nc.createVariable("iterations", np.dtype("float32").char, ("iterations",))
        E.units = "None"
        E.long_name = "iterations"
        E.axis = "ITERATIONS"
        E[0] = it

        nc.createDimension("y", len(state.y))
        E = nc.createVariable("y", np.dtype("float32").char, ("y",))
        E.units = "m"
        E.long_name = "y"
        E.axis = "Y"
        E[:] = state.y.numpy()

        nc.createDimension("x", len(state.x))
        E = nc.createVariable("x", np.dtype("float32").char, ("x",))
        E.units = "m"
        E.long_name = "x"
        E.axis = "X"
        E[:] = state.x.numpy()

        for var in params.opti_vars_to_save:
            E = nc.createVariable(
                var, np.dtype("float32").char, ("iterations", "y", "x")
            )
            E[0, :, :] = vars(state)[var].numpy()

        nc.close()

        os.system( "echo rm " + "optimize.nc" + " >> clean.sh" )

    else:
        nc = Dataset("optimize.nc", "a", format="NETCDF4", )

        d = nc.variables["iterations"][:].shape[0]

        nc.variables["iterations"][d] = it

        for var in params.opti_vars_to_save:
            nc.variables[var][d, :, :] = vars(state)[var].numpy()

        nc.close()


def _output_ncdf_optimize_final(params, state):
    """
    Write final geology after optimizing
    """
    if params.opti_save_iterat_in_ncdf==False:
        if "velbase_mag" in params.opti_vars_to_save:
            state.velbase_mag = getmag(state.uvelbase, state.vvelbase)

        if "velsurf_mag" in params.opti_vars_to_save:
            state.velsurf_mag = getmag(state.uvelsurf, state.vvelsurf)

        if "velsurfobs_mag" in params.opti_vars_to_save:
            state.velsurfobs_mag = getmag(state.uvelsurfobs, state.vvelsurfobs)
        
        if "sliding_ratio" in params.opti_vars_to_save:
            state.sliding_ratio = tf.where(state.velsurf_mag > 10, state.velbase_mag / state.velsurf_mag, np.nan)

    nc = Dataset(
        params.opti_save_result_in_ncdf,
        "w",
        format="NETCDF4",
    )

    nc.createDimension("y", len(state.y))
    E = nc.createVariable("y", np.dtype("float32").char, ("y",))
    E.units = "m"
    E.long_name = "y"
    E.axis = "Y"
    E[:] = state.y.numpy()

    nc.createDimension("x", len(state.x))
    E = nc.createVariable("x", np.dtype("float32").char, ("x",))
    E.units = "m"
    E.long_name = "x"
    E.axis = "X"
    E[:] = state.x.numpy()

    for v in params.opti_vars_to_save:
        if hasattr(state, v):
            E = nc.createVariable(v, np.dtype("float32").char, ("y", "x"))
            E.standard_name = v
            E[:] = vars(state)[v]

    nc.close()

    os.system(
        "echo rm "
        + params.opti_save_result_in_ncdf
        + " >> clean.sh"
    )


def _plot_cost_functions(params, state, costs):
    costs = np.stack(costs)

    for i in range(costs.shape[1]):
        costs[:, i] -= np.min(costs[:, i])
        costs[:, i] /= np.where(np.max(costs[:, i]) == 0, 1.0, np.max(costs[:, i]))

    fig = plt.figure(figsize=(10, 10))
    plt.plot(costs[:, 0], "-k", label="COST U")
    plt.plot(costs[:, 1], "-r", label="COST H")
    plt.plot(costs[:, 2], "-b", label="COST D")
    plt.plot(costs[:, 3], "-g", label="COST S")
    plt.plot(costs[:, 4], "--c", label="REGU H")
    plt.plot(costs[:, 5], "--m", label="REGU A")
    plt.ylim(0, 1)
    plt.legend()

    plt.savefig("convergence.png", pad_inches=0)
    plt.close("all")

    os.system(
        "echo rm "
        + "convergence.png"
        + " >> clean.sh"
    )


def _update_plot_inversion(params, state, i):
    """
    Plot thickness, velocity, mand slidingco"""

    if hasattr(state, "uvelsurfobs"):
        velsurfobs_mag = getmag(state.uvelsurfobs, state.vvelsurfobs).numpy()
    else:
        velsurfobs_mag = np.zeros_like(state.thk.numpy())

    if hasattr(state, "usurfobs"):
        usurfobs = state.usurfobs
    else:
        usurfobs = np.zeros_like(state.thk.numpy())

    velsurf_mag = getmag(state.uvelsurf, state.vvelsurf).numpy()

    #########################################################

    if i == 0:
        if params.opti_editor_plot2d == "vs":
            plt.ion()  # enable interactive mode

        # state.fig = plt.figure()
        state.fig, state.axes = plt.subplots(2, 3,figsize=(10, 8))

        state.extent = [state.x[0], state.x[-1], state.y[0], state.y[-1]]

    #########################################################

    cmap = copy.copy(matplotlib.cm.jet)
    cmap.set_bad(color="white")

    ax1 = state.axes[0, 0]

    im1 = ax1.imshow(
        np.ma.masked_where(state.thk == 0, state.thk),
        origin="lower",
        extent=state.extent,
        vmin=0,
        vmax=np.quantile(state.thk, 0.98),
        cmap=cmap,
    )
    if i == 0:
        plt.colorbar(im1, ax=ax1)
    ax1.set_title(
        "Ice thickness \n (RMS : "
        + str(int(state.rmsthk[-1]))
        + ", STD : "
        + str(int(state.stdthk[-1]))
        + ")",
        size=12,
    )
    ax1.axis("off")

    #########################################################

    ax2 = state.axes[0, 1]

    from matplotlib import colors

    im1 = ax2.imshow(
        state.slidingco,
        origin="lower",
#        norm=colors.LogNorm(),
        vmin=0.01,
        vmax=0.06,
        cmap=cmap,
    )
    if i == 0:
        plt.colorbar(im1, format="%.2f", ax=ax2)
    ax2.set_title("Iteration " + str(i) + " \n Sliding coefficient", size=12)
    ax2.axis("off")

    ########################################################

    ax3 = state.axes[0, 2]

    im1 = ax3.imshow(
        state.usurf - usurfobs,
        origin="lower",
        extent=state.extent,
        vmin=-10,
        vmax=10,
        cmap="RdBu",
    )
    if i == 0:
        plt.colorbar(im1, format="%.2f", ax=ax3)
    ax3.set_title(
        "Top surface adjustement \n (RMS : %5.1f , STD : %5.1f"
        % (state.rmsusurf[-1], state.stdusurf[-1])
        + ")",
        size=12,
    )
    ax3.axis("off")

    #########################################################

    cmap = copy.copy(matplotlib.cm.viridis)
    cmap.set_bad(color="white")

    ax4 = state.axes[1, 0]

    im1 = ax4.imshow(
        velsurf_mag, # np.ma.masked_where(state.thk == 0, velsurf_mag),
        origin="lower",
        extent=state.extent,
        norm=matplotlib.colors.LogNorm(vmin=1, vmax=5000),
        cmap=cmap,
    )
    if i == 0:
        plt.colorbar(im1, format="%.2f", ax=ax4)
    ax4.set_title(
        "Modelled velocities \n (RMS : "
        + str(int(state.rmsvel[-1]))
        + ", STD : "
        + str(int(state.stdvel[-1]))
        + ")",
        size=12,
    )
    ax4.axis("off")

    ########################################################

    ax5 = state.axes[1, 1]
    im1 = ax5.imshow(
        np.ma.masked_where(state.thk == 0, velsurfobs_mag),
        origin="lower",
        extent=state.extent,
        norm=matplotlib.colors.LogNorm(vmin=1, vmax=5000),
        cmap=cmap,
    )
    if i == 0:
        plt.colorbar(im1, format="%.2f", ax=ax5)
    ax5.set_title("Target \n Observed velocities", size=12)
    ax5.axis("off")

    #######################################################

    ax6 = state.axes[1, 2]
    im1 = ax6.imshow(
        state.divflux, # np.where(state.icemaskobs > 0.5, state.divflux,np.nan),
        origin="lower",
        extent=state.extent,
        vmin=-10,
        vmax=10,
        cmap="RdBu",
    )
    if i == 0:
        plt.colorbar(im1, format="%.2f", ax=ax6)
    ax6.set_title(
        "Flux divergence \n (RMS : %5.1f , STD : %5.1f"
        % (state.rmsdiv[-1], state.stddiv[-1])
        + ")",
        size=12,
    )
    ax6.axis("off")

    #########################################################

    if params.opti_plot2d_live:
        if params.opti_editor_plot2d == "vs":
            state.fig.canvas.draw()  # re-drawing the figure
            state.fig.canvas.flush_events()  # to flush the GUI events
        else:
            from IPython.display import display, clear_output

            clear_output(wait=True)
            display(state.fig)
    else:
        plt.savefig("resu-opti-" + str(i).zfill(4) + ".png", bbox_inches="tight", pad_inches=0.2)

        os.system( "echo rm " + "*.png" + " >> clean.sh" )


def _update_plot_inversion_simple(params, state, i):
    """
    Plot thickness, velocity, mand slidingco"""

    if hasattr(state, "uvelsurfobs"):
        velsurfobs_mag = getmag(state.uvelsurfobs, state.vvelsurfobs).numpy()
    else:
        velsurfobs_mag = np.zeros_like(state.thk.numpy())

    if hasattr(state, "usurfobs"):
        usurfobs = state.usurfobs
    else:
        usurfobs = np.zeros_like(state.thk.numpy())

    velsurf_mag = getmag(state.uvelsurf, state.vvelsurf).numpy()

    #########################################################

    if i == 0:
        if params.opti_editor_plot2d == "vs":
            plt.ion()  # enable interactive mode

        # state.fig = plt.figure()
        state.fig, state.axes = plt.subplots(1, 2)

        state.extent = [state.x[0], state.x[-1], state.y[0], state.y[-1]]

    #########################################################

    cmap = copy.copy(matplotlib.cm.jet)
    cmap.set_bad(color="white")

    ax1 = state.axes[0]

    im1 = ax1.imshow(
        np.ma.masked_where(state.thk == 0, state.thk),
        origin="lower",
        extent=state.extent,
        vmin=0,
        vmax=np.quantile(state.thk, 0.98),
        cmap=cmap,
    )
    if i == 0:
        plt.colorbar(im1, ax=ax1)
    ax1.set_title(
        "Ice thickness \n (RMS : "
        + str(int(state.rmsthk[-1]))
        + ", STD : "
        + str(int(state.stdthk[-1]))
        + ")",
        size=16,
    )
    ax1.axis("off")

    #########################################################

    cmap = copy.copy(matplotlib.cm.viridis)
    cmap.set_bad(color="white")

    ax4 = state.axes[1]

    im1 = ax4.imshow(
        np.ma.masked_where(state.thk == 0, velsurf_mag),
        origin="lower",
        extent=state.extent,
        vmin=0,
        vmax=np.nanmax(velsurfobs_mag),
        cmap=cmap,
    )
    if i == 0:
        plt.colorbar(im1, format="%.2f", ax=ax4)
    ax4.set_title(
        "Modelled velocities \n (RMS : "
        + str(int(state.rmsvel[-1]))
        + ", STD : "
        + str(int(state.stdvel[-1]))
        + ")",
        size=16,
    )
    ax4.axis("off")

    #########################################################

    if params.opti_plot2d_live:
        if params.opti_editor_plot2d == "vs":
            state.fig.canvas.draw()  # re-drawing the figure
            state.fig.canvas.flush_events()  # to flush the GUI events
        else:
            from IPython.display import display, clear_output

            clear_output(wait=True)
            display(state.fig)
    else:
        plt.savefig(
            "resu-opti-" + str(i).zfill(4) + ".png",
            pad_inches=0,
        )
        plt.close("all")

        os.system(
            "echo rm " + "*.png" + " >> clean.sh"
        )


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

def _infer_params(state, params):
    #This function allows for both parameters to be specified as varying 2D fields (you could compute them pixel-wise from VelMag by swapping in VelMag for VelPerc).
    #This is probably not a good idea, because the values of both parameters do not depend solely on the velocity at that point. But feel free to try! If you do
    #want to do that, you'll also need to un-comment the code block for smoothing and then converting the smoothed weights back to tensors (you may also want to
    #visualise the figures!), and set up a state.convexity_weights field to act as the 2D array
    import scipy
    
    #Get list of G entities in each C/get multi-valued ice mask
    #Loop over all Gs to construct parameter rasters
    
    percthreshold = 99
    NumGlaciers = int(tf.reduce_max(state.icemask).numpy())
    state.convexity_weights = tf.experimental.numpy.copy(state.icemaskobs)
    state.volumes = tf.experimental.numpy.copy(state.icemaskobs)
    state.volume_weights = tf.experimental.numpy.copy(state.icemaskobs)
    state.volume_weights = tf.where(state.icemaskobs > 0, params.opti_vol_std, 0.0)
    
    
    #Get some initial information
    VelMag = getmag(state.uvelsurfobs, state.vvelsurfobs)
    VelMag = tf.where(tf.math.is_nan(VelMag),1e-6,VelMag)
    VelMag = tf.where(VelMag==0,1e-6,VelMag)
    
    AnnualTemp = tf.reduce_mean(state.air_temp, axis=0)
    
    grady = tf.experimental.numpy.diff(state.usurfobs,axis=0)
    gradx = tf.experimental.numpy.diff(state.usurfobs,axis=1)
    
    #Pad gradients to have same size as DEM input
    paddingsx = tf.constant([[0,0],[1,1]])
    paddingsy = tf.constant([[1,1],[0,0]])
    
    grady = tf.pad(grady,paddingsy)
    gradx = tf.pad(gradx,paddingsx)
    
    #Calculate average gradient value per cell
    Slopes = (tf.math.abs(grady[1:,:])+tf.math.abs(grady[:-1,:])+tf.math.abs(gradx[:,1:])+tf.math.abs(gradx[:,:-1]))/4
    #Smooth slopes
    # Slopes = Slopes.numpy()
    # Slopes = scipy.signal.spline_filter(Slopes, lmbda=1000)
    # Slopes = tf.convert_to_tensor(Slopes)
    
    #Start of G loop
    for i in range(1,NumGlaciers+1):
        
        #Get area predictor
        Area = tf.reduce_sum(tf.where(state.icemask==i,1.0,0.0))*state.dx**2
        Area = np.round(Area.numpy()/1000**2, decimals=1)
        #print('Area is: ', Area)
        #print('Predicted volume is: ', np.exp(np.log(Area)*1.359622487))
        #Work out nominal volume target based on volume-area scaling - only has much of an effect if no other observations
        state.volumes = tf.where(state.icemaskobs == i, np.exp(np.log(Area)*1.354395785), state.volumes) #Make sure to put into km3!
        if Area <= 0.0:
            continue
        
        #Get velocity predictors
        VelMean = np.round(np.mean(VelMag[state.icemaskobs==i]),decimals=2)
        #print("Mean velocity is: ", VelMean)
        VelPerc = np.round(np.percentile(VelMag[state.icemaskobs==i], percthreshold))
        #print("99th percentile velocity is: ", VelPerc)
        
        if VelMean == 0.0:
            #state.icemask = tf.where(state.icemask == i, -1, state.icemask)
            #state.icemaskobs = tf.where(state.icemaskobs == i, -1, state.icemask)
            
            #With volume-area scaling (on the assumption these will all be very small)
            state.convexity_weights = tf.where(state.icemaskobs == i, 5.0, state.convexity_weights)
            state.slidingco = tf.where(state.icemaskobs == i, 0.1, state.slidingco)
            state.volume_weights = tf.where(state.icemaskobs == i, 0.001, state.volume_weights)
            
            continue
        
        #Get average annual air temperature across whole domain
        AvgTemp = tf.reduce_mean(AnnualTemp[state.icemaskobs==i]).numpy()
        AvgTemp = np.round(abs(AvgTemp), decimals=1)
        #print(AvgTemp)
        
        #Get thickness predictors (this uses Millan et al. (2022) thickness estimates just to give some idea of the expected ice thicknesses)
        MaxThk = tf.math.reduce_max(state.thkinit[state.icemask==i])
        MeanThk = tf.math.reduce_mean(state.thkinit[state.icemask==i])
        MaxThk = tf.math.round(MaxThk)
        MeanThk = tf.math.round(MeanThk)
        #print('Max and mean thickness are: ',MaxThk, MeanThk)
        
        #Get slope field
        #Compute spatial gradients of DEM
        #AvgSlope = np.round(tf.reduce_mean(Slopes[state.icemaskobs==i]).numpy(), decimals=1)
        AvgSlope = np.round(tf.reduce_max(state.slopes[state.icemaskobs==i]).numpy(), decimals=1)
        #print("Average Slope is: ", AvgSlope)
        
        Tidewater = params.opti_tidewater_glacier
        
        #Do regressions
        if hasattr(state, "tidewatermask"):
            if tf.reduce_max(state.tidewatermask[state.icemask == i]).numpy() == 1:
                Tidewater = True
            
        if Tidewater == True:
        
            #print('Tidewater')
            #Set up various constants
            VelMean1DCW = VelMean*-32.54498757
            Slope1DCW = AvgSlope*78.90165628
            
            MeanThkCW = MeanThk*2.39253558
            MaxThkCW = MaxThk*2.06356416
            AreaCW = Area*-0.671463838
            MaxVelCW = VelPerc*5.728069133
            MeanVelCW = VelMag*-32.54498757
            SlopeCW = Slopes*78.90165628
            
            VelMean1DSC = tf.math.log(VelMean)*0.779646843
            #Slope1DSC = tf.math.log(AvgSlope)*-0.075281289
            
            LogMeanThkSC = tf.math.log(MeanThk)*2.113449074
            LogTempSC = np.log(AvgTemp)*-0.305778358
            LogMaxThkSC = tf.math.log(MaxThk)*-1.45777084
            LogAreaSC = np.log(Area)*0.220126846
            LogMaxVelSC = np.log(VelPerc)*-1.003939161
            LogMeanVelSC = tf.math.log(VelMag)*0.779646843
            LogMeanVelSC = tf.where(tf.math.is_nan(LogMeanVelSC),0,LogMeanVelSC)
                    
            #These regressions are purely empirical and are based on a selection of 37 glaciers from around the world with thickness measurements,
            #where IGM inversions were performed and the best parameters chosen
            
            CW1D = MeanThkCW + MaxThkCW + AreaCW + MaxVelCW + VelMean1DCW + Slope1DCW - 1323.461811
            SC1D = tf.math.exp(LogMeanThkSC + LogTempSC + LogMaxThkSC + LogAreaSC + LogMaxVelSC + VelMean1DSC - 3.157484542)
            #print(CW1D.numpy(),SC1D.numpy())
            
            #state.convexity_weights[state.icemaskobs==i] = AreaCW + TempCW + MaxVelCW + MeanThkCW + SlopeCW + MeanVelCW + MaxThkCW + 6797.266974
            #state.slidingco[state.icemaskobs==i] = tf.math.exp(LogMeanThkSC + LogTempSC + LogAreaSC + LogMaxVelSC + LogMeanVelSC + LogMaxThkSC + LogSlopeSC + 31.04050337)
            
            #Max and min limiters to keep values inside sensible bounds (should only be needed for velocities very close to 0 or over 10,000 m/a)
            maxconvexityweight = 10000.0
            minconvexityweight = 5.0
            
            maxslidingco = 0.1
            minslidingco = 0.01
            
            state.convexity_weights = tf.where(state.icemaskobs == i, CW1D.numpy(), state.convexity_weights)
            state.convexity_weights = tf.where(state.convexity_weights > maxconvexityweight, maxconvexityweight, state.convexity_weights)
            state.convexity_weights = tf.where(state.convexity_weights < minconvexityweight, minconvexityweight, state.convexity_weights)
            
            state.slidingco = tf.where(state.icemaskobs == i, SC1D.numpy(), state.slidingco)
            state.slidingco = tf.where(state.slidingco > maxslidingco, maxslidingco, state.slidingco)
            state.slidingco = tf.where(state.slidingco < minslidingco, minslidingco, state.slidingco)      
           
            #Smooth both fields a lot
            ConvexityWeights = state.convexity_weights.numpy()
            SlidingCo = state.slidingco.numpy()
            
            #ConvexityWeights = scipy.signal.spline_filter(ConvexityWeights, lmbda=500000)
            #SlidingCo = scipy.signal.spline_filter(SlidingCo, lmbda=500000)
        else:
            if (AvgTemp < 7.0) & (AvgTemp > 0.0):
                #Set up various constants
                VelMean1DCW = tf.math.log(VelMean)*-188.4784857
                Slope1DCW = tf.math.log(AvgSlope)*158.3028365
                
                LogTempCW = np.log(AvgTemp)*-311.5007094
                LogMeanThkCW = tf.math.log(MeanThk)*385.4459893
                LogMeanVelCW = tf.math.log(VelMag)*-188.4784857
                LogMeanVelCW = tf.where(tf.math.is_nan(LogMeanVelCW),0,LogMeanVelCW)
                LogSlopeCW = tf.math.log(Slopes)*158.3028365
                LogSlopeCW = tf.where(tf.math.is_nan(LogSlopeCW),0,LogSlopeCW)
                LogAreaCW = np.log(Area)*248.6125172
                LogMaxThkCW = tf.math.log(MaxThk)*-512.6524209
                Log99VelCW = np.log(VelPerc)*119.2131799
                
                VelMean1DSC = VelMean*-0.000548089
                Slope1DSC = AvgSlope*0.000638754
                
                MeanVelSC = VelMag*-0.000548089
                SlopeSC = Slopes*0.000638754
                AreaSC = Area*-0.000902692
                MaxThkSC = MaxThk*0.000154141
                MaxVelSC = VelPerc*0.0000736281
                        
                #These regressions are purely empirical and are based on a selection of 37 glaciers from around the world with thickness measurements,
                #where IGM inversions were performed and the best parameters chosen
                
                #This fills the area outside the ice mask with the correct average inferred parameter values (so the smoothing works properly)
                CW1D = LogTempCW + LogMeanThkCW + VelMean1DCW + Slope1DCW + LogAreaCW + LogMaxThkCW + Log99VelCW + 798.3035279
                SC1D = VelMean1DSC + Slope1DSC + AreaSC + MaxThkSC + MaxVelSC + 0.021767969
                #print(CW1D.numpy(),SC1D.numpy())
                
                #state.convexity_weights[state.icemaskobs==i] = LogMaxThkSC + LogMeanVelCW + LogMeanThkCW + LogSlopeCW + LogAreaCW + Log99VelCW + LogTempCW + 801.9143279
                #state.slidingco[state.icemaskobs==i] = MaxThkSC + MeanVelSC + SlopeSC + AreaSC + TempSC + MaxVelSC + 0.017414339
                
                #Max and min limiters to keep values inside sensible bounds (should only be needed for velocities very close to 0 or over 10,000 m/a)
                maxconvexityweight = 1500.0
                minconvexityweight = 5.0
                
                maxslidingco = 0.1
                minslidingco = 0.01
                
                state.convexity_weights = tf.where(state.icemaskobs == i, CW1D.numpy(), state.convexity_weights)
                state.convexity_weights = tf.where(state.convexity_weights > maxconvexityweight, maxconvexityweight, state.convexity_weights)
                state.convexity_weights = tf.where(state.convexity_weights < minconvexityweight, minconvexityweight, state.convexity_weights)
                
                state.slidingco = tf.where(state.icemaskobs == i, SC1D.numpy(), state.slidingco)
                state.slidingco = tf.where(state.slidingco > maxslidingco, maxslidingco, state.slidingco)
                state.slidingco = tf.where(state.slidingco < minslidingco, minslidingco, state.slidingco)      
               
                #Smooth both fields a lot
                ConvexityWeights = state.convexity_weights.numpy()
                SlidingCo = state.slidingco.numpy()
                
                #ConvexityWeights = scipy.signal.spline_filter(ConvexityWeights, lmbda=500000)
                #SlidingCo = scipy.signal.spline_filter(SlidingCo, lmbda=500000)
                
            elif (AvgTemp >= 7.0) & (AvgTemp < 15.0):
                #Set up various constants        
                VelMean1DCW = VelMean*6.860440721
                #Slope1DCW = AvgSlope*7.076120236
                
                TempCW = AvgTemp*13.14963315
                MeanVelCW = VelMag*6.860440721
                AreaCW = Area*0.270785203
                MeanThkCW = MeanThk*-2.497513556
                MaxThkCW = MaxThk*1.190856765
                
                VelMean1DSC = VelMean*-0.006108525
                Slope1DSC = AvgSlope*0.004894406
                
                MaxVelSC = VelPerc*0.001061351
                SlopeSC = Slopes*0.004894406
                TempSC = AvgTemp*0.004704947
                MeanVelSC = VelMag*-0.006108525
                AreaSC = Area*-0.00044726
                MeanThkSC = MeanThk*0.001378985
                        
                #These regressions are purely empirical and are based on a selection of 37 glaciers from around the world with thickness measurements,
                #where IGM inversions were performed and the best parameters chosen
                
                CW1D = TempCW + VelMean1DCW + AreaCW + MeanThkCW + MaxThkCW - 102.6704589
                SC1D = MaxVelSC + Slope1DSC + TempSC + VelMean1DSC + AreaSC + MeanThkSC - 0.153944193
                #print(CW1D.numpy(),SC1D.numpy())
                
                #state.convexity_weights[state.icemaskobs==i] = MeanThkCW + MaxThkCW + MeanVelCW + AreaCW + MaxVelCW + TempCW - 32.12685937
                #state.slidingco[state.icemaskobs==i] = MaxVelSC + MeanThkSC + MaxThkSC + MeanVelSC + AreaSC + SlopeSC + TempSC - 0.028584459
                
                #Max and min limiters to keep values inside sensible bounds (should only be needed for velocities very close to 0 or over 10,000 m/a)
                maxconvexityweight = 1500.0
                minconvexityweight = 5.0
                
                maxslidingco = 0.1
                minslidingco = 0.01
                
                state.convexity_weights = tf.where(state.icemaskobs == i, CW1D.numpy(), state.convexity_weights)
                state.convexity_weights = tf.where(state.convexity_weights > maxconvexityweight, maxconvexityweight, state.convexity_weights)
                state.convexity_weights = tf.where(state.convexity_weights < minconvexityweight, minconvexityweight, state.convexity_weights)
                
                state.slidingco = tf.where(state.icemaskobs == i, SC1D.numpy(), state.slidingco)
                state.slidingco = tf.where(state.slidingco > maxslidingco, maxslidingco, state.slidingco)
                state.slidingco = tf.where(state.slidingco < minslidingco, minslidingco, state.slidingco)      
               
                #Smooth both fields a lot
                ConvexityWeights = state.convexity_weights.numpy()
                SlidingCo = state.slidingco.numpy()
                
                # ConvexityWeights = scipy.signal.spline_filter(ConvexityWeights, lmbda=500000)
                # SlidingCo = scipy.signal.spline_filter(SlidingCo, lmbda=500000)
                
            elif AvgTemp >= 15.0:
                #Set up various constants
                VelMean1DCW = tf.math.log(VelMean)*1.853792604
                Slope1DCW = tf.math.log(AvgSlope)*1.486422786
                
                LogMeanThkCW = tf.math.log(MeanThk)*0.41223867
                LogSlopeCW = tf.math.log(Slopes)*1.486422786
                LogSlopeCW = tf.where(tf.math.is_nan(LogSlopeCW),0,LogSlopeCW)
                LogMeanVelCW = tf.math.log(VelMag)*1.853792604
                LogMeanVelCW = tf.where(tf.math.is_nan(LogMeanVelCW),0,LogMeanVelCW)
                LogMaxThkCW = tf.math.log(MaxThk)*0.31532837
                LogMaxVelCW = np.log(VelPerc)*-1.542490077
                LogAreaCW = np.log(Area)*0.307141412
                
                VelMean1DSC = tf.math.log(VelMean)*0.008472703
                Slope1DSC = tf.math.log(AvgSlope)*-0.02595954
                
                LogTempSC = np.log(AvgTemp)*0.097948183
                LogMeanThkSC = tf.math.log(MeanThk)*-0.005875668
                LogSlopeSC = tf.math.log(Slopes)*-0.02595954
                LogSlopeSC = tf.where(tf.math.is_nan(LogSlopeSC),0,LogSlopeSC)
                LogMeanVelSC = tf.math.log(VelMag)*0.008472703
                LogMeanVelSC = tf.where(tf.math.is_nan(LogMeanVelSC),0,LogMeanVelSC)
                LogMaxThkSC = tf.math.log(MaxThk)*-0.010325374
                LogMaxVelSC = np.log(VelPerc)*-0.013272251
                        
                #These regressions are purely empirical and are based on a selection of 37 glaciers from around the world with thickness measurements,
                #where IGM inversions were performed and the best parameters chosen
                
                CW1D = tf.math.exp(LogMeanThkCW + Slope1DCW + VelMean1DCW + LogMaxThkCW + LogMaxVelCW + LogAreaCW - 1.681655194)
                SC1D = LogTempSC + LogMeanThkSC + Slope1DSC + VelMean1DSC + LogMaxThkSC + LogMaxVelSC - 0.072757514
                #print(CW1D.numpy(),SC1D.numpy())
                
                #state.convexity_weights[state.icemaskobs==i] = tf.math.exp(LogMeanVelCW + LogTempCW + LogMeanThkCW + LogMaxThkCW - 5.36050456)
                #state.slidingco[state.icemaskobs==i] = MaxVelSC + SlopeSC + AreaSC + MeanVelSC + MeanThkSC + MaxThkSC + 0.092110566
                
                #Max and min limiters to keep values inside sensible bounds (should only be needed for velocities very close to 0 or over 10,000 m/a)
                maxconvexityweight = 1500.0
                minconvexityweight = 5.0
                
                maxslidingco = 0.1
                minslidingco = 0.01
                
                state.convexity_weights = tf.where(state.icemaskobs == i, CW1D.numpy(), state.convexity_weights)
                state.convexity_weights = tf.where(state.convexity_weights > maxconvexityweight, maxconvexityweight, state.convexity_weights)
                state.convexity_weights = tf.where(state.convexity_weights < minconvexityweight, minconvexityweight, state.convexity_weights)
                
                state.slidingco = tf.where(state.icemaskobs == i, SC1D.numpy(), state.slidingco)
                state.slidingco = tf.where(state.slidingco > maxslidingco, maxslidingco, state.slidingco)
                state.slidingco = tf.where(state.slidingco < minslidingco, minslidingco, state.slidingco)      
               
                #Smooth both fields a lot
                ConvexityWeights = state.convexity_weights.numpy()
                SlidingCo = state.slidingco.numpy()
                
                #ConvexityWeights = scipy.signal.spline_filter(ConvexityWeights, lmbda=500000)
                #SlidingCo = scipy.signal.spline_filter(SlidingCo, lmbda=500000)
        
    #End of G loop
    #To plot weights if required
    # VolWeights = state.volume_weights.numpy()
    # VolumesNumpy = state.volumes.numpy()
    # fig = plt.figure(2, figsize=(8, 7),dpi=200) 
    # plt.subplot(2, 1, 1)
    # plt.imshow(VolWeights, cmap='jet',origin='lower')
    # plt.colorbar(label='volumeweights')
    # plt.title('volumeweights') 
    # plt.xlabel('Distance, km') 
    # plt.ylabel('Distance, km') 

    # plt.subplot(2, 1, 2)
    # plt.imshow(VolumesNumpy, cmap='jet',origin='lower')
    # plt.colorbar(label='volumes')
    # plt.title('volumes') 
    # plt.xlabel('Distance, km') 
    # plt.ylabel('Distance, km') 
    # plt.show()
    
    # ConvexityWeights = tf.convert_to_tensor(ConvexityWeights)
    # SlidingCo = tf.convert_to_tensor(SlidingCo)
    
    # state.convexity_weights = ConvexityWeights
    # state.slidingco = SlidingCo
    
    # #Reapply limiters in case smoothing has messed things up a bit
    # state.convexity_weights = tf.where(state.convexity_weights > maxconvexityweight, maxconvexityweight, state.convexity_weights)
    # state.convexity_weights = tf.where(state.convexity_weights < minconvexityweight, minconvexityweight, state.convexity_weights)
    
    # state.slidingco = tf.where(state.slidingco > maxslidingco, maxslidingco, state.slidingco)
    # state.slidingco = tf.where(state.slidingco < minslidingco, minslidingco, state.slidingco)