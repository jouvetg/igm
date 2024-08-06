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
from igm.modules.process.iceflow_v1.iceflow_v1 import initialize as initialize_iceflow_v1

def params(parser):
    parser.add_argument(
        "--working_dir",
        type=str,
        default="",
        help="Working directory (default empty string)",
    )
    
    parser.add_argument(
        "--opti_vars_to_save",
        type=list,
        default=[
            "usurf",
            "thk",
            "strflowctrl",
            "arrhenius",
            "slidingco",
            "velsurf_mag",
            "velsurfobs_mag",
            "divflux",
        ],
        help="List of variables to be recorded in the ncdef file",
    )

    parser.add_argument(
        "--opti_thr_strflowctrl",
        type=float,
        default=78.0,
        help="threshold value for strflowctrl",
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
        "--opti_regu_param_strflowctrl",
        type=float,
        default=1.0,
        help="Regularization weight for the strflowctrl field in the optimization",
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
        "--opti_usurfobs_std",
        type=float,
        default=5.0,
        help="Confidence/STD of the top ice surface as input data for the optimization",
    )
    parser.add_argument(
        "--opti_strflowctrl_std",
        type=float,
        default=5.0,
        help="Confidence/STD of strflowctrl",
    )
    parser.add_argument(
        "--opti_velsurfobs_std",
        type=float,
        default=3.0,
        help="Confidence/STD of the surface ice velocities as input data for the optimization (if 0, velsurfobs_std field must be given)",
    )
    parser.add_argument(
        "--opti_thkobs_std",
        type=float,
        default=5.0,
        help="Confidence/STD of the ice thickness profiles (unless given)",
    )
    parser.add_argument(
        "--opti_divfluxobs_std",
        type=float,
        default=1.0,
        help="Confidence/STD of the flux divergence as input data for the optimization (if 0, divfluxobs_std field must be given)",
    )
    parser.add_argument(
        "--opti_control",
        type=list,
        default=["thk", "strflowctrl", "usurf"],
        help="List of optimized variables for the optimization",
    )
    parser.add_argument(
        "--opti_cost",
        type=list,
        default=["velsurf", "thk", "usurf", "divfluxfcz", "icemask"],
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
        default=1000,
        help="Max iterations for the optimization",
    )
    parser.add_argument(
        "--opti_step_size",
        type=float,
        default=0.001,
        help="Step size for the optimization",
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


def initialize(params, state):
    initialize_iceflow_v1(params, state)

    ###### PERFORM CHECKS PRIOR OPTIMIZATIONS

    # make sure this condition is satisfied
    assert ("usurf" in params.opti_cost) == ("usurf" in params.opti_control)

    # make sure the loaded ice flow emulator has these inputs
    assert (
        state.iceflow_mapping["fieldin"]
        == ["thk", "slopsurfx", "slopsurfy", "arrhenius", "slidingco"]
    ) | (
        state.iceflow_mapping["fieldin"]
        == ["thk", "slopsurfx", "slopsurfy", "strflowctrl"]
    )

    # make sure the loaded ice flow emulator has at least these outputs
    assert all(
        [
            (f in state.iceflow_mapping["fieldout"])
            for f in ["ubar", "vbar", "uvelsurf", "vvelsurf"]
        ]
    )

    # make sure that there are lease some profiles in thkobs
    if "thk" in params.opti_cost:
        assert not tf.reduce_all(tf.math.is_nan(state.thkobs))

    ###### PREPARE DATA PRIOR OPTIMIZATIONS

    if hasattr(state, "uvelsurfobs") & hasattr(state, "vvelsurfobs"):
        state.velsurfobs = tf.stack([state.uvelsurfobs, state.vvelsurfobs], axis=-1)

    if "divfluxobs" in params.opti_cost:
        state.divfluxobs = state.smb - state.dhdt

    if not params.opti_smooth_anisotropy_factor == 1:
        _compute_flow_direction_for_anisotropic_smoothing(state)

    if hasattr(state, "thkinit"):
        state.thk = state.thkinit
    else:
        state.thk = tf.zeros_like(state.thk)

    if params.opti_init_zero_thk:
        state.thk = tf.zeros_like(state.thk)

    ###### PREPARE OPIMIZER

    if int(tf.__version__.split(".")[1]) <= 10:
        optimizer = tf.keras.optimizers.Adam(learning_rate=params.opti_step_size)
    else:
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=params.opti_step_size)

    # initial_learning_rate * decay_rate ^ (step / decay_steps)
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay( initial_learning_rate=opti_step_size, decay_steps=100, decay_rate=0.9)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # add scalng for usurf
    state.iceflow_fieldbounds["usurf"] = (
        state.iceflow_fieldbounds["slopsurfx"] * state.dx
    )

    ###### PREPARE VARIABLES TO OPTIMIZE

    if state.iceflow_mapping["fieldin"] == [
        "thk",
        "slopsurfx",
        "slopsurfy",
        "arrhenius",
        "slidingco",
    ]:
        state.iceflow_fieldbounds["strflowctrl"] = (
            state.iceflow_fieldbounds["arrhenius"]
            + state.iceflow_fieldbounds["slidingco"]
        )

    thk = tf.Variable(state.thk / state.iceflow_fieldbounds["thk"])  # normalized vars
    strflowctrl = tf.Variable(
        state.strflowctrl / state.iceflow_fieldbounds["strflowctrl"]
    )  # normalized vars
    usurf = tf.Variable(
        state.usurf / state.iceflow_fieldbounds["usurf"]
    )  # normalized vars

    state.costs = []

    state.tcomp_optimize = []

    # main loop
    for i in range(params.opti_nbitmax):
        with tf.GradientTape() as t:
            state.tcomp_optimize.append(time.time())

            # is necessary to remember all operation to derive the gradients w.r.t. control variables
            if "thk" in params.opti_control:
                t.watch(thk)
            if "usurf" in params.opti_control:
                t.watch(usurf)
            if "strflowctrl" in params.opti_control:
                t.watch(strflowctrl)

            # update surface gradient
            if (i == 0) | ("usurf" in params.opti_control):
                slopsurfx, slopsurfy = compute_gradient_tf(
                    usurf * state.iceflow_fieldbounds["usurf"], state.dx, state.dx
                )
                slopsurfx = slopsurfx / state.iceflow_fieldbounds["slopsurfx"]
                slopsurfy = slopsurfy / state.iceflow_fieldbounds["slopsurfy"]

            if state.iceflow_mapping["fieldin"] == [
                "thk",
                "slopsurfx",
                "slopsurfy",
                "arrhenius",
                "slidingco",
            ]:
                thrv = (
                    params.opti_thr_strflowctrl
                    / state.iceflow_fieldbounds["strflowctrl"]
                )
                arrhenius = tf.where(strflowctrl <= thrv, strflowctrl, thrv)
                slidingco = tf.where(strflowctrl <= thrv, 0, strflowctrl - thrv)

                # build input of the emulator
                X = tf.concat(
                    [
                        tf.expand_dims(
                            tf.expand_dims(tf.pad(thk, state.PAD, "CONSTANT"), axis=0),
                            axis=-1,
                        ),
                        tf.expand_dims(
                            tf.expand_dims(
                                tf.pad(slopsurfx, state.PAD, "CONSTANT"), axis=0
                            ),
                            axis=-1,
                        ),
                        tf.expand_dims(
                            tf.expand_dims(
                                tf.pad(slopsurfy, state.PAD, "CONSTANT"), axis=0
                            ),
                            axis=-1,
                        ),
                        tf.expand_dims(
                            tf.expand_dims(
                                tf.pad(arrhenius, state.PAD, "CONSTANT"), axis=0
                            ),
                            axis=-1,
                        ),
                        tf.expand_dims(
                            tf.expand_dims(
                                tf.pad(slidingco, state.PAD, "CONSTANT"), axis=0
                            ),
                            axis=-1,
                        ),
                    ],
                    axis=-1,
                )

            elif state.iceflow_mapping["fieldin"] == [
                "thk",
                "slopsurfx",
                "slopsurfy",
                "strflowctrl",
            ]:
                # build input of the emulator
                X = tf.concat(
                    [
                        tf.expand_dims(
                            tf.expand_dims(tf.pad(thk, state.PAD, "CONSTANT"), axis=0),
                            axis=-1,
                        ),
                        tf.expand_dims(
                            tf.expand_dims(
                                tf.pad(slopsurfx, state.PAD, "CONSTANT"), axis=0
                            ),
                            axis=-1,
                        ),
                        tf.expand_dims(
                            tf.expand_dims(
                                tf.pad(slopsurfy, state.PAD, "CONSTANT"), axis=0
                            ),
                            axis=-1,
                        ),
                        tf.expand_dims(
                            tf.expand_dims(
                                tf.pad(strflowctrl, state.PAD, "CONSTANT"), axis=0
                            ),
                            axis=-1,
                        ),
                    ],
                    axis=-1,
                )
            else:
                # ONLY these 2 above cases were implemented !!!
                sys.exit("CHANGE THE ICE FLOW EMULATOR -- IMCOMPATIBLE FOR INVERSION ")

            # evalutae th ice flow emulator
            Y = state.iceflow_model(X)

            # get the dimensions of the working array
            Ny, Nx = state.thk.shape

            # save output variables into state.variables for outputs
            for kk, f in enumerate(state.iceflow_mapping["fieldout"]):
                vars(state)[f] = Y[0, :Ny, :Nx, kk] * state.iceflow_fieldbounds[f]

            # find index of variables in output
            iubar = state.iceflow_mapping["fieldout"].index("ubar")
            ivbar = state.iceflow_mapping["fieldout"].index("vbar")
            iuvsu = state.iceflow_mapping["fieldout"].index("uvelsurf")
            ivvsu = state.iceflow_mapping["fieldout"].index("vvelsurf")

            # save output of the emaultor to compute the costs function
            ubar = (
                Y[0, :Ny, :Nx, iubar] * state.iceflow_fieldbounds["ubar"]
            )  # NOT normalized vars
            vbar = (
                Y[0, :Ny, :Nx, ivbar] * state.iceflow_fieldbounds["vbar"]
            )  # NOT normalized vars
            uvelsurf = (
                Y[0, :Ny, :Nx, iuvsu] * state.iceflow_fieldbounds["uvelsurf"]
            )  # NOT normalized vars
            vvelsurf = (
                Y[0, :Ny, :Nx, ivvsu] * state.iceflow_fieldbounds["vvelsurf"]
            )  # NOT normalized vars
            velsurf = tf.stack([uvelsurf, vvelsurf], axis=-1)  # NOT normalized vars

            # misfit between surface velocity
            if "velsurf" in params.opti_cost:
                ACT = ~tf.math.is_nan(state.velsurfobs)
                COST_U = 0.5 * tf.reduce_mean(
                    (
                        (state.velsurfobs[ACT] - velsurf[ACT])
                        / params.opti_velsurfobs_std
                    )
                    ** 2
                )
            else:
                COST_U = tf.Variable(0.0)

            # misfit between ice thickness profiles
            if "thk" in params.opti_cost:
                ACT = ~tf.math.is_nan(state.thkobs)
                COST_H = 0.5 * tf.reduce_mean(
                    (
                        (
                            state.thkobs[ACT]
                            - thk[ACT] * state.iceflow_fieldbounds["thk"]
                        )
                        / params.opti_thkobs_std
                    )
                    ** 2
                )
            else:
                COST_H = tf.Variable(0.0)

            # misfit divergence of the flux
            if ("divfluxobs" in params.opti_cost) | ("divfluxfcz" in params.opti_cost):
                divflux = compute_divflux(
                    ubar,
                    vbar,
                    thk * state.iceflow_fieldbounds["thk"],
                    state.dx,
                    state.dx,
                )

                if "divfluxfcz" in params.opti_cost:
                    ACT = state.icemaskobs > 0.5
                    if i % 10 == 0:
                        # his does not need to be comptued any iteration as this is expensive
                        res = stats.linregress(
                            state.usurf[ACT], divflux[ACT]
                        )  # this is a linear regression (usually that's enough)
                    # or you may go for polynomial fit (more gl, but may leads to errors)
                    #  weights = np.polyfit(state.usurf[ACT],divflux[ACT], 2)
                    divfluxtar = tf.where(
                        ACT, res.intercept + res.slope * state.usurf, 0.0
                    )
                #                        divfluxtar = tf.where(ACT, np.poly1d(weights)(state.usurf) , 0.0 )

                else:
                    divfluxtar = state.divfluxobs

                ACT = state.icemaskobs > 0.5
                COST_D = 0.5 * tf.reduce_mean(
                    ((divfluxtar[ACT] - divflux[ACT]) / params.opti_divfluxobs_std) ** 2
                )

            else:
                COST_D = tf.Variable(0.0)

            # misfit between top ice surfaces
            if "usurf" in params.opti_cost:
                ACT = state.icemaskobs > 0.5
                COST_S = 0.5 * tf.reduce_mean(
                    (
                        (
                            usurf[ACT] * state.iceflow_fieldbounds["usurf"]
                            - state.usurfobs[ACT]
                        )
                        / params.opti_usurfobs_std
                    )
                    ** 2
                )
            else:
                COST_S = tf.Variable(0.0)

            # force zero thikness outisde the mask
            if "icemask" in params.opti_cost:
                COST_O = 10**10 * tf.math.reduce_mean(
                    tf.where(state.icemaskobs > 0.5, 0.0, thk**2)
                )
            else:
                COST_O = tf.Variable(0.0)

            # Here one enforces non-negative ice thickness, and possibly zero-thickness in user-defined ice-free areas.
            if "thk" in params.opti_control:
                COST_HPO = 10**10 * tf.math.reduce_mean(
                    tf.where(thk >= 0, 0.0, thk**2)
                )
            else:
                COST_HPO = tf.Variable(0.0)

            # # Make sur to keep reasonable values for strflowctrl
            if "strflowctrl" in params.opti_control:
                COST_STR = 0.5 * tf.reduce_mean(
                    (
                        (
                            strflowctrl * state.iceflow_fieldbounds["strflowctrl"]
                            - params.opti_thr_strflowctrl
                        )
                        / params.opti_strflowctrl_std
                    )
                    ** 2
                )
            else:
                COST_STR = tf.Variable(0.0)

            # Here one adds a regularization terms for the ice thickness to the cost function
            if "thk" in params.opti_control:
                if params.opti_smooth_anisotropy_factor == 1:
                    dbdx = thk[:, 1:] - thk[:, :-1]
                    dbdy = thk[1:, :] - thk[:-1, :]
                    REGU_H = params.opti_regu_param_thk * (
                        tf.nn.l2_loss(dbdx) + tf.nn.l2_loss(dbdy)
                    )
                else:
                    dbdx = thk[:, 1:] - thk[:, :-1]
                    dbdx = (dbdx[1:, :] + dbdx[:-1, :]) / 2.0
                    dbdy = thk[1:, :] - thk[:-1, :]
                    dbdy = (dbdy[:, 1:] + dbdy[:, :-1]) / 2.0
                    REGU_H = params.opti_regu_param_thk * (
                        tf.nn.l2_loss((dbdx * state.flowdirx + dbdy * state.flowdiry))
                        + params.opti_smooth_anisotropy_factor
                        * tf.nn.l2_loss((dbdx * state.flowdiry - dbdy * state.flowdirx))
                        - params.opti_convexity_weight * tf.math.reduce_sum(thk)
                    )
            else:
                REGU_H = tf.Variable(0.0)

            # Here one adds a regularization terms for strflowctrl to the cost function
            if "strflowctrl" in params.opti_control:
                dadx = tf.math.abs(strflowctrl[:, 1:] - strflowctrl[:, :-1])
                dady = tf.math.abs(strflowctrl[1:, :] - strflowctrl[:-1, :])
                dadx = tf.where(
                    (state.icemaskobs[:, 1:] > 0.5) & (state.icemaskobs[:, :-1] > 0.5),
                    dadx,
                    0.0,
                )
                dady = tf.where(
                    (state.icemaskobs[1:, :] > 0.5) & (state.icemaskobs[:-1, :] > 0.5),
                    dady,
                    0.0,
                )
                REGU_A = params.opti_regu_param_strflowctrl * (
                    tf.nn.l2_loss(dadx) + tf.nn.l2_loss(dady)
                )
            else:
                REGU_A = tf.Variable(0.0)

            # sum all component into the main cost function
            COST = (
                COST_U
                + COST_H
                + COST_D
                + COST_S
                + COST_O
                + COST_HPO
                + COST_STR
                + REGU_H
                + REGU_A
            )

            vol = (
                np.sum(thk * state.iceflow_fieldbounds["thk"])
                * (state.dx**2)
                / 10**9
            )

            if i % params.opti_output_freq == 0:
                print(
                    " OPTI, step %5.0f , ICE_VOL: %7.2f , COST_U: %7.2f , COST_H: %7.2f , COST_D : %7.2f , COST_S : %7.2f , REGU_H : %7.2f , REGU_A : %7.2f "
                    % (
                        i,
                        vol,
                        COST_U.numpy(),
                        COST_H.numpy(),
                        COST_D.numpy(),
                        COST_S.numpy(),
                        REGU_H.numpy(),
                        REGU_A.numpy(),
                    )
                )

            state.costs.append(
                [
                    COST_U.numpy(),
                    COST_H.numpy(),
                    COST_D.numpy(),
                    COST_S.numpy(),
                    REGU_H.numpy(),
                    REGU_A.numpy(),
                ]
            )

            var_to_opti = []
            if "thk" in params.opti_control:
                var_to_opti.append(thk)
            if "usurf" in params.opti_control:
                var_to_opti.append(usurf)
            if "strflowctrl" in params.opti_control:
                var_to_opti.append(strflowctrl)

            # Compute gradient of COST w.r.t. X
            grads = tf.Variable(t.gradient(COST, var_to_opti))

            # this serve to restict the optimization of controls to the mask
            for ii in range(grads.shape[0]):
                grads[ii].assign(tf.where((state.icemaskobs > 0.5), grads[ii], 0))

            # One step of descent -> this will update input variable X
            optimizer.apply_gradients(
                zip([grads[i] for i in range(grads.shape[0])], var_to_opti)
            )

            # get back optimized variables in the pool of state.variables
            if "thk" in params.opti_control:
                state.thk = thk * state.iceflow_fieldbounds["thk"]
                state.thk = tf.where(state.thk < 0.01, 0, state.thk)
            if "strflowctrl" in params.opti_control:
                state.strflowctrl = (
                    strflowctrl * state.iceflow_fieldbounds["strflowctrl"]
                )
            if "usurf" in params.opti_control:
                state.usurf = usurf * state.iceflow_fieldbounds["usurf"]

            state.divflux = compute_divflux(
                state.ubar, state.vbar, state.thk, state.dx, state.dx
            )

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

    # now that the ice thickness is optimized, we can fix the bed once for all!
    state.topg = state.usurf - state.thk

    _output_ncdf_optimize_final(params, state)

    _plot_cost_functions(params, state, state.costs)

    np.savetxt(
        os.path.join(params.working_dir, "costs.dat"),
        np.stack(state.costs),
        fmt="%.10f",
        header="        COST_U        COST_H      COST_D       COST_S       REGU_H       REGU_A          HPO ",
    )

    np.savetxt(
        os.path.join(params.working_dir, "rms_std.dat"),
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
        "echo rm " + os.path.join(params.working_dir, "rms_std.dat") + " >> clean.sh"
    )
    os.system(
        "echo rm " + os.path.join(params.working_dir, "costs.dat") + " >> clean.sh"
    )
    os.system(
        "echo rm " + os.path.join(params.working_dir, "volume.dat") + " >> clean.sh"
    )
    os.system(
        "echo rm "
        + os.path.join(params.working_dir, "tcompoptimize.dat")
        + " >> clean.sh"
    )


def update(params, state):
    pass


def finalize(params, state):
    pass


def _compute_rms_std_optimization(state, i):
    I = state.icemaskobs == 1

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

    if "arrhenius" in params.opti_vars_to_save:
        state.arrhenius = tf.where(
            state.strflowctrl <= params.opti_thr_strflowctrl,
            state.strflowctrl,
            params.opti_thr_strflowctrl,
        )

    if "slidingco" in params.opti_vars_to_save:
        state.slidingco = tf.where(
            state.strflowctrl <= params.opti_thr_strflowctrl,
            0,
            state.strflowctrl - params.opti_thr_strflowctrl,
        )

    if "velsurf_mag" in params.opti_vars_to_save:
        state.velsurf_mag = getmag(state.uvelsurf, state.vvelsurf)

    if "velsurfobs_mag" in params.opti_vars_to_save:
        state.velsurfobs_mag = getmag(state.uvelsurfobs, state.vvelsurfobs)

    if it == 0:
        nc = Dataset(
            os.path.join(params.working_dir, "optimize.nc"),
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

        os.system(
            "echo rm "
            + os.path.join(params.working_dir, "optimize.nc")
            + " >> clean.sh"
        )

    else:
        nc = Dataset(
            os.path.join(params.working_dir, "optimize.nc"),
            "a",
            format="NETCDF4",
        )

        d = nc.variables["iterations"][:].shape[0]

        nc.variables["iterations"][d] = it

        for var in params.opti_vars_to_save:
            nc.variables[var][d, :, :] = vars(state)[var].numpy()

        nc.close()


def _output_ncdf_optimize_final(params, state):
    """
    Write final geology after optimizing
    """

    nc = Dataset(
        os.path.join(params.working_dir, params.opti_save_result_in_ncdf),
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
        + os.path.join(params.working_dir, params.opti_save_result_in_ncdf)
        + " >> clean.sh"
    )


def _plot_cost_functions(params, state, costs):
    costs = np.stack(costs)

    for i in range(costs.shape[1]):
        costs[:, i] -= np.min(costs[:, i])
        costs[:, i] /= np.max(costs[:, i])

    fig = plt.figure(figsize=(10, 10))
    plt.plot(costs[:, 0], "-k", label="COST U")
    plt.plot(costs[:, 1], "-r", label="COST H")
    plt.plot(costs[:, 2], "-b", label="COST D")
    plt.plot(costs[:, 3], "-g", label="COST S")
    plt.plot(costs[:, 4], "--c", label="REGU H")
    plt.plot(costs[:, 5], "--m", label="REGU A")
    plt.ylim(0, 1)
    plt.legend()

    if params.opti_plot2d_live:
        plt.show()
    else:
        plt.savefig(os.path.join(params.working_dir, "convergence.png"), pad_inches=0)
        plt.close("all")

        os.system(
            "echo rm "
            + os.path.join(params.working_dir, "convergence.png")
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
        state.fig, state.axes = plt.subplots(2, 3)

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
        #                    vmax=np.quantile(state.thk, 0.98),
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

    im1 = ax2.imshow(
        np.ma.masked_where(state.thk == 0, state.strflowctrl),
        origin="lower",
        vmin=0,
        vmax=100,
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
        size=12,
    )
    ax4.axis("off")

    ########################################################

    ax5 = state.axes[1, 1]
    im1 = ax5.imshow(
        np.ma.masked_where(state.thk == 0, velsurfobs_mag),
        origin="lower",
        extent=state.extent,
        vmin=0,
        vmax=np.nanmax(velsurfobs_mag),
        cmap=cmap,
    )
    if i == 0:
        plt.colorbar(im1, format="%.2f", ax=ax5)
    ax5.set_title("Target \n Observed velocities", size=12)
    ax5.axis("off")

    #######################################################

    ax6 = state.axes[1, 2]
    im1 = ax6.imshow(
        np.ma.masked_where(state.thk == 0, state.divflux),
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
        plt.savefig(
            os.path.join(params.working_dir, "resu-opti-" + str(i).zfill(4) + ".png"),
            pad_inches=0,
        )
        plt.close("all")

        os.system(
            "echo rm " + os.path.join(params.working_dir, "*.png") + " >> clean.sh"
        )


def _compute_flow_direction_for_anisotropic_smoothing(state):
    uvelsurfobs = tf.where(tf.math.is_nan(state.uvelsurfobs), 0.0, state.uvelsurfobs)
    vvelsurfobs = tf.where(tf.math.is_nan(state.vvelsurfobs), 0.0, state.vvelsurfobs)

    state.flowdirx = (
        uvelsurfobs[1:, 1:]
        + uvelsurfobs[:-1, 1:]
        + uvelsurfobs[1:, :-1]
        + uvelsurfobs[:-1, :-1]
    ) / 4.0
    state.flowdiry = (
        vvelsurfobs[1:, 1:]
        + vvelsurfobs[:-1, 1:]
        + vvelsurfobs[1:, :-1]
        + vvelsurfobs[:-1, :-1]
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

    # this is to plot the observed flow directions
    # fig, axs = plt.subplots(1, 1, figsize=(8,16))
    # plt.quiver(state.flowdirx,state.flowdiry)
    # axs.axis("equal")
