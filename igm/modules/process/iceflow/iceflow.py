#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file

"""
 Quick notes about the code below:
 
 The goal of this module is to compute the ice flow velocity field
 using a deep-learning emulator of the Blatter-Pattyn model.
  
 The aim of this module is
   - to initialize the ice flow and its emulator in init_iceflow
   - to update the ice flow and its emulator in update_iceflow

In update_iceflow, we compute/update with function _update_iceflow_emulated,
and retraine the iceflow emaultor in function _update_iceflow_emulator

- in _update_iceflow_emulated, we baiscially gather together all input fields
of the emulator and stack all in a single tensor X, then we compute the output
with Y = iceflow_model(X), and finally we split Y into U and V

- in _update_iceflow_emulator, we retrain the emulator. For that purpose, we
iteratively (usually we do only one iteration) compute the output of the emulator,
compute the energy associated with the state of the emulator, and compute the
gradient of the energy with respect to the emulator parameters. Then we update
the emulator parameters with the gradient descent method (Adam optimizer).
Because this step may be memory consuming, we split the computation in several
patches of size params.iflo_retrain_emulator_framesizemax. This permits to
retrain the emulator on large size arrays.

Alternatively, one can solve the Blatter-Pattyn model using a solver using 
function _update_iceflow_solved. Doing so is not very different to retrain the
emulator as we minmize the same energy, however, with different controls,
namely directly the velocity field U and V instead of the emulator parameters.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import datetime, time
import math
import tensorflow as tf

from igm.modules.utils import *

import igm
from igm import emulators
import importlib_resources

############################################


def params(parser):
    # type of ice flow computations
    parser.add_argument(
        "--iflo_type",
        type=str,
        default="emulated",
        help="Type of iceflow: it can emulated (default), solved, or in diagnostic mode to investigate the fidelity of the emulator towads the solver",
    )

    parser.add_argument(
        "--iflo_pretrained_emulator",
        type=str2bool,
        default=True,
        help="Do we take a pretrained emulator or start from scratch?",
    )
    parser.add_argument(
        "--iflo_emulator",
        type=str,
        default="",
        help="Directory path of the deep-learning pretrained ice flow model, take from the library if empty string",
    )

    # physical parameters
    parser.add_argument(
        "--iflo_init_slidingco",
        type=float,
        default=0.0464,
        help="Initial sliding coefficient slidingco",
    )
    parser.add_argument(
        "--iflo_init_arrhenius",
        type=float,
        default=78,
        help="Initial arrhenius factor arrhenuis",
    )
    parser.add_argument(
        "--iflo_enhancement_factor",
        type=float,
        default=1.0,
        help="Enhancement factor multiying the arrhenius factor",
    )
    parser.add_argument(
        "--iflo_regu_glen",
        type=float,
        default=10 ** (-5),
        help="Regularization parameter for Glen's flow law",
    )
    parser.add_argument(
        "--iflo_regu_weertman",
        type=float,
        default=10 ** (-10),
        help="Regularization parameter for Weertman's sliding law",
    )
    parser.add_argument(
        "--iflo_exp_glen",
        type=float,
        default=3,
        help="Glen's flow law exponent",
    )
    parser.add_argument(
        "--iflo_exp_weertman", type=float, default=3, help="Weertman's law exponent"
    )
    parser.add_argument(
        "--iflo_gravity_cst",
        type=float,
        default=9.81,
        help="Gravitational constant",
    )
    parser.add_argument(
        "--iflo_ice_density",
        type=float,
        default=910,
        help="Density of ice",
    )
    parser.add_argument(
        "--iflo_new_friction_param",
        type=str2bool,
        default=True,
        help="Sliding coeeficient (this describe slidingco differently with slidingco**-(1.0 / exp_weertman) instead of slidingco as before)",
    )
    parser.add_argument(
        "--iflo_save_model",
        type=str2bool,
        default=False,
        help="save the iceflow emaultor at the end of the simulation",
    )

    # vertical discretization
    parser.add_argument(
        "--iflo_Nz",
        type=int,
        default=10,
        help="Number of grid point for the vertical discretization",
    )
    parser.add_argument(
        "--iflo_vert_spacing",
        type=float,
        default=4.0,
        help="Parameter controlling the discrtuzation density to get more point near the bed than near the the surface. 1.0 means equal vertical spacing.",
    )
    parser.add_argument(
        "--iflo_thr_ice_thk",
        type=float,
        default=0.1,
        help="Threshold Ice thickness for computing strain rate",
    )

    # solver parameters
    parser.add_argument(
        "--iflo_solve_step_size",
        type=float,
        default=1,
        help="Step size for the optimizer using when solving Blatter-Pattyn in solver mode",
    )
    parser.add_argument(
        "--iflo_solve_nbitmax",
        type=int,
        default=100,
        help="Maximum number of iteration for the optimizer using when solving Blatter-Pattyn in solver mode",
    )
    parser.add_argument(
        "--iflo_solve_stop_if_no_decrease",
        type=str2bool,
        default=True,
        help="This permits to stop the solver if the energy does not decrease",
    )

    # emualtion parameters
    parser.add_argument(
        "--iflo_fieldin",
        type=list,
        default=["thk", "usurf", "arrhenius", "slidingco", "dX"],
        help="Input fields of the iceflow emulator",
    )
    parser.add_argument(
        "--iflo_dim_arrhenius",
        type=int,
        default=2,
        help="Dimension of the arrhenius factor (horizontal 2D or 3D)",
    )

    parser.add_argument(
        "--iflo_retrain_emulator_freq",
        type=int,
        default=10,
        help="Frequency at which the emulator is retrained, 0 means never, 1 means at each time step, 2 means every two time steps, etc.",
    )
    parser.add_argument(
        "--iflo_retrain_emulator_lr",
        type=float,
        default=0.00002,
        help="Learning rate for the retraining of the emulator",
    )
    parser.add_argument(
        "--iflo_retrain_emulator_nbit_init",
        type=float,
        default=1,
        help="Number of iterations done at the first time step for the retraining of the emulator",
    )
    parser.add_argument(
        "--iflo_retrain_emulator_nbit",
        type=float,
        default=1,
        help="Number of iterations done at each time step for the retraining of the emulator",
    )
    parser.add_argument(
        "--iflo_retrain_emulator_framesizemax",
        type=float,
        default=750,
        help="Size of the patch used for retraining the emulator, this is usefull for large size arrays, otherwise the GPU memory can be overloaded",
    )
    parser.add_argument(
        "--iflo_multiple_window_size",
        type=int,
        default=0,
        help="If a U-net, this force window size a multiple of 2**N",
    )
    parser.add_argument(
        "--iflo_force_max_velbar",
        type=float,
        default=0,
        help="This permits to artifically upper-bound velocities, active if > 0",
    )

    # CNN parameters
    parser.add_argument(
        "--iflo_network",
        type=str,
        default="cnn",
        help="This is the type of network, it can be cnn or unet",
    )
    parser.add_argument(
        "--iflo_activation",
        type=str,
        default="lrelu",
        help="Activation function, it can be lrelu, relu, tanh, sigmoid, etc.",
    )
    parser.add_argument(
        "--iflo_nb_layers",
        type=int,
        default=16,
        help="Number of layers in the CNN",
    )
    parser.add_argument(
        "--iflo_nb_blocks",
        type=int,
        default=4,
        help="Number of block layer in the U-net",
    )
    parser.add_argument(
        "--iflo_nb_out_filter",
        type=int,
        default=32,
        help="Number of output filters in the CNN",
    )
    parser.add_argument(
        "--iflo_conv_ker_size",
        type=int,
        default=3,
        help="Size of the convolution kernel",
    )
    parser.add_argument(
        "--iflo_dropout_rate",
        type=float,
        default=0,
        help="Dropout rate in the CNN",
    )
    parser.add_argument(
        "--iflo_exclude_borders",
        type=int,
        default=0,
        help="This is a quick fix of the border issue, other the physics informed emaulator shows zero velocity at the border",
    )
    
    parser.add_argument(
        "--iflo_cf_eswn",
        type=list,
        default=[],
        help="This forces calving front at the border of the domain in the side given in the list",
    )
    
    parser.add_argument(
        "--iflo_cf_cond",
        type=str2bool,
        default=False,
        help="This forces calving front at the border of the domain in the side given in the list",
    )
    
    parser.add_argument(
        "--iflo_regu",
        type=float,
        default=0.0,
        help="This regularizes the energy forcing ice flow to be smooth in the horizontal direction",
    )
    parser.add_argument(
        "--iflo_min_sr",
        type=float,
        default=10**(-20),
        help="Minimum strain rate",
    )
    parser.add_argument(
        "--iflo_max_sr",
        type=float,
        default=10**(20),
        help="Maximum strain rate",
    )




def initialize(params, state):
    # This makes it so that if the user included the optimize module, this intializer will not be called again.
    # This is due to the fact that the optimize module calls the initialize (and params) function of the iceflow module.
    if hasattr(state, "optimize_initializer_called"):
        return

    state.tcomp_iceflow = []

    # here we initialize variable parmaetrizing ice flow
    if not hasattr(state, "arrhenius"):
        if params.iflo_dim_arrhenius == 3:
            state.arrhenius = tf.Variable(
                tf.ones((params.iflo_Nz, state.thk.shape[0], state.thk.shape[1]))
                * params.iflo_init_arrhenius * params.iflo_enhancement_factor
            )
        else:
            state.arrhenius = tf.Variable(
                tf.ones_like(state.thk) * params.iflo_init_arrhenius * params.iflo_enhancement_factor
            )

    if not hasattr(state, "slidingco"):
        state.slidingco = tf.Variable(
            tf.ones_like(state.thk) * params.iflo_init_slidingco
        )

    # here we create a new velocity field
    if not hasattr(state, "U"):
        state.U = tf.Variable(
            tf.zeros((params.iflo_Nz, state.thk.shape[0], state.thk.shape[1]))
        )
        state.V = tf.Variable(
            tf.zeros((params.iflo_Nz, state.thk.shape[0], state.thk.shape[1]))
        )

    if not params.iflo_type == "solved":
        if int(tf.__version__.split(".")[1]) <= 10:
            state.opti_retrain = tf.keras.optimizers.Adam(
                learning_rate=params.iflo_retrain_emulator_lr
            )
        else:
            state.opti_retrain = tf.keras.optimizers.legacy.Adam(
                learning_rate=params.iflo_retrain_emulator_lr
            )

        direct_name = (
            "pinnbp"
            + "_"
            + str(params.iflo_Nz)
            + "_"
            + str(int(params.iflo_vert_spacing))
            + "_"
        )
        direct_name += (
            params.iflo_network
            + "_"
            + str(params.iflo_nb_layers)
            + "_"
            + str(params.iflo_nb_out_filter)
            + "_"
        )
        direct_name += (
            str(params.iflo_dim_arrhenius)
            + "_"
            + str(int(params.iflo_new_friction_param))
        )

        if params.iflo_pretrained_emulator:
            if params.iflo_emulator == "":
                if os.path.exists(
                    importlib_resources.files(emulators).joinpath(direct_name)
                ):
                    dirpath = importlib_resources.files(emulators).joinpath(direct_name)
                    print(
                        "Found pretrained emulator in the igm package: " + direct_name
                    )
                else:
                    print("No pretrained emulator found in the igm package")
            else:
                if os.path.exists(params.iflo_emulator):
                    dirpath = params.iflo_emulator
                    print("Found pretrained emulator: " + params.iflo_emulator)
                else:
                    print("No pretrained emulator found ")

            fieldin = []
            fid = open(os.path.join(dirpath, "fieldin.dat"), "r")
            for fileline in fid:
                part = fileline.split()
                fieldin.append(part[0])
            fid.close()
            assert params.iflo_fieldin == fieldin
            state.iceflow_model = tf.keras.models.load_model(
                os.path.join(dirpath, "model.h5"), compile=False
            )
            state.iceflow_model.compile()
        else:
            nb_inputs = len(params.iflo_fieldin) + (params.iflo_dim_arrhenius == 3) * (
                params.iflo_Nz - 1
            )
            nb_outputs = 2 * params.iflo_Nz
            # state.iceflow_model = getattr(igm, params.iflo_network)(
            #     params, nb_inputs, nb_outputs
            # )
            if params.iflo_network=='cnn':
                state.iceflow_model = cnn(params, nb_inputs, nb_outputs)
            elif params.iflo_network=='unet':
                state.iceflow_model = unet(params, nb_inputs, nb_outputs)

    if not params.iflo_type == "emulated":
        if int(tf.__version__.split(".")[1]) <= 10:
            state.optimizer = tf.keras.optimizers.Adam(
                learning_rate=params.iflo_solve_step_size
            )
        else:
            state.optimizer = tf.keras.optimizers.legacy.Adam(
                learning_rate=params.iflo_solve_step_size
            )

    # if we do disoangostic, one neds to create a solved solution
    if params.iflo_type == "diagnostic":
        state.UT = tf.Variable(
            tf.zeros((params.iflo_Nz, state.thk.shape[0], state.thk.shape[1]))
        )
        state.VT = tf.Variable(
            tf.zeros((params.iflo_Nz, state.thk.shape[0], state.thk.shape[1]))
        )

    # create the vertica discretization
    define_vertical_weight(params, state)

    Ny = state.thk.shape[0]
    Nx = state.thk.shape[1]

    # In case of a U-net, must make sure the I/O size is multiple of 2**N
    if params.iflo_multiple_window_size > 0:
        NNy = params.iflo_multiple_window_size * math.ceil(
            Ny / params.iflo_multiple_window_size
        )
        NNx = params.iflo_multiple_window_size * math.ceil(
            Nx / params.iflo_multiple_window_size
        )
        state.PAD = [[0, NNy - Ny], [0, NNx - Nx]]
    else:
        state.PAD = [[0, 0], [0, 0]]

    if not params.iflo_type == "solved":
        _update_iceflow_emulated(params, state)


def update(params, state):
    if hasattr(state, "logger"):
        state.logger.info("Update ICEFLOW at time : " + str(state.t.numpy()))

    state.tcomp_iceflow.append(time.time())

    if params.iflo_type == "emulated":
        if params.iflo_retrain_emulator_freq > 0:
            _update_iceflow_emulator(params, state)

        _update_iceflow_emulated(params, state)

    elif params.iflo_type == "solved":
        _update_iceflow_solved(params, state)

    elif params.iflo_type == "diagnostic":
        _update_iceflow_diagnostic(params, state)

    state.tcomp_iceflow[-1] -= time.time()
    state.tcomp_iceflow[-1] *= -1


def finalize(params, state):
    if params.iflo_save_model:
        save_iceflow_model(params, state)


########################################################################
########################################################################
##############  Definition of the system energy ########################
########################################################################
########################################################################


@tf.function(experimental_relax_shapes=True)
def _compute_gradient_stag(s, dX, dY):
    """
    compute spatial gradient, outcome on stagerred grid
    """

    E = 2.0 * (s[:, :, 1:] - s[:, :, :-1]) / (dX[:, :, 1:] + dX[:, :, :-1])
    diffx = 0.5 * (E[:, 1:, :] + E[:, :-1, :])

    EE = 2.0 * (s[:, 1:, :] - s[:, :-1, :]) / (dY[:, 1:, :] + dY[:, :-1, :])
    diffy = 0.5 * (EE[:, :, 1:] + EE[:, :, :-1])

    return diffx, diffy


@tf.function(experimental_relax_shapes=True)
def _compute_strainrate_Glen_tf(U, V, thk, slidingco, dX, ddz, sloptopgx, sloptopgy, thr):
    # Compute horinzontal derivatives
    dUdx = (U[:, :, :, 1:] - U[:, :, :, :-1]) / dX[0, 0, 0]
    dVdx = (V[:, :, :, 1:] - V[:, :, :, :-1]) / dX[0, 0, 0]
    dUdy = (U[:, :, 1:, :] - U[:, :, :-1, :]) / dX[0, 0, 0]
    dVdy = (V[:, :, 1:, :] - V[:, :, :-1, :]) / dX[0, 0, 0]

    # Homgenize sizes in the horizontal plan on the stagerred grid
    dUdx = (dUdx[:, :, :-1, :] + dUdx[:, :, 1:, :]) / 2
    dVdx = (dVdx[:, :, :-1, :] + dVdx[:, :, 1:, :]) / 2
    dUdy = (dUdy[:, :, :, :-1] + dUdy[:, :, :, 1:]) / 2
    dVdy = (dVdy[:, :, :, :-1] + dVdy[:, :, :, 1:]) / 2

    # homgenize sizes in the vertical plan on the stagerred grid
    if U.shape[1] > 1:
        dUdx = (dUdx[:, :-1, :, :] + dUdx[:, 1:, :, :]) / 2
        dVdx = (dVdx[:, :-1, :, :] + dVdx[:, 1:, :, :]) / 2
        dUdy = (dUdy[:, :-1, :, :] + dUdy[:, 1:, :, :]) / 2
        dVdy = (dVdy[:, :-1, :, :] + dVdy[:, 1:, :, :]) / 2

    # compute the horizontal average, these quantitites will be used for vertical derivatives
    Um = (U[:, :, 1:, 1:] + U[:, :, 1:, :-1] + U[:, :, :-1, 1:] + U[:, :, :-1, :-1]) / 4
    Vm = (V[:, :, 1:, 1:] + V[:, :, 1:, :-1] + V[:, :, :-1, 1:] + V[:, :, :-1, :-1]) / 4

    if U.shape[1] > 1:
        # vertical derivative if there is at least two layears
        dUdz = (Um[:, 1:, :, :] - Um[:, :-1, :, :]) / tf.maximum(ddz, thr)
        dVdz = (Vm[:, 1:, :, :] - Vm[:, :-1, :, :]) / tf.maximum(ddz, thr)
        slc = tf.expand_dims(_stag4(slidingco), axis=1)
        dUdz = tf.where(slc > 0, dUdz, 0.01 * dUdz)
        dVdz = tf.where(slc > 0, dVdz, 0.01 * dVdz)
    else:
        # zero otherwise
        dUdz = 0.0
        dVdz = 0.0

    # This correct for the change of coordinate z -> z - b
    dUdx = dUdx - dUdz * sloptopgx
    dUdy = dUdy - dUdz * sloptopgy
    dVdx = dVdx - dVdz * sloptopgx
    dVdy = dVdy - dVdz * sloptopgy

    Exx = dUdx
    Eyy = dVdy
    Ezz = -dUdx - dVdy
    Exy = 0.5 * dVdx + 0.5 * dUdy
    Exz = 0.5 * dUdz
    Eyz = 0.5 * dVdz
    
    srx = 0.5 * ( Exx**2 + Exy**2 + Exy**2 + Eyy**2 + Ezz**2 )
    srz = 0.5 * ( Exz**2 + Eyz**2 + Exz**2 + Eyz**2 )

    return srx, srz


def _stag2(B):
    return (B[:, 1:] + B[:, :1]) / 2


def _stag4(B):
    return (B[:, 1:, 1:] + B[:, 1:, :-1] + B[:, :-1, 1:] + B[:, :-1, :-1]) / 4


def _stag4b(B):
    return (
        B[:, :, 1:, 1:] + B[:, :, 1:, :-1] + B[:, :, :-1, 1:] + B[:, :, :-1, :-1]
    ) / 4


def _stag8(B):
    return (
        B[:, 1:, 1:, 1:]
        + B[:, 1:, 1:, :-1]
        + B[:, 1:, :-1, 1:]
        + B[:, 1:, :-1, :-1]
        + B[:, :-1, 1:, 1:]
        + B[:, :-1, 1:, :-1]
        + B[:, :-1, :-1, 1:]
        + B[:, :-1, :-1, :-1]
    ) / 8


def iceflow_energy(params, U, V, fieldin):
    thk, usurf, arrhenius, slidingco, dX = fieldin

    return _iceflow_energy(
        U,
        V,
        thk,
        usurf,
        arrhenius,
        slidingco,
        dX,
        params.iflo_Nz,
        params.iflo_vert_spacing,
        params.iflo_exp_glen,
        params.iflo_exp_weertman,
        params.iflo_regu_glen,
        params.iflo_regu_weertman,
        params.iflo_thr_ice_thk,
        params.iflo_ice_density,
        params.iflo_gravity_cst,
        params.iflo_new_friction_param,
        params.iflo_cf_cond,
        params.iflo_cf_eswn,
        params.iflo_regu,
        params.iflo_min_sr,
        params.iflo_max_sr
    )


@tf.function(experimental_relax_shapes=True)
def _iceflow_energy(
    U,
    V,
    thk,
    usurf,
    arrhenius,
    slidingco,
    dX,
    Nz,
    vert_spacing,
    exp_glen,
    exp_weertman,
    regu_glen,
    regu_weertman,
    thr_ice_thk,
    ice_density,
    gravity_cst,
    new_friction_param,
    iflo_cf_cond,
    iflo_cf_eswn,
    iflo_regu,
    min_sr,
    max_sr,
):
    # warning, the energy is here normalized dividing by int_Omega

    COND = (
        (thk[:, 1:, 1:] > 0)
        & (thk[:, 1:, :-1] > 0)
        & (thk[:, :-1, 1:] > 0)
        & (thk[:, :-1, :-1] > 0)
    )
    COND = tf.expand_dims(COND, axis=1)

    # Vertical discretization
    if Nz > 1:
        zeta = np.arange(Nz) / (Nz - 1)  # formerly ...
        #zeta = tf.range(Nz, dtype=tf.float32) / (Nz - 1)
        temp = (zeta / vert_spacing) * (1.0 + (vert_spacing - 1.0) * zeta)
        temd = temp[1:] - temp[:-1]
        dz = tf.stack([_stag4(thk) * z for z in temd], axis=1)  # formerly ..
        #dz = (tf.expand_dims(tf.expand_dims(temd,axis=-1),axis=-1)*tf.expand_dims(_stag4(thk),axis=0))
    else:
        dz = tf.expand_dims(_stag4(thk), axis=0)

    # B has Unit Mpa y^(1/n)
    B = 2.0 * arrhenius ** (-1.0 / exp_glen)

    if new_friction_param:
        C = 1.0 * slidingco  # C has unit Mpa y^m m^(-m)
    else:
        if exp_weertman == 1:
            # C has unit Mpa y^m m^(-m)
            C = 1.0 * slidingco
        else:
            C = (slidingco + 10 ** (-12)) ** -(1.0 / exp_weertman)

    p = 1.0 + 1.0 / exp_glen
    s = 1.0 + 1.0 / exp_weertman

    sloptopgx, sloptopgy = _compute_gradient_stag(usurf - thk, dX, dX)
    sloptopgx = tf.expand_dims(sloptopgx, axis=1)
    sloptopgy = tf.expand_dims(sloptopgy, axis=1)

    # TODO : sloptopgx, sloptopgy must be the elevaion of layers! not the bedrock, this probably has very little effects.

    # sr has unit y^(-1)
    srx, srz = _compute_strainrate_Glen_tf(
        U, V, thk, C, dX, dz, sloptopgx, sloptopgy, thr=thr_ice_thk
    )
    
    sr = srx + srz

    sr = tf.where(COND, sr, 0.0)
    
    srcapped = tf.clip_by_value(sr, min_sr**2, max_sr**2)

    srcapped = tf.where(COND, srcapped, 0.0)
 

    # C_shear is unit  Mpa y^(1/n) y^(-1-1/n) * m^3 = Mpa y^(-1) m^3
    if len(B.shape) == 3:
        C_shear = (
            tf.reduce_mean(
                _stag4(B)
                * tf.reduce_sum(dz * ((srcapped + regu_glen**2) ** ((p-2) / 2)) * sr, axis=1),
                axis=(-1, -2),
            )
            / p
        )
    else:
        C_shear = (
            tf.reduce_mean(
                tf.reduce_sum(
                    _stag8(B) * dz * ((srcapped + regu_glen**2) ** ((p-2) / 2)) * sr, axis=1
                ),
                axis=(-1, -2),
            )
            / p
        )
        
    if iflo_regu > 0:
        
        srx = tf.where(COND, srx, 0.0)
 
        if len(B.shape) == 3:
            C_shear_2 = ( 
                tf.reduce_mean(
                    _stag4(B)
                    * tf.reduce_sum(dz * ((srx + regu_glen**2) ** (p / 2)), axis=1),
                    axis=(-1, -2),
                )
                / p
            )
        else:
            C_shear_2 = ( 
                tf.reduce_mean(
                    tf.reduce_sum(
                        _stag8(B) * dz * ((srx + regu_glen**2) ** (p / 2)), axis=1
                    ),
                    axis=(-1, -2),
                )
                / p
            )

        C_shear = C_shear + iflo_regu*C_shear_2
        
    lsurf = usurf - thk

    sloptopgx, sloptopgy = _compute_gradient_stag(lsurf, dX, dX)

    # C_slid is unit Mpa y^m m^(-m) * m^(1+m) * y^(-1-m) * m*2 = Mpa y^(-1) m^3
    N = (
        _stag4(U[:, 0, :, :] ** 2 + V[:, 0, :, :] ** 2)
        + regu_weertman**2
        + (_stag4(U[:, 0, :, :]) * sloptopgx + _stag4(V[:, 0, :, :]) * sloptopgy) ** 2
    )
    C_slid = tf.reduce_mean(_stag4(C) * N ** (s / 2), axis=(-1, -2)) / s

    slopsurfx, slopsurfy = _compute_gradient_stag(usurf, dX, dX)
    slopsurfx = tf.expand_dims(slopsurfx, axis=1)
    slopsurfy = tf.expand_dims(slopsurfy, axis=1)

    if Nz > 1:
        uds = _stag8(U) * slopsurfx + _stag8(V) * slopsurfy
    else:
        uds = _stag4b(U) * slopsurfx + _stag4b(V) * slopsurfy

    uds = tf.where(COND, uds, 0.0)

    C_grav = (
        ice_density
        * gravity_cst
        * 10 ** (-6)
        * tf.reduce_mean(tf.reduce_sum(dz * uds, axis=1), axis=(-1, -2))
    )

    # if activae this applies the stress condition along the calving front
    if iflo_cf_cond:
        
        lsurf = usurf - thk
        
    #   Check formula (17) in [Jouvet and Graeser 2012]
        P = tf.where(lsurf<0, 0.5 * 10 ** (-6) * 9.81 * 910 * ( thk**2 - (1000/910)*lsurf**2 ) , 0.0)
        
        if len(iflo_cf_eswn) == 0:
            thkext = tf.pad(thk,[[0,0],[1,1],[1,1]],"CONSTANT",constant_values=1)
        else:
            thkext = thk
            thkext = tf.pad(thkext,[[0,0],[1,0],[0,0]],"CONSTANT",constant_values=1.0*('S' not in iflo_cf_eswn))
            thkext = tf.pad(thkext,[[0,0],[0,1],[0,0]],"CONSTANT",constant_values=1.0*('N' not in iflo_cf_eswn))
            thkext = tf.pad(thkext,[[0,0],[0,0],[1,0]],"CONSTANT",constant_values=1.0*('W' not in iflo_cf_eswn))
            thkext = tf.pad(thkext,[[0,0],[0,0],[0,1]],"CONSTANT",constant_values=1.0*('E' not in iflo_cf_eswn)) 
        
        # this permits to locate the calving front in a cell in the 4 directions
        CF_W = tf.where((thk>0)&(thkext[:,1:-1,:-2]==0),1.0,0.0)
        CF_E = tf.where((thk>0)&(thkext[:,1:-1,2:]==0),1.0,0.0) 
        CF_S = tf.where((thk>0)&(thkext[:,:-2,1:-1]==0),1.0,0.0)
        CF_N = tf.where((thk>0)&(thkext[:,2:,1:-1]==0),1.0,0.0)

        if Nz > 1:
            # Blatter-Pattyn
            dz0 = tf.stack([tf.ones_like(thk) * z for z in temd], axis=1)
            C_front = (
                tf.reduce_sum( P * tf.reduce_sum(dz0 * _stag2(U), axis=1) * CF_W, axis=(-2,-1) ) 
                - tf.reduce_sum( P * tf.reduce_sum(dz0 * _stag2(U), axis=1) * CF_E, axis=(-2,-1) ) 
                + tf.reduce_sum( P * tf.reduce_sum(dz0 * _stag2(V), axis=1) * CF_S, axis=(-2,-1) ) 
                - tf.reduce_sum( P * tf.reduce_sum(dz0 * _stag2(V), axis=1) * CF_N, axis=(-2,-1) ) 
            ) * dX[:, 0, 0]
        else:
            # SSA
            C_front = (
                tf.reduce_sum(P * U * CF_W, axis=(-2,-1) ) 
                - tf.reduce_sum(P * U * CF_E, axis=(-2,-1) ) 
                + tf.reduce_sum(P * V * CF_S, axis=(-2,-1) ) 
                - tf.reduce_sum(P * V * CF_N, axis=(-2,-1) ) 
            ) * dX[:, 0, 0]
    
        C_front = C_front / ( tf.reduce_sum(tf.ones_like(thk),axis=(-2,-1)) * dX[:, 0, 0]**2 )
        
    else:
        C_front = tf.zeros_like(C_shear)

#    print(C_shear[0].numpy(),C_slid[0].numpy(),C_grav[0].numpy(),C_front[0].numpy())

    return tf.reduce_mean(C_shear + C_slid + C_grav + C_front)


# @tf.function(experimental_relax_shapes=True)
def iceflow_energy_XY(params, X, Y):
    U, V = Y_to_UV(params, Y)

    fieldin = X_to_fieldin(params, X)

    return iceflow_energy(params, U, V, fieldin)


def Y_to_UV(params, Y):
    N = params.iflo_Nz

    U = tf.experimental.numpy.moveaxis(Y[:, :, :, :N], [-1], [1])
    V = tf.experimental.numpy.moveaxis(Y[:, :, :, N:], [-1], [1])

    return U, V


def UV_to_Y(params, U, V):
    UU = tf.experimental.numpy.moveaxis(U, [0], [-1])
    VV = tf.experimental.numpy.moveaxis(V, [0], [-1])
    RR = tf.expand_dims(
        tf.concat(
            [UU, VV],
            axis=-1,
        ),
        axis=0,
    )

    return RR


def fieldin_to_X(params, fieldin):
    X = []

    fieldin_dim = [0, 0, 1 * (params.iflo_dim_arrhenius == 3), 0, 0]

    for f, s in zip(fieldin, fieldin_dim):
        if s == 0:
            X.append(tf.expand_dims(f, axis=-1))
        else:
            X.append(tf.experimental.numpy.moveaxis(f, [0], [-1]))

    return tf.expand_dims(tf.concat(X, axis=-1), axis=0)


def X_to_fieldin(params, X):
    i = 0

    fieldin_dim = [0, 0, 1 * (params.iflo_dim_arrhenius == 3), 0, 0]

    fieldin = []

    for f, s in zip(params.iflo_fieldin, fieldin_dim):
        if s == 0:
            fieldin.append(X[:, :, :, i])
            i += 1
        else:
            fieldin.append(
                tf.experimental.numpy.moveaxis(
                    X[:, :, :, i : i + params.iflo_Nz], [-1], [1]
                )
            )
            i += params.iflo_Nz

    return fieldin


########################################################################
########################################################################
#####################  Vertical discretization #########################
########################################################################
########################################################################


def define_vertical_weight(params, state):
    """
    define_vertical_weight
    """

    zeta = np.arange(params.iflo_Nz + 1) / params.iflo_Nz
    weight = (zeta / params.iflo_vert_spacing) * (
        1.0 + (params.iflo_vert_spacing - 1.0) * zeta
    )
    weight = tf.Variable(weight[1:] - weight[:-1], dtype=tf.float32)
    state.vert_weight = tf.expand_dims(tf.expand_dims(weight, axis=-1), axis=-1)


########################################################################
########################################################################
#####################  Solver routines   ###############################
########################################################################
########################################################################


def solve_iceflow(params, state, U, V):
    """
    solve_iceflow
    """

    Cost_Glen = []

    for i in range(params.iflo_solve_nbitmax):
        with tf.GradientTape() as t:
            t.watch(U)
            t.watch(V)

            fieldin = [
                tf.expand_dims(vars(state)[f], axis=0) for f in params.iflo_fieldin
            ]

            COST = iceflow_energy(
                params, tf.expand_dims(U, axis=0), tf.expand_dims(V, axis=0), fieldin
            )

            Cost_Glen.append(COST)

            # Stop if the cost no longer decreases
            if params.iflo_solve_stop_if_no_decrease:
                if i > 1:
                    if Cost_Glen[-1] >= Cost_Glen[-2]:
                        break

            grads = tf.Variable(t.gradient(COST, [U, V]))

            state.optimizer.apply_gradients(
                zip([grads[i] for i in range(grads.shape[0])], [U, V])
            )

            if (i + 1) % 100 == 0:
                velsurf_mag = tf.sqrt(U[-1] ** 2 + V[-1] ** 2)
                print("solve :", i, COST.numpy(), np.max(velsurf_mag))

    U = tf.where(state.thk > 0, U, 0)
    V = tf.where(state.thk > 0, V, 0)

    return U, V, Cost_Glen


def _update_iceflow_solved(params, state):
    U, V, Cost_Glen = solve_iceflow(params, state, state.U, state.V)

    state.U.assign(U)
    state.V.assign(V)
    
    if params.iflo_force_max_velbar > 0:
        velbar_mag = getmag3d(state.U, state.V)
        state.U.assign(
            tf.where(
                velbar_mag >= params.iflo_force_max_velbar,
                params.iflo_force_max_velbar * (state.U / velbar_mag),
                state.U,
            )
        )
        state.V.assign(
            tf.where(
                velbar_mag >= params.iflo_force_max_velbar,
                params.iflo_force_max_velbar * (state.V / velbar_mag),
                state.V,
            )
        )

    state.COST_Glen = Cost_Glen[-1].numpy()

    update_2d_iceflow_variables(params, state)


def update_2d_iceflow_variables(params, state):
    state.uvelbase = state.U[0, :, :]
    state.vvelbase = state.V[0, :, :]
    state.ubar = tf.reduce_sum(state.U * state.vert_weight, axis=0)
    state.vbar = tf.reduce_sum(state.V * state.vert_weight, axis=0)
    state.uvelsurf = state.U[-1, :, :]
    state.vvelsurf = state.V[-1, :, :]


########################################################################
########################################################################
#####################  Emulation routines   ############################
########################################################################
########################################################################


def _update_iceflow_emulated(params, state):
    # Define the input of the NN, include scaling

    fieldin = [vars(state)[f] for f in params.iflo_fieldin]

    X = fieldin_to_X(params, fieldin)

    if params.iflo_exclude_borders>0:
        iz = params.iflo_exclude_borders
        X = tf.pad(X, [[0, 0], [iz, iz], [iz, iz], [0, 0]], "SYMMETRIC")

    Y = state.iceflow_model(X)

    if params.iflo_exclude_borders>0:
        iz = params.iflo_exclude_borders
        Y = Y[:, iz:-iz, iz:-iz, :]

    Ny, Nx = state.thk.shape
    N = params.iflo_Nz

    U, V = Y_to_UV(params, Y[:, :Ny, :Nx, :])
    U = U[0]
    V = V[0]

    #    U = tf.where(state.thk > 0, U, 0)

    state.U.assign(U)
    state.V.assign(V)

    # If requested, the speeds are artifically upper-bounded
    if params.iflo_force_max_velbar > 0:
        velbar_mag = getmag3d(state.U, state.V)
        state.U.assign(
            tf.where(
                velbar_mag >= params.iflo_force_max_velbar,
                params.iflo_force_max_velbar * (state.U / velbar_mag),
                state.U,
            )
        )
        state.V.assign(
            tf.where(
                velbar_mag >= params.iflo_force_max_velbar,
                params.iflo_force_max_velbar * (state.V / velbar_mag),
                state.V,
            )
        )

    update_2d_iceflow_variables(params, state)


def _update_iceflow_emulator(params, state):
    if (state.it < 0) | (state.it % params.iflo_retrain_emulator_freq == 0):
        fieldin = [vars(state)[f] for f in params.iflo_fieldin]

        XX = fieldin_to_X(params, fieldin)

        X = _split_into_patches(XX, params.iflo_retrain_emulator_framesizemax)

        state.COST_EMULATOR = []

        nbit = (state.it >= 0) * params.iflo_retrain_emulator_nbit + (
            state.it < 0
        ) * params.iflo_retrain_emulator_nbit_init

        iz = params.iflo_exclude_borders 

        for epoch in range(nbit):
            cost_emulator = tf.Variable(0.0)

            for i in range(X.shape[0]):
                with tf.GradientTape() as t:
                    Y = state.iceflow_model(X[i : i + 1, :, :, :])

                    if iz>0:
                        COST = iceflow_energy_XY(params, X[i : i + 1, iz:-iz, iz:-iz, :], Y[:, iz:-iz, iz:-iz, :])
                    else:
                        COST = iceflow_energy_XY(params, X[i : i + 1, :, :, :], Y[:, :, :, :])

                    cost_emulator = cost_emulator + COST

                    if (epoch + 1) % 100 == 0:
                        U, V = Y_to_UV(params, Y)
                        U = U[0]
                        V = V[0]
                        velsurf_mag = tf.sqrt(U[-1] ** 2 + V[-1] ** 2)
                        print("train : ", epoch, COST.numpy(), np.max(velsurf_mag))

                grads = t.gradient(COST, state.iceflow_model.trainable_variables)

                state.opti_retrain.apply_gradients(
                    zip(grads, state.iceflow_model.trainable_variables)
                )

                state.opti_retrain.lr = params.iflo_retrain_emulator_lr * (
                    0.95 ** (epoch / 1000)
                )

            state.COST_EMULATOR.append(cost_emulator)


def _split_into_patches(X, nbmax):
    XX = []
    ny = X.shape[1]
    nx = X.shape[2]
    sy = ny // nbmax + 1
    sx = nx // nbmax + 1
    ly = int(ny / sy)
    lx = int(nx / sx)

    for i in range(sx):
        for j in range(sy):
            XX.append(X[0, j * ly : (j + 1) * ly, i * lx : (i + 1) * lx, :])

    return tf.stack(XX, axis=0)


########################################################################
########################################################################
#####################  Diagnostic routines   ###########################
########################################################################
########################################################################


def _update_iceflow_diagnostic(params, state):
    if params.iflo_retrain_emulator_freq > 0:
        _update_iceflow_emulator(params, state)
        COST_Emulator = state.COST_EMULATOR[-1].numpy()
    else:
        COST_Emulator = 0.0

    _update_iceflow_emulated(params, state)

    if state.it % 10 == 0:
        UT, VT, Cost_Glen = solve_iceflow(params, state, state.UT, state.VT)
        state.UT.assign(UT)
        state.VT.assign(VT)
        COST_Glen = Cost_Glen[-1].numpy()

        print("nb solve iterations :", len(Cost_Glen))

        l1, l2 = computemisfit(state, state.thk, state.U - state.UT, state.V - state.VT)

        ERR = [state.t.numpy(), COST_Glen, COST_Emulator, l1, l2]

        print(ERR)

        with open("errors.txt", "ab") as f:
            np.savetxt(f, np.expand_dims(ERR, axis=0), delimiter=",", fmt="%5.5f")


def computemisfit(state, thk, U, V):
    ubar = tf.reduce_sum(state.vert_weight * U, axis=0)
    vbar = tf.reduce_sum(state.vert_weight * V, axis=0)

    VEL = tf.stack([ubar, vbar], axis=0)
    MA = tf.where(thk > 1, tf.ones_like(VEL), 0)
    # MA = tf.where(state.thk > 1, tf.ones_like(VEL), 0)

    nl1diff = tf.reduce_sum(MA * tf.abs(VEL)) / tf.reduce_sum(MA)
    nl2diff = tf.reduce_sum(MA * tf.abs(VEL) ** 2) / tf.reduce_sum(MA)

    return nl1diff.numpy(), np.sqrt(nl2diff)


########################################################################
########################################################################
#####################  Definiton of neural networks ####################
########################################################################
########################################################################


def cnn(params, nb_inputs, nb_outputs):
    """
    Routine serve to build a convolutional neural network
    """

    inputs = tf.keras.layers.Input(shape=[None, None, nb_inputs])

    conv = inputs

    if params.iflo_activation == "lrelu":
        activation = tf.keras.layers.LeakyReLU(alpha=0.01)
    else:
        activation = tf.keras.layers.ReLU()

    for i in range(int(params.iflo_nb_layers)):
        conv = tf.keras.layers.Conv2D(
            filters=params.iflo_nb_out_filter,
            kernel_size=(params.iflo_conv_ker_size, params.iflo_conv_ker_size),
            padding="same",
        )(conv)

        conv = activation(conv)

        conv = tf.keras.layers.Dropout(params.iflo_dropout_rate)(conv)

    outputs = conv

    outputs = tf.keras.layers.Conv2D(
        filters=nb_outputs,
        kernel_size=(
            1,
            1,
        ),
        activation=None,
    )(outputs)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs)


def unet(params, nb_inputs, nb_outputs):
    """
    Routine serve to define a UNET network from keras_unet_collection
    """

    from keras_unet_collection import models

    layers = np.arange(int(params.iflo_nb_blocks))

    number_of_filters = [
        params.iflo_nb_out_filter * 2 ** (layers[i]) for i in range(len(layers))
    ]

    return models.unet_2d(
        (None, None, nb_inputs),
        number_of_filters,
        n_labels=nb_outputs,
        stack_num_down=2,
        stack_num_up=2,
        activation="LeakyReLU",
        output_activation=None,
        batch_norm=False,
        pool="max",
        unpool=False,
        name="unet",
    )


########################################################################
########################################################################
#####################  MISC  / UTIL ROUTINES ###########################
########################################################################
########################################################################


def save_iceflow_model(params, state):
    directory = "iceflow-model"

    if os.path.exists(directory):
        shutil.rmtree(directory)

    os.mkdir(directory)

    state.iceflow_model.save(os.path.join(directory, "model.h5"))

    #    fieldin_dim=[0,0,1*(params.iflo_dim_arrhenius==3),0,0]

    fid = open(os.path.join(directory, "fieldin.dat"), "w")
    #    for key,gg in zip(params.iflo_fieldin,fieldin_dim):
    #        fid.write("%s %.1f \n" % (key, gg))
    for key in params.iflo_fieldin:
        print(key)
        fid.write("%s \n" % (key))
    fid.close()

    fid = open(os.path.join(directory, "vert_grid.dat"), "w")
    fid.write("%4.0f  %s \n" % (params.iflo_Nz, "# number of vertical grid point (Nz)"))
    fid.write(
        "%2.2f  %s \n"
        % (params.iflo_vert_spacing, "# param for vertical spacing (vert_spacing)")
    )
    fid.close()
