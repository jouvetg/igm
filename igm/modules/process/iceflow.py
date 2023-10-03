#!/usr/bin/env python3

# Copyright (C) 2021-2023 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3), check at the LICENSE file 

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

def params_iceflow(parser):
    
    # type of ice flow computations
    parser.add_argument(
        "--type_iceflow",
        type=str,
        default="emulated",
        help="Type of iceflow: it can emulated (default), solved, or in diagnostic mode to investigate the fidelity of the emulator towads the solver",
    )

    parser.add_argument(
        "--emulator",
        type=str,
        default="myemulator",
        help="Directory path of the deep-learning ice flow model, create a new if empty string",
    )

    # type of ice flow model
    parser.add_argument(
        "--iceflow_physics", 
        type=int, 
        default=2,
        help="2 for blatter, 4 for stokes, this is also the number of DOF (STOKES DOES NOT WORK YET, KEEP IT TO 2)"
    )

    # physical parameters
    parser.add_argument(
        "--init_slidingco",
        type=float,
        default=10000,
        help="Initial sliding coefficient slidingco",
    )
    parser.add_argument(
        "--init_arrhenius",
        type=float,
        default=78,
        help="Initial arrhenius factor arrhenuis",
    )
    parser.add_argument(
        "--regu_glen",
        type=float,
        default=10 ** (-5),
        help="Regularization parameter for Glen's flow law",
    )
    parser.add_argument(
        "--regu_weertman",
        type=float,
        default=10 ** (-10),
        help="Regularization parameter for Weertman's sliding law",
    )
    parser.add_argument(
        "--exp_glen",
        type=float,
        default=3,
        help="Glen's flow law exponent",
    )
    parser.add_argument(
        "--exp_weertman", 
        type=float, 
        default=3, 
        help="Weertman's law exponent"
    )
    parser.add_argument(
        "--gravity_cst",
        type=float,
        default=9.81,
        help="Gravitational constant",
    )
    parser.add_argument(
        "--ice_density",
        type=float,
        default=910,
        help="Density of ice",
    )
    parser.add_argument(
        "--new_friction_param",
        type=str2bool,
        default=False,
        help="ExperimentaL: this describe slidingco differently with slidingco**-(1.0 / exp_weertman) instead of slidingco",
    )

    # vertical discretization
    parser.add_argument(
        "--Nz",
        type=int,
        default=10,
        help="Number of grid point for the vertical discretization",
    )
    parser.add_argument(
        "--vert_spacing",
        type=float,
        default=4.0,
        help="Parameter controlling the discrtuzation density to get more point near the bed than near the the surface. 1.0 means equal vertical spacing.",
    )
    parser.add_argument(
        "--thr_ice_thk",
        type=float,
        default=0.1,
        help="Threshold Ice thickness for computing strain rate",
    )

    # solver parameters
    parser.add_argument(
        "--solve_iceflow_step_size",
        type=float,
        default=1,
        help="Step size for the optimizer using when solving Blatter-Pattyn in solver mode",
    )
    parser.add_argument(
        "--solve_iceflow_nbitmax",
        type=int,
        default=100,
        help="Maximum number of iteration for the optimizer using when solving Blatter-Pattyn in solver mode",
    )
    parser.add_argument(
        "--stop_if_no_decrease",
        type=str2bool,
        default=True,
        help="This permits to stop the solver if the energy does not decrease",
    )

    # emualtion parameters
    parser.add_argument(
        "--fieldin",
        type=list,
        default=["thk", "usurf", "arrhenius", "slidingco", "dX"],
        help="Input fields of the iceflow emulator",
    )    
    parser.add_argument(
        "--dim_arrhenius",
        type=int,
        default=2,
        help="Dimension of the arrhenius factor (horizontal 2D or 3D)",
    )


    parser.add_argument(
        "--retrain_iceflow_emulator_freq",
        type=int,
        default=10,
        help="Frequency at which the emulator is retrained, 0 means never, 1 means at each time step, 2 means every two time steps, etc.",
    )
    parser.add_argument(
        "--retrain_iceflow_emulator_lr",
        type=float,
        default=0.00002,
        help="Learning rate for the retraining of the emulator",
    )
    parser.add_argument(
        "--retrain_iceflow_emulator_nbit_init",
        type=float,
        default=1,
        help="Number of iterations done at the first time step for the retraining of the emulator",
    )
    parser.add_argument(
        "--retrain_iceflow_emulator_nbit",
        type=float,
        default=1,
        help="Number of iterations done at each time step for the retraining of the emulator",
    )
    parser.add_argument(
        "--retrain_iceflow_emulator_framesizemax",
        type=float,
        default=750,
        help="Size of the patch used for retraining the emulator, this is usefull for large size arrays, otherwise the GPU memory can be overloaded",
    )
    parser.add_argument(
        "--multiple_window_size",
        type=int,
        default=0,
        help="If a U-net, this force window size a multiple of 2**N",
    )
    parser.add_argument(
        "--force_max_velbar",
        type=float,
        default=0,
        help="This permits to artifically upper-bound velocities, active if > 0",
    )

    # CNN parameters
    parser.add_argument(
        "--network",
        type=str,
        default="cnn",
        help="This is the type of network, it can be cnn or unet",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="lrelu",
        help="Activation function, it can be lrelu, relu, tanh, sigmoid, etc.",
    )
    parser.add_argument(
        "--nb_layers",
        type=int,
        default=16,
        help="Number of layers in the CNN",
    )
    parser.add_argument(
        "--nb_blocks",
        type=int,
        default=4,
        help="Number of block layer in the U-net",
    )
    parser.add_argument(
        "--nb_out_filter",
        type=int,
        default=32,
        help="Number of output filters in the CNN",
    )
    parser.add_argument(
        "--conv_ker_size",
        type=int,
        default=3,
        help="Size of the convolution kernel",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0,
        help="Dropout rate in the CNN",
    )
    parser.add_argument(
        "--exclude_borders_from_iceflow",
        type=str2bool,
        default=False,
        help="This is a quick fix of the border issue, other the physics informed emaulator shows zero velocity at the border",
    )  

def initialize_iceflow(params, state):

    state.tcomp_iceflow = []

    # here we initialize variable parmaetrizing ice flow
    if not hasattr(state, "arrhenius"):
        
        if params.dim_arrhenius==3:
            state.arrhenius = tf.Variable( \
                tf.ones((params.Nz,state.thk.shape[0],state.thk.shape[1])) \
                                          * params.init_arrhenius )
        else:
            state.arrhenius = tf.Variable(tf.ones_like(state.thk) * params.init_arrhenius)

    if not hasattr(state, "slidingco"):
        state.slidingco = tf.Variable(tf.ones_like(state.thk) * params.init_slidingco)

    # here we create a new velocity field
    if not hasattr(state, "U"):
        state.U = tf.Variable(
            tf.zeros(
                (
                    params.iceflow_physics,
                    params.Nz,
                    state.thk.shape[0],
                    state.thk.shape[1],
                )
            )
        )

    if not params.type_iceflow == "solved":
        if int(tf.__version__.split(".")[1]) <= 10:
            state.opti_retrain = tf.keras.optimizers.Adam(
                learning_rate=params.retrain_iceflow_emulator_lr
            )
        else:
            state.opti_retrain = tf.keras.optimizers.legacy.Adam(
                learning_rate=params.retrain_iceflow_emulator_lr
            )

        direct_name  = 'pinnbp'+'_'+str(params.Nz)+'_'+str(int(params.vert_spacing))+'_'
        direct_name +=  params.network+'_'+str(params.nb_layers)+'_'
        direct_name +=  str(params.dim_arrhenius)+'_'+str(int(params.new_friction_param))
        
        existing_emulator = True
        
        # first check if it finds a pretrained emulator in the igm package
        if os.path.exists(importlib_resources.files(emulators).joinpath(direct_name)):
            dirpath = importlib_resources.files(emulators).joinpath(direct_name)
            if hasattr(state,'logger'):
                state.logger.info("Found pretrained emulator in the igm package: "+direct_name)
        else:
            # if not, check if it finds a pretrained emulator in the current directory
            if os.path.exists(params.emulator):
                dirpath = params.emulator
                if hasattr(state,'logger'):
                    state.logger.info("Found pretrained emulator: "+params.emulator)
            # if not, create a new one from scratch
            else:
                existing_emulator = False
                if hasattr(state,'logger'):
                    state.logger.info("No pretrained emulator found, creating a new one from scratch")

        if existing_emulator:
            fieldin = []
            fid = open(os.path.join(dirpath, "fieldin.dat"), "r")
            for fileline in fid:
                part = fileline.split()
                fieldin.append(part[0])
            fid.close()
            assert params.fieldin == fieldin 
            state.iceflow_model = tf.keras.models.load_model( os.path.join(dirpath, "model.h5") , compile=False)
            state.iceflow_model.compile()
        else:
            nb_inputs  = len(params.fieldin) + (params.dim_arrhenius==3)*(params.Nz-1)
            nb_outputs = params.iceflow_physics * params.Nz
            state.iceflow_model = getattr(igm, params.network)(params, nb_inputs, nb_outputs)

    if not params.type_iceflow == "emulated":
        if int(tf.__version__.split(".")[1]) <= 10:
            state.optimizer = tf.keras.optimizers.Adam(
                learning_rate=params.solve_iceflow_step_size
            )
        else:
            state.optimizer = tf.keras.optimizers.legacy.Adam(
                learning_rate=params.solve_iceflow_step_size
            )

    # if we do disoangostic, one neds to create a solved solution
    if params.type_iceflow == "diagnostic":
        state.UT = tf.Variable(
            tf.zeros(
                (
                    params.iceflow_physics,
                    params.Nz,
                    state.thk.shape[0],
                    state.thk.shape[1],
                )
            )
        )

    # create the vertica discretization
    define_vertical_weight(params, state)

    Ny = state.thk.shape[0]
    Nx = state.thk.shape[1]

    # In case of a U-net, must make sure the I/O size is multiple of 2**N
    if params.multiple_window_size > 0:
        NNy = params.multiple_window_size * math.ceil(Ny / params.multiple_window_size)
        NNx = params.multiple_window_size * math.ceil(Nx / params.multiple_window_size)
        state.PAD = [[0, NNy - Ny], [0, NNx - Nx]]
    else:
        state.PAD = [[0, 0], [0, 0]]
        
    _update_iceflow_emulated(params, state)

def update_iceflow(params, state):

    if hasattr(state,'logger'):
        state.logger.info("Update ICEFLOW at time : " + str(state.t.numpy()))

    state.tcomp_iceflow.append(time.time())

    if params.type_iceflow == "emulated":
        if params.retrain_iceflow_emulator_freq > 0:
            _update_iceflow_emulator(params, state)

        _update_iceflow_emulated(params, state)

    elif params.type_iceflow == "solved":
        _update_iceflow_solved(params, state)

    elif params.type_iceflow == "diagnostic":
        _update_iceflow_diagnostic(params, state)

    state.tcomp_iceflow[-1] -= time.time()
    state.tcomp_iceflow[-1] *= -1


def finalize_iceflow(params, state):
    pass 

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
def _compute_strainrate_Glen_tf(U, thk, dX, ddz, sloptopgx, sloptopgy, thr):
    # Compute horinzontal derivatives
    dUdx = (U[:, 0, :, :, 1:] - U[:, 0, :, :, :-1]) / dX[0, 0, 0]
    dVdx = (U[:, 1, :, :, 1:] - U[:, 1, :, :, :-1]) / dX[0, 0, 0]
    dUdy = (U[:, 0, :, 1:, :] - U[:, 0, :, :-1, :]) / dX[0, 0, 0]
    dVdy = (U[:, 1, :, 1:, :] - U[:, 1, :, :-1, :]) / dX[0, 0, 0]

    # Homgenize sizes in the horizontal plan on the stagerred grid
    dUdx = (dUdx[:, :, :-1, :] + dUdx[:, :, 1:, :]) / 2
    dVdx = (dVdx[:, :, :-1, :] + dVdx[:, :, 1:, :]) / 2
    dUdy = (dUdy[:, :, :, :-1] + dUdy[:, :, :, 1:]) / 2
    dVdy = (dVdy[:, :, :, :-1] + dVdy[:, :, :, 1:]) / 2

    # homgenize sizes in the vertical plan on the stagerred grid
    if U.shape[2] > 1:
        dUdx = (dUdx[:, :-1, :, :] + dUdx[:, 1:, :, :]) / 2
        dVdx = (dVdx[:, :-1, :, :] + dVdx[:, 1:, :, :]) / 2
        dUdy = (dUdy[:, :-1, :, :] + dUdy[:, 1:, :, :]) / 2
        dVdy = (dVdy[:, :-1, :, :] + dVdy[:, 1:, :, :]) / 2

    # compute the horizontal average, these quantitites will be used for vertical derivatives
    Um = (
        U[:, 0, :, 1:, 1:]
        + U[:, 0, :, 1:, :-1]
        + U[:, 0, :, :-1, 1:]
        + U[:, 0, :, :-1, :-1]
    ) / 4
    Vm = (
        U[:, 1, :, 1:, 1:]
        + U[:, 1, :, 1:, :-1]
        + U[:, 1, :, :-1, 1:]
        + U[:, 1, :, :-1, :-1]
    ) / 4

    if U.shape[2] > 1:
        # vertical derivative if there is at least two layears
        dUdz = (Um[:, 1:, :, :] - Um[:, :-1, :, :]) / tf.maximum(ddz, thr)
        dVdz = (Vm[:, 1:, :, :] - Vm[:, :-1, :, :]) / tf.maximum(ddz, thr)
    else:
        # zero otherwise
        dUdz = 0.0
        dVdz = 0.0

    # If Stokes (and not Blatter), one has to do the same with the 3rd component
    if U.shape[1] > 2:
        dWdx = (U[:, 2, :, :, 1:] - U[:, 2, :, :, :-1]) / dX[0, 0, 0]
        dWdy = (U[:, 2, :, 1:, :] - U[:, 2, :, :-1, :]) / dX[0, 0, 0]

        dWdx = (dWdx[:, :, :-1, :] + dWdx[:, :, 1:, :]) / 2
        dWdy = (dWdy[:, :, :, :-1] + dWdy[:, :, :, 1:]) / 2

        if U.shape[2] > 1:
            dWdx = (dWdx[:, :-1, :, :] + dWdx[:, 1:, :, :]) / 2
            dWdy = (dWdy[:, :-1, :, :] + dWdy[:, 1:, :, :]) / 2

        Wm = (
            U[:, 2, :, 1:, 1:]
            + U[:, 2, :, 1:, :-1]
            + U[:, 2, :, :-1, 1:]
            + U[:, 2, :, :-1, :-1]
        ) / 4

        if U.shape[2] > 1:
            dWdz = (Wm[:, 1:, :, :] - Wm[:, :-1, :, :]) / tf.maximum(ddz, thr)
        else:
            dWdz = 0.0

    # This correct for the change of coordinate z -> z - b
    dUdx = dUdx - dUdz * sloptopgx
    dUdy = dUdy - dUdz * sloptopgy
    dVdx = dVdx - dVdz * sloptopgx
    dVdy = dVdy - dVdz * sloptopgy

    if U.shape[1] > 2:
        dWdx = dWdx - dWdz * sloptopgx
        dWdy = dWdy - dWdz * sloptopgy

    # Get element of the matrix for Stokes
    if U.shape[1] > 2:
        Exx = dUdx
        Eyy = dVdy
        Ezz = dWdz
        Exy = 0.5 * dVdx + 0.5 * dUdy
        Exz = 0.5 * dUdz + 0.5 * dWdx
        Eyz = 0.5 * dVdz + 0.5 * dWdy
        DIV = Exx + Eyy + Ezz

    # Get element of the matrix for Blatter
    else:
        Exx = dUdx
        Eyy = dVdy
        Ezz = -dUdx - dVdy
        Exy = 0.5 * dVdx + 0.5 * dUdy
        Exz = 0.5 * dUdz
        Eyz = 0.5 * dVdz
        DIV = 0.0

    return (
        0.5
        * (
            Exx**2
            + Exy**2
            + Exz**2
            + Exy**2
            + Eyy**2
            + Eyz**2
            + Exz**2
            + Eyz**2
            + Ezz**2
        ),
        DIV,
    )


def _stag4(B):
    return (B[:, 1:, 1:] + B[:, 1:, :-1] + B[:, :-1, 1:] + B[:, :-1, :-1]) / 4


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

def iceflow_energy(params, U, fieldin):
    
    thk, usurf, arrhenius, slidingco, dX = fieldin
    
    return _iceflow_energy(U, thk, usurf, arrhenius, slidingco, dX,
                           params.Nz, params.vert_spacing, 
                           params.exp_glen, params.exp_weertman, 
                           params.regu_glen, params.regu_weertman,
                           params.thr_ice_thk, params.iceflow_physics,
                           params.ice_density, params.gravity_cst, 
                           params.new_friction_param)
     
@tf.function(experimental_relax_shapes=True)
def _iceflow_energy(U, thk, usurf, arrhenius, slidingco, dX,
                    Nz, vert_spacing, exp_glen, exp_weertman, 
                    regu_glen, regu_weertman, thr_ice_thk, 
                    iceflow_physics, ice_density, gravity_cst, 
                    new_friction_param):
    
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
        zeta = np.arange(Nz) / (Nz - 1)
        temp = (zeta / vert_spacing) * (1.0 + (vert_spacing - 1.0) * zeta)
        temd = temp[1:] - temp[:-1]
        dz = tf.stack([_stag4(thk) * z for z in temd], axis=1)
    else:
        dz = tf.expand_dims(_stag4(thk), axis=0)

    # B has Unit Mpa y^(1/n)
    B = 2.0 * arrhenius ** (-1.0 / exp_glen)

    if new_friction_param:
        C = slidingco # C has unit Mpa y^m m^(-m)
    else:
        if exp_weertman == 1:
            # C has unit Mpa y^m m^(-m)
            C = slidingco
        else:
            C = (slidingco + 10 ** (-12)) ** -(1.0 / exp_weertman)

    p = 1.0 + 1.0 / exp_glen
    s = 1.0 + 1.0 / exp_weertman

    sloptopgx, sloptopgy = _compute_gradient_stag(usurf - thk, dX, dX)
    sloptopgx = tf.expand_dims(sloptopgx, axis=1)
    sloptopgy = tf.expand_dims(sloptopgy, axis=1)

    # TODO : sloptopgx, sloptopgy must be the elevaion of layers! not the bedrock, this probably has very little effects.

    # sr has unit y^(-1)
    sr, div = _compute_strainrate_Glen_tf(
        U, thk, dX, dz, sloptopgx, sloptopgy, thr=thr_ice_thk
    )

    sr = tf.where(COND, sr, 0.0)

    # C_shear is unit  Mpa y^(1/n) y^(-1-1/n) * m^3 = Mpa y^(-1) m^3
    if len(B.shape)==3:
        C_shear = (
            tf.reduce_mean(
                _stag4(B)
                * tf.reduce_sum(dz * ((sr + regu_glen**2) ** (p / 2)), axis=1),
                axis=(-1, -2),
            )
            / p
        )
    else:
        C_shear = (
            tf.reduce_mean(
                tf.reduce_sum(_stag8(B) * dz * ((sr + regu_glen**2) ** (p / 2)), axis=1),
                axis=(-1, -2),
            )
            / p
        )

    # C_slid is unit Mpa y^m m^(-m) * m^(1+m) * y^(-1-m) * m*2 = Mpa y^(-1) m^3
    N = (
        _stag4(U[:, 0, 0, :, :] ** 2 + U[:, 1, 0, :, :] ** 2)
        + regu_weertman**2
        + (_stag4(U[:, 0, 0, :, :]) * sloptopgx + _stag4(U[:, 1, 0, :, :]) * sloptopgy)
        ** 2
    )
    C_slid = tf.reduce_mean(_stag4(C) * N ** (s / 2), axis=(-1, -2)) / s

    if iceflow_physics == 2:
        slopsurfx, slopsurfy = _compute_gradient_stag(usurf, dX, dX)
        slopsurfx = tf.expand_dims(slopsurfx, axis=1)
        slopsurfy = tf.expand_dims(slopsurfy, axis=1)

        uds = _stag8(U[:, 0]) * slopsurfx + _stag8(U[:, 1]) * slopsurfy
        uds = tf.where(COND, uds, 0.0)
        C_grav = (
            ice_density
            * gravity_cst
            * 10 ** (-6)
            * tf.reduce_mean(tf.reduce_sum(dz * uds, axis=1), axis=(-1, -2))
        )

    elif iceflow_physics == 4:
        # THIS IS NOT WORKING YET
        w = _stag8(U[:, 2])
        w = tf.where(COND, w, 0.0)
        p = _stag8(U[:, 3])
        p = tf.where(COND, p, 0.0)
        C_grav = (
            ice_density
            * gravity_cst
            * 10 ** (-6)
            * tf.reduce_mean(tf.reduce_sum(dz * w, axis=1), axis=(-1, -2))
            - 10 ** (-6)
            * tf.reduce_mean(tf.reduce_sum(dz * p * div, axis=1), axis=(-1, -2))
            + 10 ** (1) * tf.reduce_mean(div**2)
        )

        # here one must add a penalization term form the imcompressibility condition
        # -> [CHECK AT "Theoretical and Numerical Issues of Incompressible Fluid Flows", Frey, slide 53 from the course Sorbonne Uni.]
        # -> [CHECK AT eq. (2.45)- (2.48) of my PhD thesis ]
        # -> [CHECK AT cme358_lecture_notes_3-2.pdf, eq. (3.18)]

    #        print(C_shear[0].numpy(),C_slid[0].numpy(),C_grav[0].numpy(),C_front[0].numpy())

    return tf.reduce_mean(C_shear + C_slid + C_grav)  # * dX[:,0,0]**2


# @tf.function(experimental_relax_shapes=True)
def iceflow_energy_XY(params, X, Y):

    U = Y_to_U(params, Y)
    
    fieldin = X_to_fieldin(params, X) 

    return iceflow_energy( params, U, fieldin )

def Y_to_U(params, Y):
    N = params.Nz

    if params.iceflow_physics == 2:
        U = tf.stack(
            [
                tf.experimental.numpy.moveaxis(Y[:, :, :, :N], [-1], [1]),
                tf.experimental.numpy.moveaxis(Y[:, :, :, N:], [-1], [1]),
            ],
            axis=1,
        )
    elif params.iceflow_physics == 4:
        U = tf.stack(
            [
                tf.experimental.numpy.moveaxis(Y[:, :, :, :N], [-1], [1]),
                tf.experimental.numpy.moveaxis(Y[:, :, :, N : 2 * N], [-1], [1]),
                tf.experimental.numpy.moveaxis(Y[:, :, :, 2 * N : 3 * N], [-1], [1]),
                tf.experimental.numpy.moveaxis(Y[:, :, :, 3 * N :], [-1], [1]),
            ],
            axis=1,
        )

    return U


def U_to_Y(params, U):
    if params.iceflow_physics == 2:
        UU = tf.experimental.numpy.moveaxis(U[0], [0], [-1])
        VV = tf.experimental.numpy.moveaxis(U[1], [0], [-1])
        RR = tf.expand_dims(
            tf.concat(
                [UU, VV],
                axis=-1,
            ),
            axis=0,
        )
    elif params.iceflow_physics == 4:
        UU = tf.experimental.numpy.moveaxis(U[0], [0], [-1])
        VV = tf.experimental.numpy.moveaxis(U[1], [0], [-1])
        WW = tf.experimental.numpy.moveaxis(U[2], [0], [-1])
        PP = tf.experimental.numpy.moveaxis(U[3], [0], [-1])
        RR = tf.expand_dims(
            tf.concat(
                [UU, VV, WW, PP],
                axis=-1,
            ),
            axis=0,
        )

    return RR

def fieldin_to_X(params,fieldin):
    
    X = []
     
    fieldin_dim=[0,0,1*(params.dim_arrhenius==3),0,0]

    for f,s in zip(fieldin,fieldin_dim):
        if s==0:
            X.append( tf.expand_dims(f,  axis=-1) )
        else:
            X.append( tf.experimental.numpy.moveaxis(f, [0], [-1]) )
              
    return  tf.expand_dims( tf.concat(X, axis=-1), axis=0)  

def X_to_fieldin(params, X):

    i = 0
    
    fieldin_dim=[0,0,1*(params.dim_arrhenius==3),0,0]
    
    fieldin = []
    
    for f,s in zip(params.fieldin,fieldin_dim):
        if s==0:
            fieldin.append( X[:,:,:,i] )
            i += 1
        else:
            fieldin.append( tf.experimental.numpy.moveaxis( 
                                X[:,:,:,i:i+params.Nz], [-1], [1]
                                                          ) )
            i += params.Nz
    
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

    zeta = np.arange(params.Nz + 1) / params.Nz
    weight = (zeta / params.vert_spacing) * (1.0 + (params.vert_spacing - 1.0) * zeta)
    weight = tf.Variable(weight[1:] - weight[:-1], dtype=tf.float32)
    state.vert_weight = tf.expand_dims(tf.expand_dims(weight, axis=-1), axis=-1)


########################################################################
########################################################################
#####################  Solver routines   ###############################
########################################################################
########################################################################


def solve_iceflow(params, state, U):
    """
    solve_iceflow
    """

    Cost_Glen = []

    for i in range(params.solve_iceflow_nbitmax):
        with tf.GradientTape() as t:
            t.watch(U)
            
            fieldin = [ tf.expand_dims(vars(state)[f], axis=0) for f in params.fieldin ]

            COST = iceflow_energy( params, tf.expand_dims(U, axis=0), fieldin )
            
            Cost_Glen.append(COST)

            # Stop if the cost no longer decreases
            if params.stop_if_no_decrease:
                if i > 1:
                    if Cost_Glen[-1] >= Cost_Glen[-2]:
                        break

            grads = tf.Variable(t.gradient(COST, [U]))

            state.optimizer.apply_gradients(
                zip([grads[i] for i in range(grads.shape[0])], [U])
            )

            if (i + 1) % 100 == 0:
                velsurf_mag = tf.sqrt(U[0, -1] ** 2 + U[1, -1] ** 2)
                print("solve :", i, COST.numpy(), np.max(velsurf_mag))

    U = tf.where(state.thk > 0, U, 0)

    return U, Cost_Glen


def _update_iceflow_solved(params, state):

    U, Cost_Glen = solve_iceflow(params, state, state.U)

    state.U.assign(U)

    state.COST_Glen = Cost_Glen[-1].numpy()

    update_2d_iceflow_variables(params, state)


def update_2d_iceflow_variables(params, state):

    state.uvelbase = state.U[0, 0, :, :]
    state.vvelbase = state.U[1, 0, :, :]
    state.ubar = tf.reduce_sum(state.U[0] * state.vert_weight, axis=0)
    state.vbar = tf.reduce_sum(state.U[1] * state.vert_weight, axis=0)
    state.uvelsurf = state.U[0, -1, :, :]
    state.vvelsurf = state.U[1, -1, :, :]


########################################################################
########################################################################
#####################  Emulation routines   ############################
########################################################################
########################################################################


def _update_iceflow_emulated(params, state):

    # Define the input of the NN, include scaling 
        
    fieldin = [ vars(state)[f] for f in params.fieldin ]
    
    X = fieldin_to_X(params, fieldin)
     
    if params.exclude_borders_from_iceflow:
        X = tf.pad(X, [[0,0],[1,1], [1,1],[0,0]], "SYMMETRIC")

    Y = state.iceflow_model(X)
    
    if params.exclude_borders_from_iceflow:
        Y = Y[:,1:-1,1:-1,:]

    Ny, Nx = state.thk.shape
    N = params.Nz

    U = Y_to_U(params, Y[:, :Ny, :Nx, :])[0]

#    U = tf.where(state.thk > 0, U, 0)

    state.U.assign(U)

    # If requested, the speeds are artifically upper-bounded
    if params.force_max_velbar > 0:
        velbar_mag = tf.norm(state.U, axis=0)          
        for i in range(2):
                state.U[i].assign( 
                    tf.where( velbar_mag >= params.force_max_velbar,
                              params.force_max_velbar * (state.U[i] / velbar_mag),
                              state.U[i] ) )

    update_2d_iceflow_variables(params, state)


def _update_iceflow_emulator(params, state):

    if (state.it<0) | (state.it%params.retrain_iceflow_emulator_freq==0):
         
        fieldin = [ vars(state)[f] for f in params.fieldin ]
        
        XX = fieldin_to_X(params, fieldin)
         
        X = _split_into_patches(XX, params.retrain_iceflow_emulator_framesizemax)
         
        state.COST_EMULATOR = []
        
        nbit = (state.it>=0)*params.retrain_iceflow_emulator_nbit \
             + (state.it<0)*params.retrain_iceflow_emulator_nbit_init

        for epoch in range(nbit):
            cost_emulator = tf.Variable(0.0)

            for i in range(X.shape[0]):
                with tf.GradientTape() as t:
                    Y = state.iceflow_model(X[i : i + 1, :, :, :])

                    COST = iceflow_energy_XY(params, X[i : i + 1, :, :, :], Y)

                    cost_emulator = cost_emulator + COST

                    if (epoch + 1) % 100 == 0:
                        U = Y_to_U(params, Y)[0]
                        velsurf_mag = tf.sqrt(U[0, -1] ** 2 + U[1, -1] ** 2)
                        print("train : ", epoch, COST.numpy(), np.max(velsurf_mag))

                grads = t.gradient(COST, state.iceflow_model.trainable_variables)

                state.opti_retrain.apply_gradients(
                    zip(grads, state.iceflow_model.trainable_variables)
                )

                state.opti_retrain.lr = params.retrain_iceflow_emulator_lr * (
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

    if params.retrain_iceflow_emulator_freq > 0:
        _update_iceflow_emulator(params, state)
        COST_Emulator = state.COST_EMULATOR[-1].numpy()
    else:
        COST_Emulator = 0.0

    _update_iceflow_emulated(params, state)

    if state.it % 10 == 0:
        UT, Cost_Glen = solve_iceflow(params, state, state.UT)
        state.UT.assign(UT)
        COST_Glen = Cost_Glen[-1].numpy()

        print("nb solve iterations :", len(Cost_Glen))

        l1, l2 = computemisfit(state, state.thk, state.U - state.UT)

        ERR = [state.t.numpy(), COST_Glen, COST_Emulator, l1, l2]

        print(ERR)

        with open(os.path.join(params.working_dir, "errors.txt"), "ab") as f:
            np.savetxt(f, np.expand_dims(ERR, axis=0), delimiter=",", fmt="%5.5f")


def computemisfit(state, thk, U):
    ubar = tf.reduce_sum(state.vert_weight * U[0], axis=0)
    vbar = tf.reduce_sum(state.vert_weight * U[1], axis=0)

    VEL = tf.stack([ubar, vbar], axis=0)
    MA = tf.where(thk > 1, tf.ones_like(VEL), 0)
    #MA = tf.where(state.thk > 1, tf.ones_like(VEL), 0)

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

    if params.activation == "lrelu":
        activation = tf.keras.layers.LeakyReLU(alpha=0.01)
    else:
        activation = tf.keras.layers.ReLU()

    for i in range(int(params.nb_layers)):
        conv = tf.keras.layers.Conv2D(
            filters=params.nb_out_filter,
            kernel_size=(params.conv_ker_size, params.conv_ker_size),
            padding="same",
        )(conv)

        conv = activation(conv)

        conv = tf.keras.layers.Dropout(params.dropout_rate)(conv)

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

    layers = np.arange(int(params.nb_blocks))

    number_of_filters = [
        params.nb_out_filter * 2 ** (layers[i]) for i in range(len(layers))
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
    directory = os.path.join(params.working_dir, "iceflow-model")

    if os.path.exists(directory):
        shutil.rmtree(directory)

    os.mkdir(directory)

    state.iceflow_model.save(os.path.join(directory, "model.h5"))
    
#    fieldin_dim=[0,0,1*(params.dim_arrhenius==3),0,0]

    fid = open(os.path.join(directory, "fieldin.dat"), "w")
#    for key,gg in zip(params.fieldin,fieldin_dim):
#        fid.write("%s %.1f \n" % (key, gg))
    for key in params.fieldin:
        fid.write("%s %.1f \n" % (key))
    fid.close()

    fid = open(os.path.join(directory, "vert_grid.dat"), "w")
    fid.write("%4.0f  %s \n" % (params.Nz, "# number of vertical grid point (Nz)"))
    fid.write(
        "%2.2f  %s \n"
        % (params.vert_spacing, "# param for vertical spacing (vert_spacing)")
    )
    fid.close()
