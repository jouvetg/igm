import numpy as np
import tensorflow as tf
import os

from .utils import *
from .energy_iceflow import *
from .neural_network import *

from igm.modules.iceflow import emulators
import importlib_resources

import igm
from hydra.utils import get_original_cwd
from pathlib import Path

def biject_power(x, n):
  cond = tf.greater_equal(x, 0)
  return tf.where(cond, tf.math.pow(x, n), -tf.math.pow(-x, n))


    # filepath = Path(get_original_cwd()).joinpath(cfg.input.load_ncdf.input_file)
def initialize_iceflow_emulator(cfg, state):

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=cfg.modules.iceflow.iceflow.retrain_emulator_lr,
        decay_steps=1000,
        decay_rate=0.001)
    
    if (int(tf.__version__.split(".")[1]) <= 10) | (
        int(tf.__version__.split(".")[1]) >= 16
    ):
        state.opti_retrain = getattr(
            tf.keras.optimizers, cfg.modules.iceflow.iceflow.optimizer_emulator
        )(learning_rate=cfg.modules.iceflow.iceflow.retrain_emulator_lr)
    else:
        print("Custom Optimizer")
        state.opti_retrain = getattr(
            tf.keras.optimizers, "AdamW"
        )(learning_rate=1e-3, weight_decay=1e-2, epsilon=1e-2) # epsilon makes a huge impact with convergence if you use higher iterations!
        # )(learning_rate=1e-3)

        # state.opti_retrain = getattr(
        #     tf.keras.optimizers.legacy, cfg.modules.iceflow.iceflow.optimizer_emulator
        # )(learning_rate=cfg.modules.iceflow.iceflow.retrain_emulator_lr)

    direct_name = (
        "pinnbp"
        + "_"
        + str(cfg.modules.iceflow.iceflow.Nz)
        + "_"
        + str(int(cfg.modules.iceflow.iceflow.vert_spacing))
        + "_"
    )
    direct_name += (
        cfg.modules.iceflow.iceflow.network
        + "_"
        + str(cfg.modules.iceflow.iceflow.nb_layers)
        + "_"
        + str(cfg.modules.iceflow.iceflow.nb_out_filter)
        + "_"
    )
    direct_name += (
        str(cfg.modules.iceflow.iceflow.dim_arrhenius)
        + "_"
        + str(int(cfg.modules.iceflow.iceflow.new_friction_param))
    )

    # print("direct_name: ", direct_name)
    # print()
    # exit()
    if cfg.modules.iceflow.iceflow.pretrained_emulator:
        if cfg.modules.iceflow.iceflow.emulator == "":
            if os.path.exists(
                importlib_resources.files(emulators).joinpath(direct_name)
            ):
                # print("ye")
                dirpath = importlib_resources.files(emulators).joinpath(direct_name)
                print("Found pretrained emulator in the igm package: " + direct_name)
            else:
                print("No pretrained emulator found in the igm package")
        else:
            if os.path.exists(cfg.modules.iceflow.iceflow.emulator):
                dirpath = cfg.modules.iceflow.iceflow.emulator
                print(
                    "----------------------------------> Found pretrained emulator: "
                    + cfg.modules.iceflow.iceflow.emulator
                )
            else:
                print(
                    "----------------------------------> No pretrained emulator found "
                )

        fieldin = []
        # igm_models_path = Path(igm.__file__).parent
        # igm_models_path = igm_models_path.joinpath("/modules/iceflow/emulators/pinnbp_10_4_unet_16_32_2_1/fieldin.dat")
        # path = Path(igm.__file__).parentjoinpath("pinnbp_10_4_unet_16_32_2_1/fieldin.dat")
        # path = igm_models_path
        # path = "/home/bfinley/igm/igm/modules/iceflow/emulators/pinnbp_10_4_unet_16_32_2_1/fieldin.dat"
        # fid = open(path, "r")
        fid = open(os.path.join(dirpath, "fieldin.dat"), "r")
        for fileline in fid:
            part = fileline.split()
            fieldin.append(part[0])
        fid.close()
        assert cfg.modules.iceflow.iceflow.fieldin == fieldin
        state.iceflow_model = tf.keras.models.load_model(
            os.path.join(dirpath, "model.h5"), compile=False
        )
        state.iceflow_model.compile()
    else:
        print(
            "----------------------------------> No pretrained emulator, start from scratch."
        )
        nb_inputs = len(cfg.modules.iceflow.iceflow.fieldin) + (
            cfg.modules.iceflow.iceflow.dim_arrhenius == 3
        ) * (cfg.modules.iceflow.iceflow.Nz - 1)
        nb_outputs = 2 * cfg.modules.iceflow.iceflow.Nz
        # state.iceflow_model = getattr(igm, cfg.modules.iceflow.iceflow.network)(
        #     cfg, nb_inputs, nb_outputs
        # )
        if cfg.modules.iceflow.iceflow.network == "cnn":
            state.iceflow_model = cnn(cfg, nb_inputs, nb_outputs)
        elif cfg.modules.iceflow.iceflow.network == "unet":
            state.iceflow_model = unet(cfg, nb_inputs, nb_outputs)

    # direct_name = 'pinnbp_10_4_cnn_16_32_2_1'
    # dirpath = importlib_resources.files(emulators).joinpath(direct_name)
    # iceflow_model_pretrained = tf.keras.models.load_model(
    #     os.path.join(dirpath, "model.h5"), compile=False
    # )
    # N=16
    # pretrained_weights = [layer.get_weights() for layer in iceflow_model_pretrained.layers[:N]]
    # for i in range(N):
    #     state.iceflow_model.layers[i].set_weights(pretrained_weights[i])


def update_iceflow_emulated(cfg, state):
    # Define the input of the NN, include scaling

    Ny, Nx = state.thk.shape
    N = cfg.modules.iceflow.iceflow.Nz

    fieldin = [vars(state)[f] for f in cfg.modules.iceflow.iceflow.fieldin]

    X = fieldin_to_X(cfg, fieldin)

    if cfg.modules.iceflow.iceflow.exclude_borders > 0:
        iz = cfg.modules.iceflow.iceflow.exclude_borders
        X = tf.pad(X, [[0, 0], [iz, iz], [iz, iz], [0, 0]], "SYMMETRIC")
        # X = tf.pad(X, [[0, 0], [iz, iz], [iz, iz], [0, 0]], "SYMMETRIC")

    # print(X.shape)
    # exit()
    if cfg.modules.iceflow.iceflow.multiple_window_size == 0:
        Y = state.iceflow_model(X)
    else:
        Y = state.iceflow_model(tf.pad(X, state.PAD, "CONSTANT"))[:, :Ny, :Nx, :]

    if cfg.modules.iceflow.iceflow.exclude_borders > 0:
        iz = cfg.modules.iceflow.iceflow.exclude_borders
        Y = Y[:, iz:-iz, iz:-iz, :]

    U, V = Y_to_UV(cfg, Y)
    U = U[0]
    V = V[0]

    U = tf.where(state.thk > 0, U, 0)
    V = tf.where(state.thk > 0, V, 0)
    
    state.U.assign(U)
    state.V.assign(V)

    # If requested, the speeds are artifically upper-bounded
    if cfg.modules.iceflow.iceflow.force_max_velbar > 0:
        velbar_mag = getmag3d(state.U, state.V)
        state.U.assign(
            tf.where(
                velbar_mag >= cfg.modules.iceflow.iceflow.force_max_velbar,
                cfg.modules.iceflow.iceflow.force_max_velbar * (state.U / velbar_mag),
                state.U,
            )
        )
        state.V.assign(
            tf.where(
                velbar_mag >= cfg.modules.iceflow.iceflow.force_max_velbar,
                cfg.modules.iceflow.iceflow.force_max_velbar * (state.V / velbar_mag),
                state.V,
            )
        )

    update_2d_iceflow_variables(cfg, state)


def weertman_sliding_law(cfg, emulator_output, effective_pressure=None):

    """
    Returns a tuple of basis_vectors and sliding_shear_stress for the loss computation in IGM.
    
    For example, for weertman sliding law, the basis vectors are U_basal and V_basal
    and the sliding law shear stress is the actual sliding law (tau_s = c * ||basal_velocity||^(s-2)*basal_velocity).
    
    You can the sliding law here (equation 17, where s = 1 + m): https://github.com/jouvetg/igm-paper/blob/main/paper.pdf
    as well as here (equation 2.16): https://www.cambridge.org/core/services/aop-cambridge-core/content/view/F2D0D3A274405887B512A474D0C64C1D/S0022112016005930a.pdf/mechanical-error-estimators-for-shallow-ice-flow-models.pdf

    Returns
    -------
    Tuple
        (basis_vectors, sliding_shear_stress)
    """


    # Manually doing sliding loss (for ground truth)
    c = tf.Variable(cfg.modules.iceflow.iceflow.sliding_law.coefficient.weertman)
    # s = cfg.modules.iceflow.iceflow.sliding_law.exponent.weertman
    s = 1.0 + 1.0 / cfg.modules.iceflow.iceflow.sliding_law.exponent.weertman

    U = emulator_output[:, :, :, 0 : cfg.modules.iceflow.iceflow.Nz]
    V = emulator_output[:, :, :, cfg.modules.iceflow.iceflow.Nz :]

    U_basal = U[0, ..., 0]
    V_basal = V[0, ..., 0]

    velbase_mag = (U_basal**2 + V_basal**2) ** (1/2)

    basis_vectors = [U_basal, V_basal]
    sliding_shear_stress = [
        tf.squeeze(c * velbase_mag ** (s - 2) * U_basal), #biject_power here for fractional exponents?
        tf.squeeze(c * velbase_mag ** (s - 2) * V_basal), #biject_power here for fractional exponents?
    ]

    # print(tf.reduce_max(tf.abs(tf.squeeze(c * velbase_mag ** (s - 2) * U_basal)) * 1e6), tf.reduce_max(tf.abs(tf.squeeze(c * velbase_mag ** (s - 2) * V_basal))* 1e6))
    
    return basis_vectors, sliding_shear_stress

# from igm.modules.iceflow.energy_iceflow import _compute_gradient_stag, _stag4
def weertman_sliding_law_regularized(cfg, emulator_output, effective_pressure=None):

    """
    Returns a tuple of basis_vectors and sliding_shear_stress for the loss computation in IGM.
    
    For example, for weertman sliding law, the basis vectors are U_basal and V_basal
    and the sliding law shear stress is the actual sliding law (tau_s = c * ||basal_velocity||^(s-2)*basal_velocity).
    
    You can the sliding law here (equation 17, where s = 1 + m): https://github.com/jouvetg/igm-paper/blob/main/paper.pdf
    as well as here (equation 2.16): https://www.cambridge.org/core/services/aop-cambridge-core/content/view/F2D0D3A274405887B512A474D0C64C1D/S0022112016005930a.pdf/mechanical-error-estimators-for-shallow-ice-flow-models.pdf

    Returns
    -------
    Tuple
        (basis_vectors, sliding_shear_stress)
    """
    # N = (
    #     _stag4(U_basal ** 2 + V_basal ** 2)
    #     + regu_weertman**2
    #     + (_stag4(U_basal) * sloptopgx + _stag4(V_basal) * sloptopgy) ** 2
    # )
    # C_slid = _stag4(C) * N ** (s / 2) / s

    
    # lsurf = state.usurf - state.thk
    # sloptopgx, sloptopgy = _compute_gradient_stag(lsurf, state.dX, state.dX)

    # Manually doing sliding loss (for ground truth)
    c = tf.Variable(cfg.modules.iceflow.iceflow.sliding_law.coefficient.weertman)
    regu_weertman = tf.Variable(cfg.modules.iceflow.iceflow.regu_weertman)
    s = 1.0 + 1.0 / cfg.modules.iceflow.iceflow.sliding_law.exponent.weertman

    U = emulator_output[:, :, :, 0 : cfg.modules.iceflow.iceflow.Nz]
    V = emulator_output[:, :, :, cfg.modules.iceflow.iceflow.Nz :]

    U_basal = U[0, ..., 0]
    V_basal = V[0, ..., 0]

    velbase_mag = (U_basal**2 + V_basal**2 + regu_weertman**2) ** (1/2)

    basis_vectors = [U_basal, V_basal]
    sliding_shear_stress = [
        tf.squeeze(c * velbase_mag ** (s - 2) * U_basal),
        tf.squeeze(c * velbase_mag ** (s - 2) * V_basal),
    ]
    
    # print(tf.reduce_max(tf.abs(tf.squeeze(c * velbase_mag ** (s - 2) * U_basal))), tf.reduce_max(tf.abs(tf.squeeze(c * velbase_mag ** (s - 2) * V_basal))))

    return basis_vectors, sliding_shear_stress

def budd_sliding_law(cfg, emulator_output, effective_pressure):

    """
    Returns a tuple of basis_vectors and sliding_shear_stress for the loss computation in IGM.
    
    ...

    Returns
    -------
    Tuple
        (basis_vectors, sliding_shear_stress)
    """


    # Manually doing sliding loss (for ground truth)
    c = cfg.modules.iceflow.iceflow.sliding_law.coefficient.budd
    n = cfg.modules.iceflow.iceflow.sliding_law.exponent.budd
    N = effective_pressure * 1e6 # convert to Pascals to be consistent with the coefficents (https://esurf.copernicus.org/articles/4/159/2016/)

    U = emulator_output[:, :, :, 0 : cfg.modules.iceflow.iceflow.Nz]
    V = emulator_output[:, :, :, cfg.modules.iceflow.iceflow.Nz :]

    U_basal = U[0, ..., 0]
    V_basal = V[0, ..., 0]

    velbase_mag = (U_basal**2 + V_basal**2) ** (1/2) # assuming l2 norm...
    basis_vectors = [U_basal, V_basal]
    
    # sliding_shear_stress_u = (velbase_mag * N / c) ** (1/n) * U_basal # Pa
    # sliding_shear_stress_v = (velbase_mag * N / c) ** (1/n) * V_basal # Pa
    
    # Biject power is used to handle fractional exponents, but we just want negative values to go to zero?
    # sliding_shear_stress_u = biject_power(velbase_mag * N / c, (1/n)) * (U_basal / velbase_mag) # Pa
    # sliding_shear_stress_v = biject_power(velbase_mag * N / c, (1/n)) * (V_basal / velbase_mag) # Pa
    
    
    sliding_shear_stress_u = tf.maximum(velbase_mag * N / c, 0) ** (1/n) * (U_basal / velbase_mag) # Pa
    sliding_shear_stress_v = tf.maximum(velbase_mag * N / c, 0) ** (1/n) * (V_basal / velbase_mag) # Pa
    

    # sliding_shear_stress_u = tf.where(tf.math.is_nan(sliding_shear_stress_u), 0, sliding_shear_stress_u)
    # sliding_shear_stress_v = tf.where(tf.math.is_nan(sliding_shear_stress_v), 0, sliding_shear_stress_v)

    
    sliding_shear_stress = [
        sliding_shear_stress_u * 1e-6, # convert to MPa
        sliding_shear_stress_v * 1e-6 # convert to MPa
    ]

    # print("BUDD")
    # print(sliding_shear_stress)
    return basis_vectors, sliding_shear_stress

def regularized_coulomb_sliding_law(cfg, emulator_output, effective_pressure):
    """
    Returns a tuple of basis_vectors and sliding_shear_stress for the loss computation in IGM.
    
    https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1029/2019GL082526

    Returns
    -------
    Tuple
        (basis_vectors, sliding_shear_stress)
    """

    # print('COULOMB')
    c = cfg.modules.iceflow.iceflow.sliding_law.coefficient.coulomb
    n = cfg.modules.iceflow.iceflow.sliding_law.exponent.coulomb
    N = effective_pressure * 1e6 # convert to Pascals to be consistent with the coefficents (https://esurf.copernicus.org/articles/4/159/2016/)
    
    
    # print("Negative N?")
    N = tf.where(N < 0, 0, N) # why is N negative?
    
    # print(tf.reduce_min(N), tf.reduce_max(N))
    
    gamma_0 = cfg.modules.iceflow.iceflow.gamma_0 
    # equivalent to A_s * C^n in regularized couluomb law (equation 3)
    # https://www.science.org/doi/10.1126/sciadv.abe7798
    
    U = emulator_output[:, :, :, 0 : cfg.modules.iceflow.iceflow.Nz]
    V = emulator_output[:, :, :, cfg.modules.iceflow.iceflow.Nz :]

    U_basal = U[0, ..., 0]
    V_basal = V[0, ..., 0]

    velbase_mag = (U_basal**2 + V_basal**2) ** (1/2) # assuming l2 norm...
    basis_vectors = [U_basal, V_basal]
    
    eps = 1e-3
    numerator = velbase_mag
    denominator = velbase_mag + gamma_0 * (N** n) + eps
    
    
    # Biject power is used to handle fractional exponents, but we just want negative values to go to zero?
    # sliding_shear_stress_u = N * c * biject_power(numerator / denominator, (1/n)) * (U_basal/(velbase_mag + 1e-9)) # Pa
    # sliding_shear_stress_v = N * c * biject_power(numerator / denominator, (1/n)) * (V_basal/(velbase_mag  + 1e-9)) # Pa
    
    sliding_shear_stress_u = (N + eps) * c * tf.maximum(numerator / denominator, 0) ** (1/n) * (U_basal/(velbase_mag + eps)) # Pa
    sliding_shear_stress_v = (N + eps) * c * tf.maximum(numerator / denominator, 0) ** (1/n) * (V_basal/(velbase_mag + eps)) # Pa
    
    
    
    # print("shear stress")
    # print(sliding_shear_stress_u, sliding_shear_stress_v)
    # print(tf.reduce_max(tf.abs(sliding_shear_stress_u)), tf.reduce_max(tf.abs(sliding_shear_stress_v)))
    # sliding_shear_stress_u = (c * N) * (numerator / denominator) ** (1/n) * (U_basal/velbase_mag)
    # sliding_shear_stress_v = (c * N) * (numerator / denominator) ** (1/n) * (V_basal/velbase_mag)
    
    sliding_shear_stress = [
        sliding_shear_stress_u * 1e-6, # convert to MPa
        sliding_shear_stress_v * 1e-6, # convert to MPa
    ]
    # print(sliding_shear_stress)

    return basis_vectors, sliding_shear_stress

def coulomb_sliding_law(cfg, emulator_output, effective_pressure): # also called coulumb

    """
    Returns a tuple of basis_vectors and sliding_shear_stress for the loss computation in IGM.
    
    https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1029/2019GL082526

    Returns
    -------
    Tuple
        (basis_vectors, sliding_shear_stress)
    """

    # print('COULOMB')
    c = cfg.modules.iceflow.iceflow.sliding_law.coefficient.coulomb
    n = cfg.modules.iceflow.iceflow.sliding_law.exponent.coulomb
    N = effective_pressure * 1e6 # convert to Pascals to be consistent with the coefficents (https://esurf.copernicus.org/articles/4/159/2016/)
    
    
    # print("Negative N?")
    N = tf.where(N < 0, 0, N) # why is N negative?
    
    # print(tf.reduce_min(N), tf.reduce_max(N))
    
    gamma_0 = cfg.modules.iceflow.iceflow.gamma_0 
    # equivalent to A_s * C^n in regularized couluomb law (equation 3)
    # https://www.science.org/doi/10.1126/sciadv.abe7798
    
    U = emulator_output[:, :, :, 0 : cfg.modules.iceflow.iceflow.Nz]
    V = emulator_output[:, :, :, cfg.modules.iceflow.iceflow.Nz :]

    U_basal = U[0, ..., 0]
    V_basal = V[0, ..., 0]

    velbase_mag = (U_basal**2 + V_basal**2) ** (1/2) # assuming l2 norm...
    basis_vectors = [U_basal, V_basal]
    
    eps = 1e-3
    numerator = velbase_mag
    denominator = velbase_mag + gamma_0 * (N** n) + eps
    
    
    # Biject power is used to handle fractional exponents, but we just want negative values to go to zero?
    # sliding_shear_stress_u = N * c * biject_power(numerator / denominator, (1/n)) * (U_basal/(velbase_mag + 1e-9)) # Pa
    # sliding_shear_stress_v = N * c * biject_power(numerator / denominator, (1/n)) * (V_basal/(velbase_mag  + 1e-9)) # Pa
    
    sliding_shear_stress_u = (N + eps) * c * tf.maximum(numerator / denominator, 0) ** (1/n) * (U_basal/(velbase_mag + eps)) # Pa
    sliding_shear_stress_v = (N + eps) * c * tf.maximum(numerator / denominator, 0) ** (1/n) * (V_basal/(velbase_mag + eps)) # Pa
    
    
    
    # print("shear stress")
    # print(sliding_shear_stress_u, sliding_shear_stress_v)
    # print(tf.reduce_max(tf.abs(sliding_shear_stress_u)), tf.reduce_max(tf.abs(sliding_shear_stress_v)))
    # sliding_shear_stress_u = (c * N) * (numerator / denominator) ** (1/n) * (U_basal/velbase_mag)
    # sliding_shear_stress_v = (c * N) * (numerator / denominator) ** (1/n) * (V_basal/velbase_mag)
    
    sliding_shear_stress = [
        sliding_shear_stress_u * 1e-6, # convert to MPa
        sliding_shear_stress_v * 1e-6, # convert to MPa
    ]
    # print(sliding_shear_stress)

    return basis_vectors, sliding_shear_stress

def get_sliding_law_function(method):

    if method == "weertman":
        return weertman_sliding_law
    elif method == "weertman_regularized":
        return weertman_sliding_law_regularized
    elif method == "coulomb":
        return regularized_coulomb_sliding_law
    # elif method == "coulomb":
    #     return coulomb_sliding_law
    elif method == "budd":
        return budd_sliding_law
    else:
        raise NotImplementedError("Sliding law method not implemented. Please specify between 'weertman', 'coulomb' or 'budd'.")

def get_median(v):
    v = tf.reshape(v, [-1])
    m = v.get_shape()[0]//2
    return tf.reduce_min(tf.nn.top_k(v, m, sorted=False).values)

def update_iceflow_emulator(cfg, state):
    # print("Are we retraining?")
    # print(state.it < 0)
    # print(state.it % cfg.modules.iceflow.iceflow.retrain_emulator_freq)
    if (state.it < 0) | (
        state.it % cfg.modules.iceflow.iceflow.retrain_emulator_freq == 0
    ):
        # print("RETRAINING")
        # exit()
        fieldin = [vars(state)[f] for f in cfg.modules.iceflow.iceflow.fieldin]

        ########################

        # thkext = tf.pad(state.thk,[[1,1],[1,1]],"CONSTANT",constant_values=1)
        # # this permits to locate the calving front in a cell in the 4 directions
        # state.CF_W = tf.where((state.thk>0)&(thkext[1:-1,:-2]==0),1.0,0.0)
        # state.CF_E = tf.where((state.thk>0)&(thkext[1:-1,2:]==0),1.0,0.0)
        # state.CF_S = tf.where((state.thk>0)&(thkext[:-2,1:-1]==0),1.0,0.0)
        # state.CF_N = tf.where((state.thk>0)&(thkext[2:,1:-1]==0),1.0,0.0)

        ########################

        XX = fieldin_to_X(cfg, fieldin)
        X = _split_into_patches(
            XX, cfg.modules.iceflow.iceflow.retrain_emulator_framesizemax
        )

        Ny = X.shape[1]
        Nx = X.shape[2]

        PAD = compute_PAD(cfg, Nx, Ny)

        state.COST_EMULATOR = tf.constant(0.0)

        nbit = int(
            (state.it >= 0) * cfg.modules.iceflow.iceflow.retrain_emulator_nbit
            + (state.it < 0) * cfg.modules.iceflow.iceflow.retrain_emulator_nbit_init
        )

        iz = cfg.modules.iceflow.iceflow.exclude_borders

        for epoch in range(nbit):
            cost_emulator = tf.Variable(0.0)

            for i in range(X.shape[0]):
                with tf.GradientTape(persistent=True) as t:

                    Y = state.iceflow_model(
                        # tf.pad(X[i : i + 1, :, :, :], PAD, "REFLECT")
                        tf.pad(X[i : i + 1, :, :, :], PAD, "CONSTANT")
                    )[:, :Ny, :Nx, :]

                    # print(X.shape)
                    # print(state.thk.shape)
                    # print(state.usurf.shape)
                    # print(state.arrhenius.shape)
                    
                    # state.custom_arrhenius_bottom = state.arrhenius[0, ...]
                    # state.custom_arrhenius_top = state.arrhenius[-1, ...]
                    # print(state.slidingco.shape)
                    # print(state.dX.shape)
                    # exit()
                    # state.custom_Y = Y
                    # # Basal shear stress computation for loss function
                    sliding_law_method = cfg.modules.iceflow.iceflow.sliding_law.method
                    sliding_law = get_sliding_law_function(sliding_law_method)
                    
                    effective_pressure = None
                    if sliding_law_method != "weertman":
                        if not hasattr(state, "effective_pressure"):
                            effective_pressure = 910 * 9.81 * state.thk * 0.2 # simple effective pressure (20% of overburden ice pressure)
                        else:
                            effective_pressure = state.effective_pressure
                    # print("effective pressure")
                    # print(state.effective_pressure)        
                    # effective_pressure = 910 * 9.81 * state.thk * 0.2 # simple effective pressure (20% of overburden ice pressure)
                    # print(effective_pressure)
                    # exit()
                        
                    basis_vectors, sliding_shear_stress = sliding_law(cfg, Y, effective_pressure)
                    
                    U_basal, V_basal = basis_vectors # ? Is there a way to make this more generalizable for different sliding laws / different bases?
                    number_of_cells = tf.cast(Nx * Ny, tf.float32)

                    if iz > 0:
                        C_shear, C_grav, C_float, B, srcapped, srx, srz, Exx, Exy, Eyy, Ezz, Exz, Eyz, slc, custom_dVdy, custom_V = iceflow_energy_XY(
                            cfg,
                            X[i : i + 1, iz:-iz, iz:-iz, :],
                            Y[:, iz:-iz, iz:-iz, :],
                        )
                    else:
                        C_shear, C_grav, C_float, B, srcapped, srx, srz, Exx, Exy, Eyy, Ezz, Exz, Eyz, slc, custom_dVdy, custom_V = iceflow_energy_XY(
                            cfg, X[i : i + 1, :, :, :], Y[:, :, :, :]
                        )
                    
                    
                    
                    # C_shear = tf.where(tf.math.is_nan(C_shear),0,C_shear)
                    # C_grav = tf.where(tf.math.is_nan(C_grav),0,C_grav)
                    # # import tensorflow_probability as tfp
                    # C_shear = get_median(C_shear)
                    # C_grav = get_median(C_grav)
                    # C_float = get_median(C_float)
                    
                    # COST_no_sliding = C_shear + C_grav + C_float
                    
                    COST_no_sliding = (
                        tf.reduce_mean(C_shear) # / (tf.reduce_max(C_shear) - tf.reduce_min(C_shear))
                        + tf.reduce_mean(C_grav) #/ (tf.reduce_max(C_grav) - tf.reduce_min(C_grav))
                        + tf.reduce_mean(C_float) #/ (tf.reduce_max(C_float) - tf.reduce_min(C_float))
                    )
                    
                    # print(slc.shape, custom_dVdy.shape, custom_V.shape)
                    # exit()
                    # print(X[i : i + 1, :, :, :].shape)
                    # print(iz)
                    # C_shear = tf.pad(C_shear, [[0, 0], [1, 0], [0, 1]], "CONSTANT")
                    # C_grav = tf.pad(C_grav, [[0, 0], [1, 0], [0, 1]], "CONSTANT")
                    # # B = tf.pad(B, [[0, 0], [0, 0], [1, 0], [0, 1]], "CONSTANT")
                    # srcapped = tf.pad(srcapped, [[0, 0], [0, 0], [1, 0], [0, 1]], "CONSTANT")
                    # srx = tf.pad(srx, [[0, 0], [0, 0], [1, 0], [0, 1]], "CONSTANT")
                    # srz = tf.pad(srz, [[0, 0], [0, 0], [1, 0], [0, 1]], "CONSTANT")
                    
                    # Exx = tf.pad(Exx, [[0, 0], [0, 0], [1, 0], [0, 1]], "CONSTANT")
                    # Exy = tf.pad(Exy, [[0, 0], [0, 0], [1, 0], [0, 1]], "CONSTANT")
                    # Eyy = tf.pad(Eyy, [[0, 0], [0, 0], [1, 0], [0, 1]], "CONSTANT")
                    # Ezz = tf.pad(Ezz, [[0, 0], [0, 0], [1, 0], [0, 1]], "CONSTANT")
                    # Exz = tf.pad(Exz, [[0, 0], [0, 0], [1, 0], [0, 1]], "CONSTANT")
                    # Eyz = tf.pad(Eyz, [[0, 0], [0, 0], [1, 0], [0, 1]], "CONSTANT")
                    
                    # slc = tf.pad(slc, [[0, 0], [0, 0], [1, 0], [0, 1]], "CONSTANT")
                    # custom_dVdy = tf.pad(custom_dVdy, [[0, 0], [0, 0], [1, 0], [0, 1]], "CONSTANT")
                    # custom_V = tf.pad(custom_V, [[0, 0], [0, 0], [1, 0], [0, 1]], "CONSTANT")
                    
                    # print(C_shear.shape)
                    # print(B.shape)
                    # print(srcapped.shape)
                    # print(srx.shape)
                    # print(srz.shape)
                    # exit()
                    # state.B = B[0,0,:,:]
                    # state.srcapped = srcapped[0,0,:,:]
                    # state.srx = srx[0,0,:,:]
                    # state.srz = srz[0,0,:,:]
                    # state.Exx = Exx[0,0,:,:]
                    # state.Exy = Exy[0,0,:,:]
                    # state.Eyy = Eyy[0,0,:,:]
                    # state.Ezz = Ezz[0,0,:,:]
                    # state.Exz = Exz[0,0,:,:]
                    # state.Eyz = Eyz[0,0,:,:]
                    # state.C_shear = tf.squeeze(C_shear)
                    # state.C_grav = tf.squeeze(C_grav)
                    
                    # state.slc = tf.squeeze(slc)
                    # state.custom_V = custom_V[0,0,:,:]
                    # state.custom_dVdy = custom_dVdy[0,0,:,:]

                    if (epoch + 1) % 30 == 0:
                        print(
                            "---------- > ",
                            tf.reduce_mean(C_shear).numpy(),
                            tf.reduce_mean(C_grav).numpy(),
                            tf.reduce_mean(C_float).numpy(),
                        )
                        
                        print("maxes", tf.reduce_max(C_shear).numpy(), tf.reduce_max(C_grav).numpy(), tf.reduce_max(C_float).numpy())
                        print("mins", tf.reduce_min(C_shear).numpy(), tf.reduce_min(C_grav).numpy(), tf.reduce_min(C_float).numpy())
                        print("medians", tf.reduce_mean(C_shear).numpy(), tf.reduce_mean(C_grav).numpy(), tf.reduce_mean(C_float).numpy())
                        
                        # print("Loss Term Costs")
                        # print("----------------------------------------") # temporary
                        # print(f"C_shear: {tf.reduce_sum(C_shear).numpy()} | Has NaNs? {tf.math.is_nan(C_shear).numpy().any()}")
                        # print(f"C_grav: {tf.reduce_sum(C_grav).numpy()} | Has NaNs? {tf.math.is_nan(C_grav).numpy().any()}")
                        # print(f"C_float: {tf.reduce_sum(C_float).numpy()} | Has NaNs? {tf.math.is_nan(C_float).numpy().any()}")

                    #                    state.C_shear = tf.pad(C_shear[0],[[0,1],[0,1]],"CONSTANT")
                    #                    state.C_slid  = tf.pad(C_slid[0],[[0,1],[0,1]],"CONSTANT")
                    #                    state.C_grav  = tf.pad(C_grav[0],[[0,1],[0,1]],"CONSTANT")
                    #                    state.C_float = C_float[0]

                    # print(state.C_shear.shape, state.C_slid.shape, state.C_grav.shape, state.C_float.shape,state.thk.shape )

                    cost_emulator = cost_emulator + COST_no_sliding

                    if (epoch + 1) % 100 == 0:
                        U, V = Y_to_UV(cfg, Y)
                        U = U[0]
                        V = V[0]
                        velsurf_mag = tf.sqrt(U[-1] ** 2 + V[-1] ** 2)
                        print("train : ", epoch, COST_no_sliding.numpy(), np.max(velsurf_mag))
                
                
                sliding_gradients_manual = t.gradient(
                    [U_basal, V_basal],
                    state.iceflow_model.trainable_variables,
                    output_gradients=sliding_shear_stress,
                )
                
                # print(C_shear, tf.reduce_any(tf.math.is_nan(C_shear)))
                # exit()
                # print(f"Costs for border cells: {tf.reduce_sum(C_shear[...,-50]),tf.reduce_sum(C_grav[:,-50]), tf.reduce_sum(C_float[:,-50])}")
                # All gradients other than sliding loss
                grads_no_sliding = t.gradient(COST_no_sliding, state.iceflow_model.trainable_variables)
                
                # Combining sliding loss gradients with other loss term gradients
                combined_gradients = [
                    grad + (sliding_grad / number_of_cells) # divide by number of cells as other loss terms in IGM use reduce_mean and not reduce_sum
                    for grad, sliding_grad in zip(grads_no_sliding, sliding_gradients_manual)
                ]
                
                # Clipping gradients
                # combined_gradients, _ = tf.clip_by_global_norm(combined_gradients, 5.0)
                
                state.opti_retrain.apply_gradients(
                    zip(combined_gradients, state.iceflow_model.trainable_variables)
                )

                state.opti_retrain.lr = (
                    cfg.modules.iceflow.iceflow.retrain_emulator_lr
                    * (0.95 ** (epoch / 1000))
                )

            # print(cost_emulator.shape, state.COST_EMULATOR.shape)
            # if epoch == 0:
            #     state.COST_EMULATOR = tf.stack([state.COST_EMULATOR, cost_emulator])
            # else:
            #     state.COST_EMULATOR = tf.concat([state.COST_EMULATOR, [cost_emulator]], 0)
            # state.COST_EMULATOR.append(cost_emulator)

    if len(cfg.modules.iceflow.iceflow.save_cost_emulator) > 0:
        np.savetxt(
            cfg.modules.iceflow.iceflow.output_directory
            + cfg.modules.iceflow.iceflow.save_cost_emulator
            + "-"
            + str(state.it)
            + ".dat",
            np.array(state.COST_EMULATOR),
            fmt="%5.10f",
        )


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


def save_iceflow_model(cfg, state):
    directory = "iceflow-model"

    import shutil

    if os.path.exists(directory):
        shutil.rmtree(directory)

    os.mkdir(directory)

    state.iceflow_model.save(os.path.join(directory, "model.h5"))

    #    fieldin_dim=[0,0,1*(cfg.modules.iceflow.iceflow.dim_arrhenius==3),0,0]

    fid = open(os.path.join(directory, "fieldin.dat"), "w")
    #    for key,gg in zip(cfg.modules.iceflow.iceflow.fieldin,fieldin_dim):
    #        fid.write("%s %.1f \n" % (key, gg))
    for key in cfg.modules.iceflow.iceflow.fieldin:
        print(key)
        fid.write("%s \n" % (key))
    fid.close()

    fid = open(os.path.join(directory, "vert_grid.dat"), "w")
    fid.write(
        "%4.0f  %s \n"
        % (cfg.modules.iceflow.iceflow.Nz, "# number of vertical grid point (Nz)")
    )
    fid.write(
        "%2.2f  %s \n"
        % (
            cfg.modules.iceflow.iceflow.vert_spacing,
            "# param for vertical spacing (vert_spacing)",
        )
    )
    fid.close()
