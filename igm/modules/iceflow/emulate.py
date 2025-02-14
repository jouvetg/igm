import numpy as np
import tensorflow as tf
import os

from .utils import *
from .energy_iceflow import *
from .neural_network import *

from igm.modules.iceflow import emulators
import importlib_resources


def initialize_iceflow_emulator(cfg, state):

    if (int(tf.__version__.split(".")[1]) <= 10) | (
        int(tf.__version__.split(".")[1]) >= 16
    ):
        state.opti_retrain = getattr(
            tf.keras.optimizers, cfg.modules.iceflow.iceflow.optimizer_emulator
        )(learning_rate=cfg.modules.iceflow.iceflow.retrain_emulator_lr)
    else:
        state.opti_retrain = getattr(
            tf.keras.optimizers.legacy, cfg.modules.iceflow.iceflow.optimizer_emulator
        )(learning_rate=cfg.modules.iceflow.iceflow.retrain_emulator_lr)

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

    if cfg.modules.iceflow.iceflow.pretrained_emulator:
        if cfg.modules.iceflow.iceflow.emulator == "":
            if os.path.exists(
                importlib_resources.files(emulators).joinpath(direct_name)
            ):
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

    velbase_mag = U_basal**2 + V_basal**2

    basis_vectors = [U_basal, V_basal]
    sliding_shear_stress = [
        tf.squeeze(c * velbase_mag ** ((s - 2) / 2) * U_basal),
        tf.squeeze(c * velbase_mag ** ((s - 2) / 2) * V_basal),
    ]

    return basis_vectors, sliding_shear_stress

def get_simple_effective_pressure(state):
    p_i = 910
    g = 9.81
    percentage = 0.45
    
    ice_overburden_pressure = p_i * g * state.thk
    water_pressure = ice_overburden_pressure * percentage
    
    effective_pressure = ice_overburden_pressure - water_pressure
    
    return effective_pressure

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
    N = effective_pressure

    U = emulator_output[:, :, :, 0 : cfg.modules.iceflow.iceflow.Nz]
    V = emulator_output[:, :, :, cfg.modules.iceflow.iceflow.Nz :]

    U_basal = U[0, ..., 0]
    V_basal = V[0, ..., 0]

    velbase_mag = (U_basal**2 + V_basal**2) ** (1/2) # assuming l2 norm...
    basis_vectors = (U_basal, V_basal)
    
    sliding_shear_stress_u = (velbase_mag * N / c) ** (1/n) * (U_basal / velbase_mag)
    sliding_shear_stress_v = (velbase_mag * N / c) ** (1/n) * (V_basal / velbase_mag)
    # sliding_shear_stress_u = (c / N) ** (-1/n) * U_basal * tf.abs(U_basal) ** (1/n - 1) # check if its absolute value or not...
    # sliding_shear_stress_v = (c / N) ** (-1/n) * V_basal * tf.abs(V_basal) ** (1/n - 1) # check if its absolute value or not...
    
    sliding_shear_stress = (
        sliding_shear_stress_u,
        sliding_shear_stress_v,
    )

    return basis_vectors, sliding_shear_stress

def regularized_coulomb_sliding_law(cfg, emulator_output, effective_pressure): # also called coulumb

    """
    Returns a tuple of basis_vectors and sliding_shear_stress for the loss computation in IGM.
    
    https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1029/2019GL082526

    Returns
    -------
    Tuple
        (basis_vectors, sliding_shear_stress)
    """

    c = cfg.modules.iceflow.iceflow.sliding_law.coefficient.coulomb
    n = cfg.modules.iceflow.iceflow.sliding_law.exponent.coulomb
    N = effective_pressure
    gamma_0 = cfg.modules.iceflow.iceflow.gamma_0 
    # equivalent to A_s * C^n in regularized couluomb law (equation 3)
    # https://www.science.org/doi/10.1126/sciadv.abe7798
    
    U = emulator_output[:, :, :, 0 : cfg.modules.iceflow.iceflow.Nz]
    V = emulator_output[:, :, :, cfg.modules.iceflow.iceflow.Nz :]

    U_basal = U[0, ..., 0]
    V_basal = V[0, ..., 0]

    velbase_mag = (U_basal**2 + V_basal**2) ** (1/2) # assuming l2 norm...
    basis_vectors = (U_basal, V_basal)
    
    numerator = velbase_mag
    denominator = velbase_mag + gamma_0 * (N ** n)
    
    sliding_shear_stress_u = (c * N) * (numerator / denominator) ** (1/n) * (U_basal/velbase_mag)
    sliding_shear_stress_v = (c * N) * (numerator / denominator) ** (1/n) * (V_basal/velbase_mag)
    
    sliding_shear_stress = (
        sliding_shear_stress_u,
        sliding_shear_stress_v,
    )

    return basis_vectors, sliding_shear_stress

def get_sliding_law_function(method):

    if method == "weertman":
        return weertman_sliding_law
    elif method == "coulomb":
        return regularized_coulomb_sliding_law
    elif method == "budd":
        return budd_sliding_law
    else:
        raise NotImplementedError("Sliding law method not implemented. Please specify between 'weertman', 'coulomb' or 'budd'.")

def _stag4(B):
    return (B[:, 1:, 1:] + B[:, 1:, :-1] + B[:, :-1, 1:] + B[:, :-1, :-1]) / 4

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

def update_iceflow_emulator(cfg, state):
    if (state.it < 0) | (
        state.it % cfg.modules.iceflow.iceflow.retrain_emulator_freq == 0
    ):
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

        state.COST_EMULATOR = []

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
                        tf.pad(X[i : i + 1, :, :, :], PAD, "CONSTANT")
                    )[:, :Ny, :Nx, :]

                    # # Basal shear stress computation for loss function
                    sliding_law_method = cfg.modules.iceflow.iceflow.sliding_law.method
                    sliding_law = get_sliding_law_function(sliding_law_method)
                    effective_pressure = None
                    basis_vectors, sliding_shear_stress_law = sliding_law(cfg, Y, effective_pressure)
                    
                    # print(Nx, Ny, sliding_shear_stress_law)
                    
                    number_of_cells = tf.cast(Nx * Ny, tf.float32)
                    
                    # print(sliding_shear_stress_law[1])
                    # exit()
                    # taking mean to match IGM implementation (weighting) as other loss terms are also averaged over the domain
                    # for i in range(len(sliding_shear_stress_law)):
                        # print(sliding_shear_stress_law[i])
                    
                    # sliding_shear_stress_law = sliding_shear_stress_law / number_of_cells
                    # print(sliding_shear_stress_law[0].shape)
                    # exit()
                    
                    # if sliding_law_method != "weertman":
                        # effective_pressure = state.effective_pressure
                    #     # effective_pressure = get_simple_effective_pressure(state)
                    U = Y[:, :, :, 0 : cfg.modules.iceflow.iceflow.Nz]
                    V = Y[:, :, :, cfg.modules.iceflow.iceflow.Nz :]

                    U_basal = U[0, ..., 0]
                    V_basal = V[0, ..., 0]
                    
                    # c = 1.0 * cfg.modules.iceflow.iceflow.sliding_law.coefficient.weertman
                    s = 1.0 + 1.0 / cfg.modules.iceflow.iceflow.sliding_law.exponent.weertman
    
                    # N = U_basal ** 2 + V_basal ** 2
                    # C_slid = c * N ** (s / 2) / s
    
                    # cost_sliding = tf.reduce_mean(C_slid)
                    
                    # Manually doing sliding loss (for ground truth)
                    c = tf.Variable(cfg.modules.iceflow.iceflow.init_slidingco)
                    # s = cfg.modules.iceflow.iceflow.exp_weertman

                    # U = Y[:, :, :, 0 : cfg.modules.iceflow.iceflow.Nz]
                    # V = Y[:, :, :, cfg.modules.iceflow.iceflow.Nz :]

                    # U_basal = U[0, ..., 0]
                    # V_basal = V[0, ..., 0]

                    # velbase = tf.stack([U_basal, V_basal], axis=-1)

                    ### Sliding loss (ground truth - matches what was done before in IGM without using the gradient directly)

                    lsurf = state.usurf - state.thk
                    sloptopgx, sloptopgy = _compute_gradient_stag(tf.expand_dims(lsurf, axis=0), tf.expand_dims(state.dX, axis=0), tf.expand_dims(state.dX, axis=0))
                    C = state.slidingco

                    # C_slid is unit Mpa m^-1 m/y m = Mpa m/y
                    # N = (
                    #     _stag4(U_basal ** 2 + V_basal ** 2)
                    #     + cfg.modules.iceflow.iceflow.regu_weertman**2
                    #     + (_stag4(U_basal) * sloptopgx + _stag4(V_basal) * sloptopgy) ** 2
                    # )
                    # C_slid = _stag4(tf.expand_dims(C, axis=0)) * N ** (s / 2) / s
                    
                    N = U_basal**2 + V_basal**2  # velbase magntude
                    C_slid = (c / s) * N ** (s / 2)

                    ###
                    

                    sliding_loss = tf.reduce_mean(C_slid)
                    # print(sliding_loss)
                    # exit()
                    

                    # basis_vectors, sliding_shear_stress = sliding_law(cfg, Y, effective_pressure)

                    if iz > 0:
                        C_shear, C_slid_inside, C_grav, C_float = iceflow_energy_XY(
                            cfg,
                            X[i : i + 1, iz:-iz, iz:-iz, :],
                            Y[:, iz:-iz, iz:-iz, :],
                        )
                        # C_shear, C_grav, C_float = iceflow_energy_XY(
                        #     cfg,
                        #     X[i : i + 1, iz:-iz, iz:-iz, :],
                        #     Y[:, iz:-iz, iz:-iz, :],
                        # )
                    else:
                        C_shear, C_slid_inside, C_grav, C_float = iceflow_energy_XY(
                            cfg, X[i : i + 1, :, :, :], Y[:, :, :, :]
                        )
                        # C_shear, C_grav, C_float = iceflow_energy_XY(
                        #     cfg, X[i : i + 1, :, :, :], Y[:, :, :, :]
                        # )

                    # print("SLIDING LOSSES COMPARED")
                    # print(C_slid_inside)
                    # print(C_slid)
                    # exit()
                    COST_all = (
                        # tf.reduce_mean(C_shear)
                        tf.reduce_mean(C_slid)
                        # + tf.reduce_mean(C_grav)
                        # + tf.reduce_mean(C_float)
                        + tf.reduce_mean(C_shear)
                        + tf.reduce_mean(C_grav)
                        + tf.reduce_mean(C_float)
                    )
                    
                    COST_no_sliding = (
                        # tf.reduce_mean(C_shear)
                        # + tf.reduce_mean(C_slid)
                        # + tf.reduce_mean(C_grav)
                        # + tf.reduce_mean(C_float)
                        tf.reduce_mean(C_shear)
                        + tf.reduce_mean(C_grav)
                        + tf.reduce_mean(C_float)
                    )
                    
                    
                    
                    # basis_vectors, sliding_shear_stress = tf.gradient(COS)

                    if (epoch + 1) % 100 == 0:
                        print(
                            "---------- > ",
                            # tf.reduce_mean(C_shear).numpy(),
                            # tf.reduce_mean(C_slid).numpy(),
                            # tf.reduce_mean(C_grav).numpy(),
                            # tf.reduce_mean(C_float).numpy(),
                            tf.reduce_mean(C_shear).numpy(),
                            tf.reduce_mean(C_grav).numpy(),
                            tf.reduce_mean(C_float).numpy(),
                        )

                    #                    state.C_shear = tf.pad(C_shear[0],[[0,1],[0,1]],"CONSTANT")
                    #                    state.C_slid  = tf.pad(C_slid[0],[[0,1],[0,1]],"CONSTANT")
                    #                    state.C_grav  = tf.pad(C_grav[0],[[0,1],[0,1]],"CONSTANT")
                    #                    state.C_float = C_float[0]

                    # print(state.C_shear.shape, state.C_slid.shape, state.C_grav.shape, state.C_float.shape,state.thk.shape )

                    cost_emulator = cost_emulator + COST_all

                    if (epoch + 1) % 100 == 0:
                        U, V = Y_to_UV(cfg, Y)
                        U = U[0]
                        V = V[0]
                        velsurf_mag = tf.sqrt(U[-1] ** 2 + V[-1] ** 2)
                        print("train : ", epoch, COST_all.numpy(), np.max(velsurf_mag))
                
                # basis_vectors, sliding_shear_stress_law = sliding_law(cfg, Y, effective_pressure)
                # print('what')
                # print(number_of_cells)
                sliding_shear_stress = t.gradient(sliding_loss, [U_basal, V_basal])

                # print("autograd",sliding_shear_stress)
                # print("manual",sliding_shear_stress_law)
                # exit()
                
                # Sliding loss gradients                
                sliding_gradients = t.gradient(
                    [U_basal, V_basal],
                    state.iceflow_model.trainable_variables,
                    output_gradients=sliding_shear_stress,
                )
                
                sliding_gradients_manual = t.gradient(
                    [U_basal, V_basal],
                    state.iceflow_model.trainable_variables,
                    output_gradients=sliding_shear_stress_law,
                )
                
                # for i in range(len(sliding_gradients_manual)):
                    # sliding_gradients_manual[i] = sliding_gradients_manual[i] / number_of_cells
                    
                # sliding_gradients_manual = sliding_gradients_manual / number_of_cells
                # print("autograd gradients",sliding_gradients[-1])
                # print("manual gradients",sliding_gradients_manual[-1])
                # exit()

                # All gradients other than sliding loss
                grads_no_sliding = t.gradient(COST_no_sliding, state.iceflow_model.trainable_variables)
                grads_all = t.gradient(COST_all, state.iceflow_model.trainable_variables)

                # Combining sliding loss gradients with other loss term gradients
                combined_gradients = [
                    grad + (sliding_grad / number_of_cells)
                    for grad, sliding_grad in zip(grads_no_sliding, sliding_gradients_manual)
                ]
                
                # print('final grads')
                # print(combined_gradients[-3], grads_all[-3])
                # exit()

                state.opti_retrain.apply_gradients(
                    zip(combined_gradients, state.iceflow_model.trainable_variables)
                )
                # state.opti_retrain.apply_gradients(
                #     zip(grads_all, state.iceflow_model.trainable_variables)
                # )

                state.opti_retrain.lr = (
                    cfg.modules.iceflow.iceflow.retrain_emulator_lr
                    * (0.95 ** (epoch / 1000))
                )

            state.COST_EMULATOR.append(cost_emulator)

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
