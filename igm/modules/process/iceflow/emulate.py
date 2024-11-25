
import numpy as np 
import tensorflow as tf 
import os

from .utils import *
from .energy_iceflow import *
from .neural_network import *

from igm import emulators
import importlib_resources
  
def initialize_iceflow_emulator(params,state):

    if (int(tf.__version__.split(".")[1]) <= 10) | (int(tf.__version__.split(".")[1]) >= 16) :
        state.opti_retrain = getattr(tf.keras.optimizers,params.iflo_optimizer_emulator)(
            learning_rate=params.iflo_retrain_emulator_lr, # Default is 0.001
#            beta_1=0.9,           # Change beta1 from default 0.9
#            beta_2=0.999,         # Default is 0.999
#            epsilon=1e-3          # Default is 1e-7
        )
    else:
        state.opti_retrain = getattr(tf.keras.optimizers.legacy,params.iflo_optimizer_emulator)( 
            learning_rate=params.iflo_retrain_emulator_lr, # Default is 0.001
#            beta_1=0.9,           # Change beta1 from default 0.9
#            beta_2=0.999,         # Default is 0.999
#            epsilon=1e-3         # Default is 1e-7
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
                print("----------------------------------> Found pretrained emulator: " + params.iflo_emulator)
            else:
                print("----------------------------------> No pretrained emulator found ")

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
        print("----------------------------------> No pretrained emulator, start from scratch.") 
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
        elif params.iflo_network=='fourier':
            state.iceflow_model = fourier(params, nb_inputs, nb_outputs)

    # direct_name = 'pinnbp_10_4_cnn_16_32_2_1'        
    # dirpath = importlib_resources.files(emulators).joinpath(direct_name)
    # iceflow_model_pretrained = tf.keras.models.load_model(
    #     os.path.join(dirpath, "model.h5"), compile=False
    # )
    # N=16
    # pretrained_weights = [layer.get_weights() for layer in iceflow_model_pretrained.layers[:N]]
    # for i in range(N):
    #     state.iceflow_model.layers[i].set_weights(pretrained_weights[i])

def update_iceflow_emulated(params, state):
    # Define the input of the NN, include scaling

    Ny, Nx = state.thk.shape
    N = params.iflo_Nz

    fieldin = [vars(state)[f] for f in params.iflo_fieldin]

    X = fieldin_to_X(params, fieldin)

    if params.iflo_exclude_borders>0:
        iz = params.iflo_exclude_borders
        X = tf.pad(X, [[0, 0], [iz, iz], [iz, iz], [0, 0]], "SYMMETRIC")
        
    if params.iflo_multiple_window_size==0:
        Y = state.iceflow_model(X)
    else:
        Y = state.iceflow_model(tf.pad(X, state.PAD, "CONSTANT"))[:, :Ny, :Nx, :]

    if params.iflo_exclude_borders>0:
        iz = params.iflo_exclude_borders
        Y = Y[:, iz:-iz, iz:-iz, :]

    U, V = Y_to_UV(params, Y)
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


def update_iceflow_emulator(params, state):

    early_stopping = EarlyStopping(relative_min_delta=params.iflo_emulate_misfit_rel_min_delta, 
                                   patience=params.iflo_emulate_misfit_patience)
    
    if (state.it < 0) | (state.it % params.iflo_retrain_emulator_freq == 0):
        fieldin = [vars(state)[f] for f in params.iflo_fieldin]

########################

        # thkext = tf.pad(state.thk,[[1,1],[1,1]],"CONSTANT",constant_values=1)
        # # this permits to locate the calving front in a cell in the 4 directions
        # state.CF_W = tf.where((state.thk>0)&(thkext[1:-1,:-2]==0),1.0,0.0)
        # state.CF_E = tf.where((state.thk>0)&(thkext[1:-1,2:]==0),1.0,0.0) 
        # state.CF_S = tf.where((state.thk>0)&(thkext[:-2,1:-1]==0),1.0,0.0)
        # state.CF_N = tf.where((state.thk>0)&(thkext[2:,1:-1]==0),1.0,0.0)

########################

        track_mem = False

        XX = fieldin_to_X(params, fieldin)

        X = _split_into_patches(XX, params.iflo_retrain_emulator_framesizemax)
        
        Ny = X.shape[1]
        Nx = X.shape[2]
        
        PAD = compute_PAD(params,Nx,Ny)

        state.COST_EMULATOR = []

        nbit = int((state.it >= 0) * params.iflo_retrain_emulator_nbit + (
            state.it < 0
        ) * params.iflo_retrain_emulator_nbit_init)

        iz = params.iflo_exclude_borders 

        for epoch in range(nbit):
            cost_emulator = tf.Variable(0.0)

            for i in range(X.shape[0]):
                with tf.GradientTape() as tape:

                    if track_mem:
                        gpu_info = tf.config.experimental.get_memory_info("GPU:0")
                        gp0 = gpu_info['current']

                    Y = state.iceflow_model(tf.pad(X[i:i+1, :, :, :], PAD, "CONSTANT"))[:,:Ny,:Nx,:]
 
                    if track_mem:
                        gpu_info = tf.config.experimental.get_memory_info("GPU:0")
                        gp1 = gpu_info['current']
                    
                    if iz>0:
                        C_shear, C_slid, C_grav, C_float = iceflow_energy_XY(params, X[i : i + 1, iz:-iz, iz:-iz, :], Y[:, iz:-iz, iz:-iz, :])
                    else:
                        C_shear, C_slid, C_grav, C_float = iceflow_energy_XY(params, X[i : i + 1, :, :, :], Y[:, :, :, :])

                    if track_mem:
                        gpu_info = tf.config.experimental.get_memory_info("GPU:0")
                        gp2 = gpu_info['current']
 
                    COST = tf.reduce_mean(C_shear) + tf.reduce_mean(C_slid) \
                         + tf.reduce_mean(C_grav)  + tf.reduce_mean(C_float)
                    
                    # if (epoch + 1) % 100 == 0:
                    #     print("---------- > ", tf.reduce_mean(C_shear).numpy(), tf.reduce_mean(C_slid).numpy(), tf.reduce_mean(C_grav).numpy(), tf.reduce_mean(C_float).numpy())

#                    state.C_shear = tf.pad(C_shear[0],[[0,1],[0,1]],"CONSTANT")
#                    state.C_slid  = tf.pad(C_slid[0],[[0,1],[0,1]],"CONSTANT")
#                    state.C_grav  = tf.pad(C_grav[0],[[0,1],[0,1]],"CONSTANT")
#                    state.C_float = C_float[0] 

                    # print(state.C_shear.shape, state.C_slid.shape, state.C_grav.shape, state.C_float.shape,state.thk.shape )

                    cost_emulator = cost_emulator + COST

                    # if (epoch + 1) % 100 == 0:
                    #     U, V = Y_to_UV(params, Y)
                    #     U = U[0]
                    #     V = V[0]
                    #     velsurf_mag = tf.sqrt(U[-1] ** 2 + V[-1] ** 2)
                    #     print("train : ", epoch, COST.numpy(), np.max(velsurf_mag))

                    if track_mem:
                        print(f"GPU memory consum for NN and COST: {(gp1 - gp0) / 1024**2:.2f} MB and {(gp2 - gp1) / 1024**2:.2f} MB")

                # gpu_info = tf.config.experimental.get_memory_info("GPU:0")
                # print(f"Emulate Peak GPU memory: {gpu_info['current'] / 1024**2:.2f} MB")

                grads = tape.gradient(COST, state.iceflow_model.trainable_variables)
 
                #if (epoch + 1) % 100 == 0:
                #    print("Gradient norm:", tf.linalg.global_norm(grads).numpy())

                state.opti_retrain.apply_gradients(
                    zip(grads, state.iceflow_model.trainable_variables)
                )

                state.opti_retrain.lr = params.iflo_retrain_emulator_lr * (
                    0.95 ** (epoch / 1000)
                )

            state.COST_EMULATOR.append(cost_emulator)

            if early_stopping.should_stop(np.sqrt(COST.numpy())):
                break
    
        # print("Emule : ", epoch, ": ", COST.numpy())

    if len(params.iflo_save_cost_emulator)>0:
        np.savetxt(params.iflo_output_directory+params.iflo_save_cost_emulator+'-'+str(state.it)+'.dat', np.array(state.COST_EMULATOR), fmt="%5.10f")

 
# def _update_iceflow_emulator_lbfgs(params, state):

#     import tensorflow_probability as tfp

#     Cost_Glen = [0]
 
#     def COST(iceflow_model):

#         fieldin = [vars(state)[f] for f in params.iflo_fieldin]
 
#         X = fieldin_to_X(params, fieldin) 

#         Y = iceflow_model(X)
        
#         C_shear, C_slid, C_grav, C_float = iceflow_energy_XY(params, X, Y)

#         COST = tf.reduce_mean(C_shear) + tf.reduce_mean(C_slid) \
#              + tf.reduce_mean(C_grav)  + tf.reduce_mean(C_float)
           
#         return COST

#     def loss_and_gradients_function(trainable_variables):
#         with tf.GradientTape() as tape:
#             tape.watch(trainable_variables)
#             loss = COST(iceflow_model)
#             gradients = tape.gradient(loss, trainable_variables)
#         return loss, gradients
    
#     if (state.it < 0) | (state.it % params.iflo_retrain_emulator_freq == 0):
  
#         state.COST_EMULATOR = [0]
  
#         trainable_variables = tfp.optimizer.lbfgs_minimize(
#                 value_and_gradients_function=loss_and_gradients_function,
#                 initial_position=trainable_variables,
#                 max_iterations=params.iflo_retrain_emulator_nbit,
#                 tolerance=1e-8)

# #    if len(params.iflo_save_cost_emulator)>0:
# #         np.savetxt(params.iflo_output_directory+params.iflo_save_cost_emulator+'-'+str(state.it)+'.dat', np.array(state.COST_EMULATOR), fmt="%5.10f")
 


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


def save_iceflow_model(params, state):
    directory = "iceflow-model"
    
    import shutil

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
