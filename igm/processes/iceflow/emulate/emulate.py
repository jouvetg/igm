#!/usr/bin/env python3

# Copyright (C) 2021-2025 IGM authors 
# Published under the GNU GPL (Version 3), check at the LICENSE file

import numpy as np 
import tensorflow as tf 
import os

from ..utils import *
from ..energy_iceflow.energy_iceflow import * 
from .neural_network import *
from ..emulate import emulators
import importlib_resources 
  
def initialize_iceflow_emulator(cfg, state):

    if (int(tf.__version__.split(".")[1]) <= 10) | (int(tf.__version__.split(".")[1]) >= 16) :
        state.opti_retrain = getattr(tf.keras.optimizers,cfg.processes.iceflow.optimizer_emulator)(
            learning_rate=cfg.processes.iceflow.retrain_emulator_lr,
            epsilon=cfg.processes.iceflow.optimizer_emulator_epsilon,
            clipnorm=cfg.processes.iceflow.optimizer_emulator_clipnorm
        )
    else:
        state.opti_retrain = getattr(tf.keras.optimizers.legacy,cfg.processes.iceflow.optimizer_emulator)( 
            learning_rate=cfg.processes.iceflow.retrain_emulator_lr,
            epsilon=cfg.processes.iceflow.optimizer_emulator_epsilon,
            clipnorm=cfg.processes.iceflow.optimizer_emulator_clipnorm
        )

    direct_name = (
        "pinnbp"
        + "_"
        + str(cfg.processes.iceflow.Nz)
        + "_"
        + str(int(cfg.processes.iceflow.vert_spacing))
        + "_"
    )
    direct_name += (
        cfg.processes.iceflow.network
        + "_"
        + str(cfg.processes.iceflow.nb_layers)
        + "_"
        + str(cfg.processes.iceflow.nb_out_filter)
        + "_"
    )
    direct_name += (
        str(cfg.processes.iceflow.dim_arrhenius)
        + "_"
        + str(int(cfg.processes.iceflow.new_friction_param))
    )

    if cfg.processes.iceflow.pretrained_emulator:
        if cfg.processes.iceflow.emulator == "":
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
            if os.path.exists(cfg.processes.iceflow.emulator):
                dirpath = cfg.processes.iceflow.emulator
                print("----------------------------------> Found pretrained emulator: " + cfg.processes.iceflow.emulator)
            else:
                print("----------------------------------> No pretrained emulator found ")

        fieldin = []
        fid = open(os.path.join(dirpath, "fieldin.dat"), "r")
        for fileline in fid:
            part = fileline.split()
            fieldin.append(part[0])
        fid.close()
        assert cfg.processes.iceflow.fieldin == fieldin
        state.iceflow_model = tf.keras.models.load_model(
            os.path.join(dirpath, "model.h5"), compile=False
        )
        state.iceflow_model.compile() 
    else:
        print("----------------------------------> No pretrained emulator, start from scratch.") 
        nb_inputs = len(cfg.processes.iceflow.fieldin) + (cfg.processes.iceflow.dim_arrhenius == 3) * (
            cfg.processes.iceflow.Nz - 1
        )
        nb_outputs = 2 * cfg.processes.iceflow.Nz
        state.iceflow_model = getattr(igm.processes.iceflow.emulate.emulate, cfg.processes.iceflow.network)(
            cfg, nb_inputs, nb_outputs
        )

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
    N = cfg.processes.iceflow.Nz

    fieldin = [vars(state)[f] for f in cfg.processes.iceflow.fieldin]

    X = fieldin_to_X(cfg, fieldin)

    if cfg.processes.iceflow.exclude_borders>0:
        iz = cfg.processes.iceflow.exclude_borders
        X = tf.pad(X, [[0, 0], [iz, iz], [iz, iz], [0, 0]], "SYMMETRIC")
        
    if cfg.processes.iceflow.multiple_window_size==0:
        Y = state.iceflow_model(X)
    else:
        Y = state.iceflow_model(tf.pad(X, state.PAD, "CONSTANT"))[:, :Ny, :Nx, :]

    if cfg.processes.iceflow.exclude_borders>0:
        iz = cfg.processes.iceflow.exclude_borders
        Y = Y[:, iz:-iz, iz:-iz, :]

    U, V = Y_to_UV(cfg, Y)
    U = U[0]
    V = V[0]

    state.U = tf.where(state.thk > 0, U, 0)
    state.V = tf.where(state.thk > 0, V, 0)

    # If requested, the speeds are artifically upper-bounded
    if cfg.processes.iceflow.force_max_velbar > 0:
        velbar_mag = getmag3d(state.U, state.V)
        state.U = \
            tf.where(
                velbar_mag >= cfg.processes.iceflow.force_max_velbar,
                cfg.processes.iceflow.force_max_velbar * (state.U / velbar_mag),
                state.U,
            )
        state.V = \
            tf.where(
                velbar_mag >= cfg.processes.iceflow.force_max_velbar,
                cfg.processes.iceflow.force_max_velbar * (state.V / velbar_mag),
                state.V,
            ) 

    update_2d_iceflow_variables(cfg, state)


def update_iceflow_emulator(cfg, state, it):
 
    run_it = False
    if cfg.processes.iceflow.retrain_emulator_freq > 0:
       run_it = (it % cfg.processes.iceflow.retrain_emulator_freq == 0)
 
    warm_up = int(it <= cfg.processes.iceflow.retrain_warm_up_it)

    if (warm_up | run_it):
        
        fieldin = [vars(state)[f] for f in cfg.processes.iceflow.fieldin]

########################

        # thkext = tf.pad(state.thk,[[1,1],[1,1]],"CONSTANT",constant_values=1)
        # # this permits to locate the calving front in a cell in the 4 directions
        # state.CF_W = tf.where((state.thk>0)&(thkext[1:-1,:-2]==0),1.0,0.0)
        # state.CF_E = tf.where((state.thk>0)&(thkext[1:-1,2:]==0),1.0,0.0) 
        # state.CF_S = tf.where((state.thk>0)&(thkext[:-2,1:-1]==0),1.0,0.0)
        # state.CF_N = tf.where((state.thk>0)&(thkext[2:,1:-1]==0),1.0,0.0)

########################

        XX = fieldin_to_X(cfg, fieldin)

        X = split_into_patches(XX, cfg.processes.iceflow.retrain_emulator_framesizemax)
        
        Ny = X.shape[1]
        Nx = X.shape[2]
        
        PAD = compute_PAD(cfg,Nx,Ny)

        state.COST_EMULATOR = []

        if warm_up:
            nbit = cfg.processes.iceflow.retrain_emulator_nbit_init
            state.opti_retrain.lr = cfg.processes.iceflow.retrain_emulator_lr_init
        else:
            nbit = cfg.processes.iceflow.retrain_emulator_nbit
            state.opti_retrain.lr = cfg.processes.iceflow.retrain_emulator_lr

        iz = cfg.processes.iceflow.exclude_borders 

        for epoch in range(nbit):
            cost_emulator = tf.Variable(0.0)

            for i in range(X.shape[0]):
                with tf.GradientTape() as t:

                    Y = state.iceflow_model(tf.pad(X[i:i+1, :, :, :], PAD, "CONSTANT"))[:,:Ny,:Nx,:]
                    
                    if iz>0:
                        C_shear, C_slid, C_grav, C_float = iceflow_energy_XY(cfg, X[i : i + 1, iz:-iz, iz:-iz, :], Y[:, iz:-iz, iz:-iz, :])
                    else:
                        C_shear, C_slid, C_grav, C_float = iceflow_energy_XY(cfg, X[i : i + 1, :, :, :], Y[:, :, :, :])
 
                    COST = tf.reduce_mean(C_shear) + tf.reduce_mean(C_slid) \
                         + tf.reduce_mean(C_grav)  + tf.reduce_mean(C_float)
                    
                    if (epoch + 1) % 100 == 0:
                        print("---------- > ", tf.reduce_mean(C_shear).numpy(), tf.reduce_mean(C_slid).numpy(), tf.reduce_mean(C_grav).numpy(), tf.reduce_mean(C_float).numpy())

#                    state.C_shear = tf.pad(C_shear[0],[[0,1],[0,1]],"CONSTANT")
#                    state.C_slid  = tf.pad(C_slid[0],[[0,1],[0,1]],"CONSTANT")
#                    state.C_grav  = tf.pad(C_grav[0],[[0,1],[0,1]],"CONSTANT")
#                    state.C_float = C_float[0] 

                    # print(state.C_shear.shape, state.C_slid.shape, state.C_grav.shape, state.C_float.shape,state.thk.shape )

                    cost_emulator = cost_emulator + COST

                    if (epoch + 1) % 100 == 0:
                        U, V = Y_to_UV(cfg, Y)
                        U = U[0]
                        V = V[0]
                        velsurf_mag = tf.sqrt(U[-1] ** 2 + V[-1] ** 2)
                        print("train : ", epoch, COST.numpy(), np.max(velsurf_mag))

                grads = t.gradient(COST, state.iceflow_model.trainable_variables)

                state.opti_retrain.apply_gradients(
                    zip(grads, state.iceflow_model.trainable_variables)
                )

#                gradient_norm = tf.linalg.global_norm(grads)

                state.opti_retrain.lr = cfg.processes.iceflow.retrain_emulator_lr * (
                    0.95 ** (epoch / 1000)
                )

            state.COST_EMULATOR.append(cost_emulator)
            
    
    if len(cfg.processes.iceflow.save_cost_emulator)>0:
        np.savetxt(cfg.processes.iceflow.output_directory+cfg.processes.iceflow.save_cost_emulator+'-'+str(it)+'.dat', np.array(state.COST_EMULATOR), fmt="%5.10f")

def split_into_patches(X, nbmax):
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

    #    fieldin_dim=[0,0,1*(cfg.processes.iceflow.dim_arrhenius==3),0,0]

    fid = open(os.path.join(directory, "fieldin.dat"), "w")
    #    for key,gg in zip(cfg.processes.iceflow.fieldin,fieldin_dim):
    #        fid.write("%s %.1f \n" % (key, gg))
    for key in cfg.processes.iceflow.fieldin:
        print(key)
        fid.write("%s \n" % (key))
    fid.close()

    fid = open(os.path.join(directory, "vert_grid.dat"), "w")
    fid.write("%4.0f  %s \n" % (cfg.processes.iceflow.Nz, "# number of vertical grid point (Nz)"))
    fid.write(
        "%2.2f  %s \n"
        % (cfg.processes.iceflow.vert_spacing, "# param for vertical spacing (vert_spacing)")
    )
    fid.close()
