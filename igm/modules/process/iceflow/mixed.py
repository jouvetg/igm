import numpy as np 
import tensorflow as tf 

from .utils import *
from .solve import *
from .emulate import *

def initialize_iceflow_mixed(params,state):

    initialize_iceflow_emulator(params,state)
    
    initialize_iceflow_solver(params,state)

    state.Utar = tf.zeros((params.iflo_Nz, state.thk.shape[0], state.thk.shape[1])) 
    state.Vtar = tf.zeros((params.iflo_Nz, state.thk.shape[0], state.thk.shape[1]))

    state.Uexa = tf.zeros((params.iflo_Nz, state.thk.shape[0], state.thk.shape[1]))
    state.Vexa = tf.zeros((params.iflo_Nz, state.thk.shape[0], state.thk.shape[1]))

    state.mixed_misfit = []

def update_iceflow_mixed(params, state):
    
    cold = (not params.iflo_pretrained_emulator) & (state.it < params.iflo_pretraining_nbit)

    # print("Status cold ? at it : ", cold, state.it)

    if cold:
 
        state.Utar, state.Vtar, Cost_Glen = solve_iceflow(params, state, state.Utar, state.Vtar)
 
        # ubartar = tf.reduce_sum(state.vert_weight * Utar, axis=0)
        # vbartar = tf.reduce_sum(state.vert_weight * Vtar, axis=0)
        # state.velbartar = tf.sqrt(ubartar ** 2 + vbartar ** 2)
 
        Cost_Emulator = update_iceflow_emulator_misfit(params, state, state.Utar , state.Vtar)
 
    else:
       
        if params.iflo_retrain_emulator_freq > 0:
            if state.it % params.iflo_retrain_emulator_freq == 0:
                update_iceflow_emulator(params, state)

    update_iceflow_emulated(params, state)

    # ubar = tf.reduce_sum(state.vert_weight * state.U, axis=0)
    # vbar = tf.reduce_sum(state.vert_weight * state.V, axis=0)

    if params.iflow_track_error:

        state.Uexa, state.Vexa, Cost_Glen = solve_iceflow(params, state, state.Uexa, state.Vexa)
  
        l1,l2 = compute_l1l2bar(state, state.Uexa - state.U, state.Vexa - state.V)

        state.mixed_misfit.append([int(cold), state.it, state.t.numpy(),
                                   l1, l2, Cost_Glen[-1].numpy(), state.COST_EMULATOR[-1].numpy()])

def finalize_iceflow_mixed(params, state):

    if params.iflow_track_error:
        
        state.mixed_misfit = np.stack(state.mixed_misfit)
        np.savetxt("errors_mixed.txt", state.mixed_misfit, delimiter=",", fmt="%5.5f")

        # import matplotlib.pyplot as plt

        # data = np.loadtxt("errors_mixed.txt", delimiter=",")
        # time = data[:, 0]  # First column for time
        # c = data[:, 2]    # Fifth column for l1
        # e = data[:, 3]    # Sixth column for l2
    
        # # Create a figure and axis
        # fig, ax1 = plt.subplots()
        
        # # Plot l1 against time
        # ax1.set_xlabel('Time')
        # ax1.set_ylabel('cost glen', color='tab:blue')
        # ax1.plot(time, c, color='tab:blue', label='cost', marker='o')
        # ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        # # Create a second y-axis to plot l2
        # ax2 = ax1.twinx()
        # ax2.set_ylabel('error emulator', color='tab:red')
        # ax2.plot(time, e, color='tab:red', label='error', marker='s')
        # ax2.tick_params(axis='y', labelcolor='tab:red')
        
        # # Add a title and show the grid
        # plt.title('Plot of cost and error against Time')
        # ax1.grid()
        
        # # Show the plot
        # plt.savefig("mixed.png")
    

def update_iceflow_emulator_misfit(params, state, Utar, Vtar):

    early_stopping = EarlyStopping(relative_min_delta=params.iflo_emulate_misfit_rel_min_delta, 
                                   patience=params.iflo_emulate_misfit_patience)

    fieldin = [vars(state)[f] for f in params.iflo_fieldin]

    X = fieldin_to_X(params, fieldin)

    Ny = X.shape[1]
    Nx = X.shape[2]
    
    PAD = compute_PAD(params,Nx,Ny)

    COST_EMULATOR = []

    if state.it <= 0:
        nbit = params.iflo_retrain_emulator_nbit_init
    else:
        nbit = params.iflo_retrain_emulator_nbit
  
    for i in range(nbit):
 
        with tf.GradientTape() as tape:

            # gpu_info = tf.config.experimental.get_memory_info("GPU:0")
            # print(f"Emulate Peak GPU memory (A): {gpu_info['current'] / 1024**2:.2f} MB")

            Y = state.iceflow_model(tf.pad(X, PAD, "CONSTANT"))[:,:Ny,:Nx,:]

            U, V = Y_to_UV(params, Y)
 
            # gpu_info = tf.config.experimental.get_memory_info("GPU:0")
            # print(f"Emulate Peak GPU memory (B): {gpu_info['current'] / 1024**2:.2f} MB")

            COST = tf.reduce_mean( (U[0]-Utar)**2 + (V[0]-Vtar)**2 )

            COST_EMULATOR.append(COST)

            # if (i + 1) % 100 == 0:  
            #     velsurf_mag = tf.sqrt((U[0][-1]-Utar[-1])**2 + (V[0][-1]-Vtar[-1])**2)
            #     print("train : ", i, COST.numpy(), np.max(velsurf_mag))
 
        # gpu_info = tf.config.experimental.get_memory_info("GPU:0")
        # print(f"Emulate Peak GPU memory (C): {gpu_info['current'] / 1024**2:.2f} MB")

        grads = tape.gradient(COST, state.iceflow_model.trainable_variables)

        # gpu_info = tf.config.experimental.get_memory_info("GPU:0")
        # print(f"Emulate Peak GPU memory (D): {gpu_info['current'] / 1024**2:.2f} MB")

        state.opti_retrain.apply_gradients(
            zip(grads, state.iceflow_model.trainable_variables)
        )

        # gpu_info = tf.config.experimental.get_memory_info("GPU:0")
        # print(f"Emulate Peak GPU memory (E): {gpu_info['current'] / 1024**2:.2f} MB")

        # state.opti_retrain.lr = params.iflo_retrain_emulator_lr * (
        #     0.95 ** (i / 1000)
        # )

        # if i % 20 == 0:
        #     print("Emul. : ", i, ": ", np.sqrt(COST.numpy()))

        if early_stopping.should_stop(np.sqrt(COST.numpy())):
            break

    
    print("EmuMi : ", i, ": ", np.sqrt(COST.numpy()), " , max iceflow : ", tf.norm(tf.sqrt(U[0] ** 2 + V[0] ** 2),ord=np.inf).numpy())

    return COST_EMULATOR


def compute_l1l2bar(state, U, V):

    ubar = tf.reduce_sum(state.vert_weight * U, axis=0)
    vbar = tf.reduce_sum(state.vert_weight * V, axis=0)

    VEL = tf.stack([ubar, vbar], axis=0)
    MA = tf.where(state.thk > 1, tf.ones_like(VEL), 0) 

    nl1 = tf.reduce_sum(MA * tf.abs(VEL)) / tf.reduce_sum(MA)
    nl2 = tf.reduce_sum(MA * tf.abs(VEL) ** 2) / tf.reduce_sum(MA)

    return nl1.numpy(), np.sqrt(nl2.numpy())
