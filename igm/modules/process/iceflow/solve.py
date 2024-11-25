import numpy as np 
import tensorflow as tf 
from .utils import *
from .energy_iceflow import *

def initialize_iceflow_solver(params,state):

    if int(tf.__version__.split(".")[1]) <= 10:
        state.optimizer = getattr(tf.keras.optimizers,params.iflo_optimizer_solver)(
            learning_rate=params.iflo_solve_step_size
        )
    else:
        state.optimizer = getattr(tf.keras.optimizers.legacy,params.iflo_optimizer_solver)(
            learning_rate=params.iflo_solve_step_size
        )

def solve_iceflow(params, state, U, V):
    """
    solve_iceflow
    """

    Cost_Glen = []

    track_mem = False

    fieldin = [
        tf.expand_dims(vars(state)[f], axis=0) for f in params.iflo_fieldin
    ]
    
    early_stopping = EarlyStopping(relative_min_delta=params.iflo_solve_rel_min_delta, patience=params.iflo_solve_patience)

    for i in range(params.iflo_solve_nbitmax):
        with tf.GradientTape() as tape:
            tape.watch(U)
            tape.watch(V)

            if track_mem:
                gpu_info = tf.config.experimental.get_memory_info("GPU:0")
                gp0 = gpu_info['current']

            C_shear, C_slid, C_grav, C_float = iceflow_energy(
                params, tf.expand_dims(U, axis=0), tf.expand_dims(V, axis=0), fieldin
            )

            COST = tf.reduce_mean(C_shear) + tf.reduce_mean(C_slid) \
                 + tf.reduce_mean(C_grav)  + tf.reduce_mean(C_float)

            Cost_Glen.append(COST)

            if track_mem:
                gpu_info = tf.config.experimental.get_memory_info("GPU:0")
                gp1 = gpu_info['current']

            if track_mem:
                print(f"GPU memory consum for COST: {(gp1 - gp0) / 1024**2:.2f} MB")

            #if (i + 1) % 100 == 0:
            #    print("---------- > ", tf.reduce_mean(C_shear).numpy(), tf.reduce_mean(C_slid).numpy(), tf.reduce_mean(C_grav).numpy(), tf.reduce_mean(C_float).numpy())

#            state.C_shear = tf.pad(C_shear[0],[[0,1],[0,1]],"CONSTANT")
#            state.C_slid  = tf.pad(C_slid[0],[[0,1],[0,1]],"CONSTANT")
#            state.C_grav  = tf.pad(C_grav[0],[[0,1],[0,1]],"CONSTANT")
#            state.C_float = C_float[0] 

            # Stop if the cost no longer decreases
            # if params.iflo_solve_stop_if_no_decrease:
            #     if i > 1:
            #         if Cost_Glen[-1] >= Cost_Glen[-2]:
            #             break

        grads = tape.gradient(COST, [U, V])
 
        # gpu_info = tf.config.experimental.get_memory_info("GPU:0")
        # print(f"Solve Peak GPU memory (D): {gpu_info['current'] / 1024**2:.2f} MB")

        state.optimizer.apply_gradients(zip(grads, [U, V]))
 
        # if (i + 1) % 100 == 0:
        #     velsurf_mag = tf.sqrt(U[-1] ** 2 + V[-1] ** 2)
        #     print("solve :", i, COST.numpy(), np.max(velsurf_mag)) 

#        if i % 20 == 0:
#            print("Solve : ", i, ": ", COST.numpy())

        if early_stopping.should_stop(COST.numpy()): 
            break

    U = tf.where(state.thk > 0, U, 0)
    V = tf.where(state.thk > 0, V, 0)
 
    # print("Solve : ", i, ": ", COST.numpy(), " , max iceflow : ", tf.norm(tf.sqrt(U ** 2 + V ** 2),ord=np.inf).numpy())

    return U, V, Cost_Glen

def solve_iceflow_lbfgs(params, state, U, V):

    import tensorflow_probability as tfp

    Cost_Glen = []
 
    def COST(UV):

        U = UV[0]
        V = UV[1]

        fieldin = [
            tf.expand_dims(vars(state)[f], axis=0) for f in params.iflo_fieldin
        ]

        C_shear, C_slid, C_grav, C_float = iceflow_energy(
            params, tf.expand_dims(U, axis=0), tf.expand_dims(V, axis=0), fieldin
        )

        COST = tf.reduce_mean(C_shear) + tf.reduce_mean(C_slid) \
             + tf.reduce_mean(C_grav)  + tf.reduce_mean(C_float)
            
        return COST

    def loss_and_gradients_function(UV):
        with tf.GradientTape() as tape:
            tape.watch(UV)
            loss = COST(UV)
            Cost_Glen.append(loss)
            gradients = tape.gradient(loss, UV)
        return loss, gradients
    
    UV = tf.stack([U, V], axis=0) 

    optimizer = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=loss_and_gradients_function,
            initial_position=UV,
            max_iterations=params.iflo_solve_nbitmax,
            tolerance=1e-8)
    
    UV = optimizer.position

    U = UV[0]
    V = UV[1]
 
    return U, V, Cost_Glen

def update_iceflow_solved(params, state):

    if params.iflo_optimizer_lbfgs:
        U, V, Cost_Glen = solve_iceflow_lbfgs(params, state, state.U, state.V)
    else:
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
        
    if len(params.iflo_save_cost_solver)>0:
        np.savetxt(params.iflo_output_directory+params.iflo_save_cost_solver+'-'+str(state.it)+'.dat', np.array(Cost_Glen),  fmt="%5.10f")

    state.COST_Glen = Cost_Glen[-1].numpy()

    update_2d_iceflow_variables(params, state)
 