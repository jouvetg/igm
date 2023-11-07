
### <h1 align="center" id="title">IGM module flow_dt_thk </h1>

# Description:

This IGM module wraps up module iceflow, time and thk (check for each modules for the documentation).
 
# Parameters: 


|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--iflo_type`|`emulated`|Type of iceflow: it can emulated (default), solved, or in diagnostic mode to investigate the fidelity of the emulator towads the solver|
||`--iflo_emulator`|`myemulator`|Directory path of the deep-learning ice flow model, create a new if empty string|
||`--iflo_init_slidingco`|`0.0464`|Initial sliding coefficient slidingco|
||`--iflo_init_arrhenius`|`78`|Initial arrhenius factor arrhenuis|
||`--iflo_regu_glen`|`1e-05`|Regularization parameter for Glen's flow law|
||`--iflo_regu_weertman`|`1e-10`|Regularization parameter for Weertman's sliding law|
||`--iflo_exp_glen`|`3`|Glen's flow law exponent|
||`--iflo_exp_weertman`|`3`|Weertman's law exponent|
||`--iflo_gravity_cst`|`9.81`|Gravitational constant|
||`--iflo_ice_density`|`910`|Density of ice|
||`--iflo_new_friction_param`||Sliding coeeficient (this describe slidingco differently with slidingco**-(1.0 / exp_weertman) instead of slidingco as before)|
||`--iflo_Nz`|`10`|Number of grid point for the vertical discretization|
||`--iflo_vert_spacing`|`4.0`|Parameter controlling the discrtuzation density to get more point near the bed than near the the surface. 1.0 means equal vertical spacing.|
||`--iflo_thr_ice_thk`|`0.1`|Threshold Ice thickness for computing strain rate|
||`--iflo_solve_step_size`|`1`|Step size for the optimizer using when solving Blatter-Pattyn in solver mode|
||`--iflo_solve_nbitmax`|`100`|Maximum number of iteration for the optimizer using when solving Blatter-Pattyn in solver mode|
||`--iflo_solve_stop_if_no_decrease`||This permits to stop the solver if the energy does not decrease|
||`--iflo_fieldin`|`['thk', 'usurf', 'arrhenius', 'slidingco', 'dX']`|Input fields of the iceflow emulator|
||`--iflo_dim_arrhenius`|`2`|Dimension of the arrhenius factor (horizontal 2D or 3D)|
||`--iflo_retrain_emulator_freq`|`10`|Frequency at which the emulator is retrained, 0 means never, 1 means at each time step, 2 means every two time steps, etc.|
||`--iflo_retrain_emulator_lr`|`2e-05`|Learning rate for the retraining of the emulator|
||`--iflo_retrain_emulator_nbit_init`|`1`|Number of iterations done at the first time step for the retraining of the emulator|
||`--iflo_retrain_emulator_nbit`|`1`|Number of iterations done at each time step for the retraining of the emulator|
||`--iflo_retrain_emulator_framesizemax`|`750`|Size of the patch used for retraining the emulator, this is usefull for large size arrays, otherwise the GPU memory can be overloaded|
||`--iflo_multiple_window_size`|`0`|If a U-net, this force window size a multiple of 2**N|
||`--iflo_force_max_velbar`|`0`|This permits to artifically upper-bound velocities, active if > 0|
||`--iflo_network`|`cnn`|This is the type of network, it can be cnn or unet|
||`--iflo_activation`|`lrelu`|Activation function, it can be lrelu, relu, tanh, sigmoid, etc.|
||`--iflo_nb_layers`|`16`|Number of layers in the CNN|
||`--iflo_nb_blocks`|`4`|Number of block layer in the U-net|
||`--iflo_nb_out_filter`|`32`|Number of output filters in the CNN|
||`--iflo_conv_ker_size`|`3`|Size of the convolution kernel|
||`--iflo_dropout_rate`|`0`|Dropout rate in the CNN|
||`--iflo_exclude_borders`||This is a quick fix of the border issue, other the physics informed emaulator shows zero velocity at the border|
||`--time_start`|`2000.0`|Start modelling time|
||`--time_end`|`2100.0`|End modelling time|
||`--time_save`|`10`|Save result frequency for many modules (in year)|
||`--time_cfl`|`0.3`|CFL number for the stability of the mass conservation scheme, it must be below 1|
||`--time_step_max`|`1.0`|Maximum time step allowed, used only with slow ice|
