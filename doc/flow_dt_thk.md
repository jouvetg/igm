
### <h1 align="center" id="title">IGM module `flow_dt_thk` </h1>

# Description:

This IGM module wraps up modules `iceflow`, `time_step` and `thk` (check for each modules for the documentation). Calling module `flow_dt_thk` is the same as calling modules [`iceflow`, `time_step`, `thk`] in order.
 
# Parameters: 


|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--type_iceflow`|`emulated`|Type of iceflow: it can emulated (default), solved, or in diagnostic mode to investigate the fidelity of the emulator towads the solver|
||`--emulator`|`f21_pinnbp_GJ_23_a`|Directory path of the deep-learning ice flow model, create a new if empty string|
||`--iceflow_physics`|`2`|2 for blatter, 4 for stokes, this is also the number of DOF (STOKES DOES NOT WORK YET, KEEP IT TO 2)|
||`--init_slidingco`|`10000`|Initial sliding coefficient slidingco|
||`--init_arrhenius`|`78`|Initial arrhenius factor arrhenuis|
||`--regu_glen`|`1e-05`|Regularization parameter for Glen's flow law|
||`--regu_weertman`|`1e-10`|Regularization parameter for Weertman's sliding law|
||`--exp_glen`|`3`|Glen's flow law exponent|
||`--exp_weertman`|`3`|Weertman's law exponent|
||`--gravity_cst`|`9.81`|Gravitational constant|
||`--ice_density`|`910`|Density of ice|
||`--Nz`|`10`|Number of grid point for the vertical discretization|
||`--vert_spacing`|`4.0`|Parameter controlling the discrtuzation density to get more point near the bed than near the the surface. 1.0 means equal vertical spacing.|
||`--thr_ice_thk`|`0.1`|Threshold Ice thickness for computing strain rate|
||`--solve_iceflow_step_size`|`1`|Step size for the optimizer using when solving Blatter-Pattyn in solver mode|
||`--solve_iceflow_nbitmax`|`100`|Maximum number of iteration for the optimizer using when solving Blatter-Pattyn in solver mode|
||`--stop_if_no_decrease`||This permits to stop the solver if the energy does not decrease|
||`--fieldin`|`['thk', 'usurf', 'arrhenius', 'slidingco', 'dX']`|Input fields of the iceflow emulator|
||`--dim_arrhenius`|`2`|Dimension of the arrhenius factor (horizontal 2D or 3D)|
||`--retrain_iceflow_emulator_freq`|`10`|Frequency at which the emulator is retrained, 0 means never, 1 means at each time step, 2 means every two time steps, etc.|
||`--retrain_iceflow_emulator_lr`|`2e-05`|Learning rate for the retraining of the emulator|
||`--retrain_iceflow_emulator_nbit`|`1`|Number of iterations done at each time step for the retraining of the emulator|
||`--retrain_iceflow_emulator_framesizemax`|`750`|Size of the patch used for retraining the emulator, this is usefull for large size arrays, otherwise the GPU memory can be overloaded|
||`--multiple_window_size`|`0`|If a U-net, this force window size a multiple of 2**N|
||`--force_max_velbar`|`0`|This permits to artifically upper-bound velocities, active if > 0|
||`--network`|`cnn`|This is the type of network, it can be cnn or unet|
||`--activation`|`lrelu`|Activation function, it can be lrelu, relu, tanh, sigmoid, etc.|
||`--nb_layers`|`16`|Number of layers in the CNN|
||`--nb_blocks`|`4`|Number of block layer in the U-net|
||`--nb_out_filter`|`32`|Number of output filters in the CNN|
||`--conv_ker_size`|`3`|Size of the convolution kernel|
||`--dropout_rate`|`0`|Dropout rate in the CNN|
||`--time_start`|`2000.0`|Start modelling time|
||`--time_end`|`2100.0`|End modelling time|
||`--time_save`|`10`|Save result frequency for many modules (in year)|
||`--cfl`|`0.3`|CFL number for the stability of the mass conservation scheme, it must be below 1|
||`--time_step_max`|`1.0`|Maximum time step allowed, used only with slow ice|
