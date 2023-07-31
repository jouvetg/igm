
### <h1 align="center" id="title">IGM module flow_dt_thk </h1>

# Description:

This IGM module wraps up module iceflow, time_step and thk (check for each modules for the documentation).
 
# Parameters: 


|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--type_iceflow`|`emulated`|emulated, solved, diagnostic|
||`--emulator`|`f21_pinnbp_GJ_23_a`|Directory path of the deep-learning ice flow model,               create a new if empty string|
||`--iceflow_physics`|`2`|2 for blatter, 4 for stokes, this is also the number of DOF|
||`--init_slidingco`|`10000`|Initial sliding coeeficient slidingco (default: 0)|
||`--init_arrhenius`|`78`|Initial arrhenius factor arrhenuis (default: 78)|
||`--regu_glen`|`1e-05`|Regularization parameter for Glen's flow law|
||`--regu_weertman`|`1e-10`|Regularization parameter for Weertman's sliding law|
||`--exp_glen`|`3`|Glen's flow law exponent|
||`--exp_weertman`|`3`|Weertman's law exponent|
||`--Nz`|`10`|Nz for the vertical discretization|
||`--vert_spacing`|`4.0`|1.0 for equal vertical spacing, 4.0 otherwise (4.0)|
||`--thr_ice_thk`|`0.1`|Threshold Ice thickness for computing strain rate|
||`--solve_iceflow_step_size`|`1`|solver_step_size|
||`--solve_iceflow_nbitmax`|`100`|solver_nbitmax|
||`--stop_if_no_decrease`||stop_if_no_decrease for the solver|
||`--fieldin`|`['thk', 'usurf', 'arrhenius', 'slidingco', 'dX']`|Input parameter of the iceflow emulator|
||`--z_dept_arrhenius`||dimension of each field in z|
||`--retrain_iceflow_emulator_freq`|`10`|retrain_iceflow_emulator_freq|
||`--retrain_iceflow_emulator_lr`|`2e-05`|retrain_iceflow_emulator_lr|
||`--retrain_iceflow_emulator_nbit`|`1`|retrain_iceflow_emulator_nbit|
||`--retrain_iceflow_emulator_framesizemax`|`750`|retrain_iceflow_emulator_framesizemax|
||`--multiple_window_size`|`0`|If a U-net, this force window size a multiple of 2**N (default: 0)|
||`--force_max_velbar`|`0`|This permits to artif. upper-bound velocities, active if > 0 (default: 0)|
||`--network`|`cnn`|This is the type of network, it can be cnn or unet|
||`--activation`|`lrelu`|lrelu|
||`--nb_layers`|`16`|nb_layers|
||`--nb_blocks`|`4`|Number of block layer in the U-net|
||`--nb_out_filter`|`32`|nb_out_filter|
||`--conv_ker_size`|`3`|conv_ker_size|
||`--dropout_rate`|`0`|dropout_rate|
||`--tstart`|`2000.0`|Start modelling time (default 2000)|
||`--tend`|`2100.0`|End modelling time (default: 2100)|
||`--tsave`|`10`|Save result each X years (default: 10)|
||`--cfl`|`0.3`|CFL number for the stability of the mass conservation scheme,         it must be below 1 (Default: 0.3)|
||`--dtmax`|`10.0`|Maximum time step allowed, used only with slow ice (default: 10.0)|
