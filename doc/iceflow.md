### <h1 align="center" id="title">IGM module iceflow </h1>

# Description:

This IGM module models ice flow using a Convolutional Neural Network based on 
Physics Informed Neural Network as described in this 
[paper](https://eartharxiv.org/repository/view/5335/).
You may find pre-trained and ready-to-use ice 
flow emulators, e.g. using the default emulator = f21_pinnbp_GJ_23_a, or using 
an initial untrained with emulator =''. Most important parameters are

- physical parameters (init_slidingco,init_arrhenius, exp_glen,exp_weertman)
- vertical pdiscrezation params (Nz,vert_spacing)
- learning rate (retrain_iceflow_emulator_lr)
- retraining frequency (retrain_iceflow_emulator_freq)
- CNN parameters (...)

While this module was targeted for deep learning emulation, it is possible to
use the solver (setting params.type_iceflow='solved'), or use the two together
(setting params.type_iceflow='diagnostic') to assess the emaultor against the
solver. Most important parameters for solving are :

- solve_iceflow_step_size
- solve_iceflow_nbitmax


 
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
||`--gravity_cst`|`9.81`|Gravitational constant|
||`--ice_density`|`910`|Density of ice|
||`--Nz`|`10`|Nz for the vertical discretization|
||`--vert_spacing`|`4.0`|1.0 for equal vertical spacing, 4.0 otherwise (4.0)|
||`--thr_ice_thk`|`0.1`|Threshold Ice thickness for computing strain rate|
||`--solve_iceflow_step_size`|`1`|solver_step_size|
||`--solve_iceflow_nbitmax`|`100`|solver_nbitmax|
||`--stop_if_no_decrease`||stop_if_no_decrease for the solver|
||`--fieldin`|`['thk', 'usurf', 'arrhenius', 'slidingco', 'dX']`|Input parameter of the iceflow emulator|
||`--dim_arrhenius`|`2`|dimension of the arrhenius factor (horizontal 2D or 3D)|
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
