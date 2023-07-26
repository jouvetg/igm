
argmark
=======

# Usage:


```bash
usage: argmark [-h] [--type_iceflow TYPE_ICEFLOW] [--emulator EMULATOR]
               [--iceflow_physics ICEFLOW_PHYSICS] [--init_slidingco INIT_SLIDINGCO]
               [--init_arrhenius INIT_ARRHENIUS] [--regu_glen REGU_GLEN]
               [--regu_weertman REGU_WEERTMAN] [--exp_glen EXP_GLEN] [--exp_weertman EXP_WEERTMAN]
               [--Nz NZ] [--vert_spacing VERT_SPACING] [--thr_ice_thk THR_ICE_THK]
               [--solve_iceflow_step_size SOLVE_ICEFLOW_STEP_SIZE]
               [--solve_iceflow_nbitmax SOLVE_ICEFLOW_NBITMAX]
               [--stop_if_no_decrease STOP_IF_NO_DECREASE] [--fieldin FIELDIN]
               [--z_dept_arrhenius Z_DEPT_ARRHENIUS]
               [--retrain_iceflow_emulator_freq RETRAIN_ICEFLOW_EMULATOR_FREQ]
               [--retrain_iceflow_emulator_lr RETRAIN_ICEFLOW_EMULATOR_LR]
               [--retrain_iceflow_emulator_nbit RETRAIN_ICEFLOW_EMULATOR_NBIT]
               [--retrain_iceflow_emulator_framesizemax RETRAIN_ICEFLOW_EMULATOR_FRAMESIZEMAX]
               [--multiple_window_size MULTIPLE_WINDOW_SIZE] [--force_max_velbar FORCE_MAX_VELBAR]
               [--network NETWORK] [--activation ACTIVATION] [--nb_layers NB_LAYERS]
               [--nb_blocks NB_BLOCKS] [--nb_out_filter NB_OUT_FILTER]
               [--conv_ker_size CONV_KER_SIZE] [--dropout_rate DROPOUT_RATE] [--tstart TSTART]
               [--tend TEND] [--tsave TSAVE] [--cfl CFL] [--dtmax DTMAX]

```
# Arguments

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
