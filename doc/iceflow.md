### <h1 align="center" id="title">IGM module `iceflow` </h1>

# Description:

This IGM module models ice flow dynamics in 3D using a Convolutional Neural Network based on Physics Informed Neural Network as described in this [paper](https://eartharxiv.org/repository/view/5335/). In more details, we train a CNN to minimise the energy associated with high-order ice flow equations within the time iterations of a glacier evolution model. As a result, our iflo_emulator is a computationally-efficient alternative to traditional solvers, it is capable to handle a variety of ice flow regimes and memorize previous solutions.

Pre-trained emulators are provided by defaults (parameter `iflo_emulator`). However, a from scratch iflo_emulator can be requested with `iflo_emulator=""`. The most important parameters are:

- physical parameters 

```json 
"iflo_init_slidingco": 10000.0  # Init slid. coeff. ($Mpa^{-3} y^{-1} m$)
"iflo_init_arrhenius": 78.0     # Init Arrhenius cts ($Mpa^{-3} y^{-1}$)
"iflo_exp_glen": 3              # Glen's exponent
"iflo_exp_weertman":  3         # Weertman's sliding law exponent
```

- related to the vertical discretization:

```json 
"iflo_Nz": 10                 # number of vertical layers
"iflo_vert_spacing": 4.0     # 1.0 for equal vertical spacing, 4.0 otherwise
```

- learning rate and frequency of retraining:

```json 
"iflo_retrain_emulator_lr": 0.00002 
"iflo_retrain_emulator_freq": 5     
```

While this module was targeted for deep learning emulation, it important parameters for solving are :

is possible to
use the solver (`iflo_type='solved'`) instead of the default iflo_emulator (`iflo_type='emulated'`), or use the two together (`iflo_type='diagnostic'`) to assess the emaultor against the solver. Most important parameters for solving are :

```json 
"iflo_solve_step_size": 0.00002 
"iflo_solve_nbitmax": 5     
```

One may choose between 2D arrhenius factor by changing parameters between `iflo_dim_arrhenius=2` or `iflo_dim_arrhenius=3` -- le later is necessary for the enthalpy model.

When treating ery large arrays, retraining must be done sequentially patch-wise for memory reason. The size of the pathc is controlled by parameter `iflo_multiple_window_size=750`.

# Reference

```
@article{jouvet2023ice,
  title={Ice flow model emulator based on physics-informed deep learning},
  author={Jouvet, Guillaume and Cordonnier, Guillaume},
  year={2023},
  publisher={EarthArXiv}
}
```

 
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
