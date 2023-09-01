### <h1 align="center" id="title">IGM module `iceflow` </h1>

# Description:

This IGM module models ice flow dynamics in 3D using a Convolutional Neural Network based on Physics Informed Neural Network as described in this [paper](https://eartharxiv.org/repository/view/5335/). In more details, we train a CNN to minimise the energy associated with high-order ice flow equations within the time iterations of a glacier evolution model. As a result, our emulator is a computationally-efficient alternative to traditional solvers, it is capable to handle a variety of ice flow regimes and memorize previous solutions.

Pre-trained emulators are provided by defaults (parameter `emulator`). However, a from scratch emulator can be requested with `emulator=""`. The most important parameters are:

- physical parameters 

```json 
"init_slidingco": 10000.0  # Init slid. coeff. ($Mpa^{-3} y^{-1} m$)
"init_arrhenius": 78.0     # Init Arrhenius cts ($Mpa^{-3} y^{-1}$)
"exp_glen": 3              # Glen's exponent
"exp_weertman":  3         # Weertman's sliding law exponent
```

- related to the vertical discretization:

```json 
"Nz": 10                 # number of vertical layers
"vert_spacing": 4.0     # 1.0 for equal vertical spacing, 4.0 otherwise
```

- learning rate and frequency of retraining:

```json 
"retrain_iceflow_emulator_lr": 0.00002 
"retrain_iceflow_emulator_freq": 5     
```

While this module was targeted for deep learning emulation, it important parameters for solving are :

is possible to
use the solver (`type_iceflow='solved'`) instead of the default emulator (`type_iceflow='emulated'`), or use the two together (`type_iceflow='diagnostic'`) to assess the emaultor against the solver. Most important parameters for solving are :

```json 
"solve_iceflow_step_size": 0.00002 
"solve_iceflow_nbitmax": 5     
```

One may choose between 2D arrhenius factor by changing parameters between `dim_arrhenius=2` or `dim_arrhenius=3` -- le later is necessary for the enthalpy model.

When treating ery large arrays, retraining must be done sequentially patch-wise for memory reason. The size of the pathc is controlled by parameter `multiple_window_size=750`.

# Reference

```
@article{jouvet2023ice,
  title={Ice flow model emulator based on physics-informed deep learning},
  author={Jouvet, Guillaume and Cordonnier, Guillaume},
  year={2023},
  publisher={EarthArXiv}
}
```

