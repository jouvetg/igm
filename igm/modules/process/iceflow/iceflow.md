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

