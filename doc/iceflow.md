### <h1 align="center" id="title">IGM module `iceflow` </h1>
 
# Description:

This IGM module models ice flow dynamics in 3D using a Convolutional Neural Network based on Physics Informed Neural Network as described in this [paper](https://eartharxiv.org/repository/view/5335/). In more details, we train a CNN to minimise the energy associated with high-order ice flow equations within the time iterations of a glacier evolution model. As a result, our iflo_emulator is a computationally-efficient alternative to traditional solvers, it is capable to handle a variety of ice flow regimes and memorize previous solutions.

This module permits to load, pretrain, retrain, and evaluate ice flow emulator. In addition, it can also be used for data assimilata / model inversion (former module `optimize`) setting this option:
```json 
"iflo_run_data_assimilation": true,
```
Check the documentation for using the `optimize` option below.


It can also be used for pretraining an emulator (former module `pretraining`) setting option: 
```json 
"iflo_run_pretraining": true,
```
Check the documentation for using the `pretraining` option below.

## Iceflow



Pre-trained emulators are provided by defaults (parameter `iflo_emulator`). However, a from scratch iflo_emulator can be requested with `iflo_emulator=""`. The most important parameters are:

- physical parameters 

```json 
"iflo_init_slidingco": 0.045    # Init slid. coeff. ($Mpa y^{1/3} m^{-1/3}$)
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

For mor info, check at the following reference:

```
@article{jouvet2023ice,
  title={Ice flow model emulator based on physics-informed deep learning},
  author={Jouvet, Guillaume and Cordonnier, Guillaume},
  year={2023},
  publisher={EarthArXiv}
}
```
  
# Optimize
 
A data assimilation module of IGM permits to seek optimal ice thickness, top ice surface, and ice flow parametrization, that best explains observational data such as surface ice speeds, ice thickness profiles, top ice surface while being consistent with the ice flow iflo_emulator used in forwrd modelling. This page explains how to use the data assimilation module as a preliminary step in IGM of a forward/prognostic model run with module `iceflow` with the option:
```json 
"iflo_run_data_assimilation": true,
```

**Note that the optimization currently requires some experience, and some parameter tunning may be needed before getting a meanigfully results. Use it with care, and with a certain dose of parameter exploration. Do not hesitate to get in contact with us for chcecking the consistency of results.**

### Getting the data 

The first thing you need to do is to get as much data as possible. Data list includes:

* Observed surface ice velocities ${\bf u}^{s,obs}$, e.g. from Millan and al. (2022).
* Surface top elevation $s^{obs}$, e.g. SRTM, ESA GLO-30, ...
* Ice thickness profiles $h_p^{obs}$, e.g. GlaThiDa
* Glacier outlines, and resulting mask, e.g. from the Randolph Glacier Inventory.

Of course, you may not have all these data, which is fine. You work with a reduced amount of data, however, you will have make assumptions to reduce the number of variables to optimize (controls) to keep the optimization problem well-posed (i.e., with a unique solution).

Thes data can be obtained using the IGM module `oggm_shop`, or loading these 2D gridded variables using module `load_ncdf` or `load_tif` using convention variable names but ending with `obs`. E.g. `usurfobs` (observed top surface elevation), `thkobs` (observed thickness profiles, use nan or novalue where no data is available), `icemaskobs` (this mask from RGI outline serve to enforce zero ice thickness outside the mask), `uvelsurfobs` and `vvelsurfobs` (x- and y- components of the horizontal surface ice velocity, use nan or novalue where no data is available), `thkinit` (this may be a formerly-inferred ice thickness field to initalize the inverse model, otherwise it would start from thk=0).

**Use the IGM `oggm_shop` to download all the data you need using OGGM and the GlaThiDa database.**
 
### General optimization setting

The optimization problem consists of finding spatially varying fields ($h$, $c$, $s$) that minimize the cost function
$$\mathcal{J}(h,c,s)=\mathcal{C}^u+\mathcal{C}^h+\mathcal{C}^s+\mathcal{C}^{d}+\mathcal{R}^h+\mathcal{R}^{c}+\mathcal{P}^h,$$

where $\mathcal{C}^u$ is the misfit between modeled and observed surface ice velocities ($\mathcal{F}$ is the output of the ice flow iflo_emulator/neural iflo_network):
$$\mathcal{C}^u=\int_{\Omega}\frac{1}{2\sigma_u^2}\left|{\bf u}^{s,obs}-\mathcal{F}(h,\frac{\partial s}{\partial x},\frac{\partial s}{\partial y},c)\right|^2,$$

where $\mathcal{C}^h$ is the misfit between modeled and observed ice thickness profiles:
$$\mathcal{C}^h=\sum_{p=1,...,P} \sum_{i=1,...,M_p}\frac{1}{2 \sigma_h^2}|h_p^{obs}(x^p_i, y^p_i)-h (x^p_i, y^p_i)|^2,$$

where $\mathcal{C}^s$ is the misfit between the modeled and observed top ice surface:
$$\mathcal{C}^s=\int_{\Omega}\frac{1}{2 \sigma_s^2}\left|s-s^{obs}\right|^2,$$

where $\mathcal{C}^{d}$ is a misfit term between the flux divergence and its polynomial 
regression $d$ with respect to the ice surface elevation $s(x,y)$ to enforce smoothness with  dependence to $s$:
$$\mathcal{C}^{d}=\int_{\Omega}\frac{1}{2 \sigma_d^2}\left|\nabla \cdot (h {\bar{\bf u}})-d \right|^2,$$

where $\mathcal{R}^h$ is a regularization term to enforce anisotropic smoothness and convexity of $h$:
$$\mathcal{R}^h=\alpha_h\int_{h>0}\left(|\nabla h \cdot \tilde{{\bf u}}^{s,obs} |^2+\beta|\nabla h \cdot (\tilde{{\bf u}}^{s,obs})^{\perp} |^2-\gamma h\right),$$

where $\mathcal{R}^{c}$ is a regularization term to enforce smooth c:
$$\mathcal{R}^{c}=\alpha_{\tilde{A}}\int_{\Omega}|\nabla c|^2,$$

where $\mathcal{P}^h$ is a penalty term to enforce nonnegative ice thickness, and zero thickness outside a given mask:
$$\mathcal{P}^h=10^{10} \times \left(\int_{h<0} h^2+\int_{\mathcal{M}^{\rm ice-free}} h^2 \right).$$

Check at the reference paper given below for more explanation on the regularization terms.

### Define controls and cost components

The above optimization problem is given in the most general case, however, you may select only some components according to your data as follows: 

* the list of control variables you wish to optimize, e.g., 
```json
"opti_control": ['thk','slidingco','usurf'] # this is the most general case  
"opti_control": ['thk','usurf'] # this will only optimize ice thk and top surf 
"opti_control": ['thk'] # this will only optimize ice thickness 
```
* the list of cost components you wish to minimize, e.g.
```json
"opti_cost": ['velsurf','thk','usurf','divfluxfcz','icemask']  # most general case  
"opti_cost": ['velsurf','icemask'] # Here only fit surface velocity and ice mask.
```

**I recomend to start with a simple optimization, starting with one single control (typically `thk`), and a few target/cost component (typically `velsurf` and `icemask`), and then to increase the complexity of the optimization (adding controls and cost components) once the the most simple give meaningfull results. Make sure to keep a balance between controls and constraints to ensure the problem to keep the problem well-posed, and prevents against multiple solutions.**

### Exploring parameters

There are parameters that may need to tune for each application.

First, you may change your expected confidence levels (i.e. tolerance to fit the data) $\sigma^u, \sigma^h, \sigma^s, \sigma^d$ to fit surface ice velocity, ice thickness, surface top elevation, or divergence of the flux as follows:

```json
"opti_velsurfobs_std": 5 # unit m/y
"opti_thkobs_std" : 5 # unit m
"opti_usurfobs_std" : 5 # unit m
"opti_divfluxobs_std": 1 # unit m/y
```

Second, you may change regularization parameters such as i) $\alpha^h, \alpha^A$, which control the regularization weights for the ice thickness and strflowctrl (increasing $\alpha^h, \alpha^A$ will make thse fields spatially smoother), or ii) parameters beta and gamma involved for regularizing the ice thickness h. Taking beta=1 occurs to enforce isotropic smoothing, reducing beta will make the smoothing more and more anisotropic to enforce further smoothing along ice flow directions than accross directions (as expected for the topography of a glacier bedrock, which was eroded over long times). Setting parameter gamma to a small value may be usefull to add a bit of convexity in the system. This may help when initializing the inverse modelled with zero thickness, or to treat margin regions with no data available. These parameters may be changed as follows:

```json 
"opti_regu_param_thk": 10.0            # weight for the regul. of thk
"opti_regu_param_slidingco": 1.0     # weight for the regul. of slidingco
"opti_smooth_anisotropy_factor": 0.2
"opti_convexity_weight":  0.002
```

Lastly, there are a couple of other parameters we may be interest to change e.g.

```json
"opti_nbitmax": 1000   # Number of it. for the optimization
"opti_step_size": 0.001  # step size in the optimization iterative algorithm
"opti_init_zero_thk": True   # Force init zero ice thk (otherwise take thkinit)
```

There is also a further option: the convexity weight and the slidingco can be inferred automatically by the model. These values are calibrated only for IGM v2.2.1 and a particular set of costs and controls, and are based on a series of regressions calculated through manual inversions to find the best parameters for 50 glaciers of different types and sizes around the world (see Samuel's forthcoming paper when it's published). In other words, they are purely empirical and are likely to be a bit off for any different set of costs and controls, but should work tolerably well on any glacier anywhere on the planet, at least to give you somewhere to start exploring the parameter space. If this behaviour is desired, you MUST use RGI7.0 (C or G) and the oggm_shop module. If using C, you will also need to set the oggm_sub_entity_mask parameter to True. Within the optimize module, opti_infer_params must also be set to true.
For small glaciers with no velocity observations, the model will also use volume-area scaling to provide an additional constraint with in the inference framework - this all happens automatically, but note the opti_vol_std parameter that you can fiddle around with if you want to force it to pay more or less attention to volume (by default, this is 1000.0 - which will give a very small cost - anywhere with velocity data, and 0.001 - which will give a big cost - anywhere lacking velocity data. The parameter only controls the default value where this is other data - the 0.001 where there's no velocity data is hard-coded).
A final parameter - opti_tidewater_glacier - can also be set to True to force the inference code to treat the glacier as a tidewater-type glacier. If the RGI identifies a glacier as tidewater, it will be treated as such anyway, but this parameter gives you the option to force it (note: setting the parameter to False - its default value - will not cause the model to treat RGI-identified tidewater glaciers as non-tidewater - there is no option to do that).

### Monitoring the optimization

You may monitor the data assimilation during the inverse modelling in several ways:

* Check that the components of the costs decrease over time, the value of cost are printed during the optimization, and a graph is produced at the end.
* Set up parameter `plot_result` to  True and `plt2d_live` to True to monitor in live time the evolution of the field your are optimizing such as the ice thickness, the surface ice speeds, ect ... You may also check (hopefully decreasing) STD given in the figure.
* You may do the same monitoring after the run looking at optimize.nc reuesting this file to be written.
* If you asked divfluxfcz to be in the parameter list `opti_cost`, you should check what look like the divergence of the flux (divflux).

For mor info, check at the following reference:

```
@article{jouvet2023inversion,
  author =        {Jouvet, Guillaume},
  journal =       {Journal of Glaciology},
  number =        {273},
  pages =         {13--26},
  publisher =     {Cambridge University Press},
  title =         {{Inversion of a Stokes glacier flow model emulated by deep learning}},
  volume =        {69},
  year =          {2023},
  doi =           {10.1017/jog.2022.41},
}

@article{jouvet2023ice,
  title={Ice flow model emulator based on physics-informed deep learning},
  author={Jouvet, Guillaume and Cordonnier, Guillaume},
  year={2023},
  publisher={EarthArXiv}
}
```



 
# Pretraining

This module performs a pretraining of the ice flow iflo_emulator on a glacier catalogue to improve the performance of the emaulator when used in glacier forward run. The pretraining can be relatively computationally demanding task (a couple of hours). This module should be called alone independently of any other igm module. Here is an example of paramter file:

```json
{
  "modules_preproc": [],
  "modules_process": ["iceflow"],
  "modules_postproc": [],
  "iflo_run_pretraining": true,
  "data_dir": "surflib3d_shape_100",
  "iflo_solve_nbitmax": 2000,
  "iflo_solve_stop_if_no_decrease": false,
  "iflo_retrain_emulator_lr": 0.0001,
  "iflo_dim_arrhenius": 3,
  "soft_begining": 500
}
```

 To run it, one first needs to have available a glacier catalogue.
 I provide here [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8332898.svg)](https://doi.org/10.5281/zenodo.8332898) a dataset of a glacier catalogue (mountain glaciers) I have mostly used for pretraining IGM emaulators.

Once downloaded (or self generated), the folder 
"surflib3d_shape_100" can be re-organized into a subfolder "train" and a subfolder "test"  as follows:

```
├── test
│   └── NZ000_A78_C0
└── train
    ├── ALP02_A78_C0
    ├── ALP03_A78_C0
    ├── ALP04_A78_C0
    ├── ALP05_A78_C0
    ├── ALP06_A78_C0
    ├── ALP11_A78_C0
    ├── ALP17_A78_C0
```

The path (or name of the data folder) must be pass in parameter `data_dir`.

 
# Parameters: 


|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--working_dir`|``|Working directory (default empty string)|
||`--modules_preproc`|`['oggm_shop']`|List of pre-processing modules|
||`--modules_process`|`['iceflow', 'time', 'thk']`|List of processing modules|
||`--modules_postproc`|`['write_ncdf', 'plot2d', 'print_info']`|List of post-processing modules|
||`--logging`||Activate the looging|
||`--logging_file`|``|Logging file name, if empty it prints in the screen|
||`--print_params`||Print definitive parameters in a file for record|
||`--iflo_type`|`emulated`|Type of iceflow: it can emulated (default), solved, or in diagnostic mode to investigate the fidelity of the emulator towads the solver|
||`--iflo_pretrained_emulator`||Do we take a pretrained emulator or start from scratch?|
||`--iflo_emulator`|``|Directory path of the deep-learning pretrained ice flow model, take from the library if empty string|
||`--iflo_init_slidingco`|`0.0464`|Initial sliding coefficient slidingco|
||`--iflo_init_arrhenius`|`78`|Initial arrhenius factor arrhenuis|
||`--iflo_regu_glen`|`1e-05`|Regularization parameter for Glen's flow law|
||`--iflo_regu_weertman`|`1e-10`|Regularization parameter for Weertman's sliding law|
||`--iflo_exp_glen`|`3`|Glen's flow law exponent|
||`--iflo_exp_weertman`|`3`|Weertman's law exponent|
||`--iflo_gravity_cst`|`9.81`|Gravitational constant|
||`--iflo_ice_density`|`910`|Density of ice|
||`--iflo_new_friction_param`||Sliding coeeficient (this describe slidingco differently with slidingco**-(1.0 / exp_weertman) instead of slidingco as before)|
||`--iflo_save_model`||save the iceflow emaultor at the end of the simulation|
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






|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--working_dir`|``|Working directory (default empty string)|
||`--modules_preproc`|`['oggm_shop']`|List of pre-processing modules|
||`--modules_process`|`['iceflow', 'time', 'thk']`|List of processing modules|
||`--modules_postproc`|`['write_ncdf', 'plot2d', 'print_info']`|List of post-processing modules|
||`--logging`||Activate the looging|
||`--logging_file`|``|Logging file name, if empty it prints in the screen|
||`--print_params`||Print definitive parameters in a file for record|
||`--opti_vars_to_save`|`['usurf', 'thk', 'slidingco', 'velsurf_mag', 'velsurfobs_mag', 'divflux', 'icemask']`|List of variables to be recorded in the ncdef file|
||`--opti_init_zero_thk`|`False`|Initialize the optimization with zero ice thickness|
||`--opti_regu_param_thk`|`10.0`|Regularization weight for the ice thickness in the optimization|
||`--opti_regu_param_slidingco`|`1`|Regularization weight for the strflowctrl field in the optimization|
||`--opti_smooth_anisotropy_factor`|`0.2`|Smooth anisotropy factor for the ice thickness regularization in the optimization|
||`--opti_convexity_weight`|`0.002`|Convexity weight for the ice thickness regularization in the optimization|
||`--opti_usurfobs_std`|`2.0`|Confidence/STD of the top ice surface as input data for the optimization|
||`--opti_velsurfobs_std`|`1.0`|Confidence/STD of the surface ice velocities as input data for the optimization (if 0, velsurfobs_std field must be given)|
||`--opti_thkobs_std`|`3.0`|Confidence/STD of the ice thickness profiles (unless given)|
||`--opti_divfluxobs_std`|`1.0`|Confidence/STD of the flux divergence as input data for the optimization (if 0, divfluxobs_std field must be given)|
||`--opti_control`|`['thk']`|List of optimized variables for the optimization|
||`--opti_cost`|`['velsurf', 'thk', 'icemask']`|List of cost components for the optimization|
||`--opti_nbitmin`|`50`|Min iterations for the optimization|
||`--opti_nbitmax`|`500`|Max iterations for the optimization|
||`--opti_step_size`|`1`|Step size for the optimization|
||`--opti_output_freq`|`50`|Frequency of the output for the optimization|
||`--opti_save_result_in_ncdf`|`geology-optimized.nc`|Geology input file|
||`--opti_plot2d_live`||plot2d_live_inversion|
||`--opti_plot2d`||plot 2d inversion|
||`--opti_save_iterat_in_ncdf`||write_ncdf_optimize|
||`--opti_editor_plot2d`|`vs`|optimized for VS code (vs) or spyder (sp) for live plot|





|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--working_dir`|``|Working directory (default empty string)|
||`--modules_preproc`|`['oggm_shop']`|List of pre-processing modules|
||`--modules_process`|`['iceflow', 'time', 'thk']`|List of processing modules|
||`--modules_postproc`|`['write_ncdf', 'plot2d', 'print_info']`|List of post-processing modules|
||`--logging`||Activate the looging|
||`--logging_file`|``|Logging file name, if empty it prints in the screen|
||`--print_params`||Print definitive parameters in a file for record|
||`--data_dir`|`surflib3d_shape_100`|Directory of the data of the glacier catalogu|
||`--batch_size`|`1`|Batch size|
||`--freq_test`|`20`|Frequence of the test|
||`--train_iceflow_emulator_restart_lr`|`2500`|Restart frequency for the learning rate|
||`--epochs`|`5000`|Number of epochs|
||`--min_arrhenius`|`5`|Minium Arrhenius factor|
||`--max_arrhenius`|`151`|Maximum Arrhenius factor|
||`--min_slidingco`|`0`|Minimum sliding coefficient|
||`--max_slidingco`|`20000`|Maximum sliding coefficient|
||`--min_coarsen`|`0`|Minimum coarsening factor|
||`--max_coarsen`|`2`|Maximum coarsening factor|
||`--soft_begining`|`500`|soft_begining, if 0 explore all parameters btwe min and max, otherwise,               only explore from this iteration while keeping mid-value fir the first it.|
