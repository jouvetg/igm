
### <h1 align="center" id="title">IGM module `pretaining` </h1>

# Description:

This module performs a pretraining of the ice flow emulator on a glacier catalogue to improve the performance of the emaulator when used in glacier forward run. The pretraining can be relatively computationally demanding task (a couple of hours). This module should be called alone independently of any other igm module. Here is an example of paramter file:

```json
{
  "modules_preproc": ["pretraining"],
  "modules_process": [],
  "modules_postproc": [],
  "data_dir": "surflib3d_shape_100",
  "solve_iceflow_nbitmax": 2000,
  "stop_if_no_decrease": false,
  "retrain_iceflow_emulator_lr": 0.0001,
  "dim_arrhenius": 3,
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
||`--opti_vars_to_save`|`['usurf', 'thk', 'strflowctrl', 'arrhenius', 'slidingco', 'velsurf_mag', 'velsurfobs_mag', 'divflux']`|List of variables to be recorded in the ncdef file|
||`--opti_thr_strflowctrl`|`78.0`|threshold value for strflowctrl|
||`--opti_init_zero_thk`|`False`|Initialize the optimization with zero ice thickness|
||`--opti_regu_param_thk`|`10.0`|Regularization weight for the ice thickness in the optimization|
||`--opti_regu_param_strflowctrl`|`1.0`|Regularization weight for the strflowctrl field in the optimization|
||`--opti_smooth_anisotropy_factor`|`0.2`|Smooth anisotropy factor for the ice thickness regularization in the optimization|
||`--opti_convexity_weight`|`0.002`|Convexity weight for the ice thickness regularization in the optimization|
||`--opti_usurfobs_std`|`5.0`|Confidence/STD of the top ice surface as input data for the optimization|
||`--opti_strflowctrl_std`|`5.0`|Confidence/STD of strflowctrl|
||`--opti_velsurfobs_std`|`3.0`|Confidence/STD of the surface ice velocities as input data for the optimization (if 0, velsurfobs_std field must be given)|
||`--opti_thkobs_std`|`5.0`|Confidence/STD of the ice thickness profiles (unless given)|
||`--opti_divfluxobs_std`|`1.0`|Confidence/STD of the flux divergence as input data for the optimization (if 0, divfluxobs_std field must be given)|
||`--opti_control`|`['thk', 'strflowctrl', 'usurf']`|List of optimized variables for the optimization|
||`--opti_cost`|`['velsurf', 'thk', 'usurf', 'divfluxfcz', 'icemask']`|List of cost components for the optimization|
||`--opti_nbitmin`|`50`|Min iterations for the optimization|
||`--opti_nbitmax`|`1000`|Max iterations for the optimization|
||`--opti_step_size`|`0.001`|Step size for the optimization|
||`--opti_output_freq`|`50`|Frequency of the output for the optimization|
||`--geology_optimized_file`|`geology-optimized.nc`|Geology input file|
||`--plot2d_live_inversion`||plot2d_live_inversion|
||`--plot2d_inversion`||plot 2d inversion|
||`--write_ncdf_optimize`||write_ncdf_optimize|
||`--editor_plot2d_optimize`|`vs`|optimized for VS code (vs) or spyder (sp) for live plot|
