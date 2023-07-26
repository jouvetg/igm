
### <h1 align="center" id="title">IGM module optimize_v1 </h1>

# Description:

This function does the data assimilation (inverse modelling) to optimize thk, 
strflowctrl and usurf from observational data from the follwoing reference:

@article{jouvet2023inversion,
  title={Inversion of a Stokes glacier flow model emulated by deep learning},
  author={Jouvet, Guillaume},
  journal={Journal of Glaciology},
  volume={69},
  number={273},
  pages={13--26},
  year={2023},
  publisher={Cambridge University Press}
}

# I/O

Input: usurfobs,uvelsurfobs,vvelsurfobs,thkobs, ...
Output: thk, strflowctrl, usurf
 
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
