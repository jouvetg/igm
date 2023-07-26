
### <h1 align="center" id="title">IGM module optimize </h1>

# Description:

This function does the data assimilation (inverse modelling) to optimize thk, 
slidingco and usurf from observational data from the follwoing reference:

@article{jouvet2023ice,
  title={Ice flow model emulator based on physics-informed deep learning},
  author={Jouvet, Guillaume and Cordonnier, Guillaume},
  year={2023},
  publisher={EarthArXiv}
}

# I/=

Input: usurfobs,uvelsurfobs,vvelsurfobs,thkobs, ...
Output: thk, slidingco, usurf
 
# Parameters: 


|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--opti_vars_to_save`|`['usurf', 'thk', 'slidingco', 'velsurf_mag', 'velsurfobs_mag', 'divflux', 'icemask']`|List of variables to be recorded in the ncdef file|
||`--opti_init_zero_thk`|`False`|Initialize the optimization with zero ice thickness|
||`--opti_regu_param_thk`|`5.0`|Regularization weight for the ice thickness in the optimization|
||`--opti_regu_param_slidingco`|`1`|Regularization weight for the strflowctrl field in the optimization|
||`--opti_smooth_anisotropy_factor`|`0.2`|Smooth anisotropy factor for the ice thickness regularization in the optimization|
||`--opti_convexity_weight`|`0.002`|Convexity weight for the ice thickness regularization in the optimization|
||`--opti_usurfobs_std`|`2.0`|Confidence/STD of the top ice surface as input data for the optimization|
||`--opti_velsurfobs_std`|`2.0`|Confidence/STD of the surface ice velocities as input data for the optimization (if 0, velsurfobs_std field must be given)|
||`--opti_thkobs_std`|`3.0`|Confidence/STD of the ice thickness profiles (unless given)|
||`--opti_divfluxobs_std`|`1.0`|Confidence/STD of the flux divergence as input data for the optimization (if 0, divfluxobs_std field must be given)|
||`--opti_control`|`['thk']`|List of optimized variables for the optimization|
||`--opti_cost`|`['velsurf', 'thk', 'icemask']`|List of cost components for the optimization|
||`--opti_nbitmin`|`50`|Min iterations for the optimization|
||`--opti_nbitmax`|`500`|Max iterations for the optimization|
||`--opti_step_size`|`1`|Step size for the optimization|
||`--opti_output_freq`|`50`|Frequency of the output for the optimization|
||`--geology_optimized_file`|`geology-optimized.nc`|Geology input file|
||`--plot2d_live_inversion`||plot2d_live_inversion|
||`--plot2d_inversion`||plot 2d inversion|
||`--write_ncdf_optimize`||write_ncdf_optimize|
||`--editor_plot2d_optimize`|`vs`|optimized for VS code (vs) or spyder (sp) for live plot|
