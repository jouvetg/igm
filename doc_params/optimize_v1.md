
argmark
=======

# Usage:


```bash
usage: argmark [-h] [--opti_vars_to_save OPTI_VARS_TO_SAVE]
               [--opti_thr_strflowctrl OPTI_THR_STRFLOWCTRL]
               [--opti_init_zero_thk OPTI_INIT_ZERO_THK]
               [--opti_regu_param_thk OPTI_REGU_PARAM_THK]
               [--opti_regu_param_strflowctrl OPTI_REGU_PARAM_STRFLOWCTRL]
               [--opti_smooth_anisotropy_factor OPTI_SMOOTH_ANISOTROPY_FACTOR]
               [--opti_convexity_weight OPTI_CONVEXITY_WEIGHT]
               [--opti_usurfobs_std OPTI_USURFOBS_STD]
               [--opti_strflowctrl_std OPTI_STRFLOWCTRL_STD]
               [--opti_velsurfobs_std OPTI_VELSURFOBS_STD] [--opti_thkobs_std OPTI_THKOBS_STD]
               [--opti_divfluxobs_std OPTI_DIVFLUXOBS_STD] [--opti_control OPTI_CONTROL]
               [--opti_cost OPTI_COST] [--opti_nbitmin OPTI_NBITMIN] [--opti_nbitmax OPTI_NBITMAX]
               [--opti_step_size OPTI_STEP_SIZE] [--opti_output_freq OPTI_OUTPUT_FREQ]
               [--geology_optimized_file GEOLOGY_OPTIMIZED_FILE]
               [--plot2d_live_inversion PLOT2D_LIVE_INVERSION]
               [--plot2d_inversion PLOT2D_INVERSION] [--write_ncdf_optimize WRITE_NCDF_OPTIMIZE]
               [--editor_plot2d_optimize EDITOR_PLOT2D_OPTIMIZE]

```
# Arguments

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
