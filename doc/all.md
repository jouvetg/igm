
argmark
=======

# Description


IGM
# Usage:


```bash
usage: argmark [-h] [--working_dir WORKING_DIR] [--modules_preproc MODULES_PREPROC]
               [--modules_process MODULES_PROCESS] [--modules_postproc MODULES_POSTPROC]
               [--logging LOGGING] [--logging_file LOGGING_FILE] [--print_params PRINT_PARAMS]
               [--coarsen_tif COARSEN_TIF] [--crop_tif CROP_TIF] [--crop_tif_xmin CROP_TIF_XMIN]
               [--crop_tif_xmax CROP_TIF_XMAX] [--crop_tif_ymin CROP_TIF_YMIN]
               [--crop_tif_ymax CROP_TIF_YMAX] [--data_dir DATA_DIR] [--batch_size BATCH_SIZE]
               [--freq_test FREQ_TEST]
               [--train_iceflow_emulator_restart_lr TRAIN_ICEFLOW_EMULATOR_RESTART_LR]
               [--epochs EPOCHS] [--min_arrhenius MIN_ARRHENIUS] [--max_arrhenius MAX_ARRHENIUS]
               [--min_slidingco MIN_SLIDINGCO] [--max_slidingco MAX_SLIDINGCO]
               [--min_coarsen MIN_COARSEN] [--max_coarsen MAX_COARSEN]
               [--soft_begining SOFT_BEGINING] [--mask_shapefile MASK_SHAPEFILE]
               [--mask_invert MASK_INVERT] [--RGI_ID RGI_ID] [--preprocess PREPROCESS] [--dx DX]
               [--border BORDER] [--thk_source THK_SOURCE] [--vel_source VEL_SOURCE]
               [--include_glathida INCLUDE_GLATHIDA] [--path_glathida PATH_GLATHIDA]
               [--save_input_ncdf SAVE_INPUT_NCDF] [--input_file INPUT_FILE]
               [--coarsen_ncdf COARSEN_NCDF] [--crop_ncdf CROP_NCDF]
               [--crop_ncdf_xmin CROP_NCDF_XMIN] [--crop_ncdf_xmax CROP_NCDF_XMAX]
               [--crop_ncdf_ymin CROP_NCDF_YMIN] [--crop_ncdf_ymax CROP_NCDF_YMAX]
               [--opti_vars_to_save OPTI_VARS_TO_SAVE] [--opti_init_zero_thk OPTI_INIT_ZERO_THK]
               [--opti_regu_param_thk OPTI_REGU_PARAM_THK]
               [--opti_regu_param_slidingco OPTI_REGU_PARAM_SLIDINGCO]
               [--opti_smooth_anisotropy_factor OPTI_SMOOTH_ANISOTROPY_FACTOR]
               [--opti_convexity_weight OPTI_CONVEXITY_WEIGHT]
               [--opti_usurfobs_std OPTI_USURFOBS_STD] [--opti_velsurfobs_std OPTI_VELSURFOBS_STD]
               [--opti_thkobs_std OPTI_THKOBS_STD] [--opti_divfluxobs_std OPTI_DIVFLUXOBS_STD]
               [--opti_control OPTI_CONTROL] [--opti_cost OPTI_COST] [--opti_nbitmin OPTI_NBITMIN]
               [--opti_nbitmax OPTI_NBITMAX] [--opti_step_size OPTI_STEP_SIZE]
               [--opti_output_freq OPTI_OUTPUT_FREQ]
               [--geology_optimized_file GEOLOGY_OPTIMIZED_FILE]
               [--plot2d_live_inversion PLOT2D_LIVE_INVERSION]
               [--plot2d_inversion PLOT2D_INVERSION] [--write_ncdf_optimize WRITE_NCDF_OPTIMIZE]
               [--editor_plot2d_optimize EDITOR_PLOT2D_OPTIMIZE]
               [--smb_update_freq SMB_UPDATE_FREQ] [--smb_simple_file SMB_SIMPLE_FILE]
               [--smb_simple_array SMB_SIMPLE_ARRAY] [--type_iceflow TYPE_ICEFLOW]
               [--emulator EMULATOR] [--iceflow_physics ICEFLOW_PHYSICS]
               [--init_slidingco INIT_SLIDINGCO] [--init_arrhenius INIT_ARRHENIUS]
               [--regu_glen REGU_GLEN] [--regu_weertman REGU_WEERTMAN] [--exp_glen EXP_GLEN]
               [--exp_weertman EXP_WEERTMAN] [--gravity_cst GRAVITY_CST]
               [--ice_density ICE_DENSITY] [--new_friction_param NEW_FRICTION_PARAM] [--Nz NZ]
               [--vert_spacing VERT_SPACING] [--thr_ice_thk THR_ICE_THK]
               [--solve_iceflow_step_size SOLVE_ICEFLOW_STEP_SIZE]
               [--solve_iceflow_nbitmax SOLVE_ICEFLOW_NBITMAX]
               [--stop_if_no_decrease STOP_IF_NO_DECREASE] [--fieldin FIELDIN]
               [--dim_arrhenius DIM_ARRHENIUS]
               [--retrain_iceflow_emulator_freq RETRAIN_ICEFLOW_EMULATOR_FREQ]
               [--retrain_iceflow_emulator_lr RETRAIN_ICEFLOW_EMULATOR_LR]
               [--retrain_iceflow_emulator_nbit_init RETRAIN_ICEFLOW_EMULATOR_NBIT_INIT]
               [--retrain_iceflow_emulator_nbit RETRAIN_ICEFLOW_EMULATOR_NBIT]
               [--retrain_iceflow_emulator_framesizemax RETRAIN_ICEFLOW_EMULATOR_FRAMESIZEMAX]
               [--multiple_window_size MULTIPLE_WINDOW_SIZE] [--force_max_velbar FORCE_MAX_VELBAR]
               [--network NETWORK] [--activation ACTIVATION] [--nb_layers NB_LAYERS]
               [--nb_blocks NB_BLOCKS] [--nb_out_filter NB_OUT_FILTER]
               [--conv_ker_size CONV_KER_SIZE] [--dropout_rate DROPOUT_RATE]
               [--exclude_borders_from_iceflow EXCLUDE_BORDERS_FROM_ICEFLOW]
               [--time_start TIME_START] [--time_end TIME_END] [--time_save TIME_SAVE] [--cfl CFL]
               [--time_step_max TIME_STEP_MAX] [--erosion_cst EROSION_CST]
               [--erosion_exp EROSION_EXP] [--erosion_update_freq EROSION_UPDATE_FREQ]
               [--tracking_method TRACKING_METHOD] [--frequency_seeding FREQUENCY_SEEDING]
               [--density_seeding DENSITY_SEEDING] [--speed_rockflow SPEED_ROCKFLOW]
               [--vars_to_save_tif_ex VARS_TO_SAVE_TIF_EX]
               [--output_file_ncdf_ex OUTPUT_FILE_NCDF_EX]
               [--vars_to_save_ncdf_ex VARS_TO_SAVE_NCDF_EX]
               [--output_file_ncdf_ts OUTPUT_FILE_NCDF_TS]
               [--add_topography_to_particles ADD_TOPOGRAPHY_TO_PARTICLES]
               [--editor_plot2d EDITOR_PLOT2D] [--plot_live PLOT_LIVE]
               [--plot_particles PLOT_PARTICLES] [--varplot VARPLOT] [--varplot_max VARPLOT_MAX]

```
# Arguments

|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--working_dir`|``|Working directory (default empty string)|
||`--modules_preproc`|`['oggm_data_prep']`|List of pre-processing modules|
||`--modules_process`|`['iceflow', 'time_step', 'thk']`|List of processing modules|
||`--modules_postproc`|`['write_ncdf_ex', 'write_plot2d', 'print_info']`|List of post-processing modules|
||`--logging`||Activate the looging|
||`--logging_file`|``|Logging file name, if empty it prints in the screen|
||`--print_params`||Print definitive parameters in a file for record|
||`--coarsen_tif`|`1`|coarsen the data to a coarser resolution (default: 1), e.g. 2 would be twice coarser ignore data each 2 grid points|
||`--crop_tif`|`False`|Crop the data with xmin, xmax, ymin, ymax (default: False)|
||`--crop_tif_xmin`|`None`|crop_xmin|
||`--crop_tif_xmax`|`None`|crop_xmax|
||`--crop_tif_ymin`|`None`|crop_ymin|
||`--crop_tif_ymax`|`None`|crop_ymax|
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
||`--mask_shapefile`|`icemask.shp`|Icemask input file (default: icemask.shp)|
||`--mask_invert`||Invert ice mask if the mask is where the ice should be (default: False)|
||`--RGI_ID`|`RGI60-11.01450`|RGI ID|
||`--preprocess`||Use preprocessing|
||`--dx`|`100`|Spatial resolution (need preprocess false to change it)|
||`--border`|`30`|Safe border margin  (need preprocess false to change it)|
||`--thk_source`|`consensus_ice_thickness`|millan_ice_thickness or consensus_ice_thickness|
||`--vel_source`|`millan_ice_velocity`|Source of the surface velocities (millan_ice_velocity or its_live)|
||`--include_glathida`||Make observation file (for IGM inverse)|
||`--path_glathida`|``|Path where the Glathida Folder is store, so that you don't need               to redownload it at any use of the script, if empty it will be in the home directory|
||`--save_input_ncdf`||Write prepared data into a geology file|
||`--input_file`|`input.nc`|NetCDF input data file|
||`--coarsen_ncdf`|`1`|Coarsen the data from NetCDF file by a certain (integer) number: 2 would be twice coarser ignore data each 2 grid points|
||`--crop_ncdf`|`False`|Crop the data from NetCDF file with given top/down/left/right bounds|
||`--crop_ncdf_xmin`|`None`|X left coordinate for cropping the NetCDF data|
||`--crop_ncdf_xmax`|`None`|X right coordinate for cropping the NetCDF data|
||`--crop_ncdf_ymin`|`None`|Y bottom coordinate fro cropping the NetCDF data|
||`--crop_ncdf_ymax`|`None`|Y top coordinate for cropping the NetCDF data|
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
||`--smb_update_freq`|`1`|Update the mass balance each X years (1)|
||`--smb_simple_file`|`smb_simple_param.txt`|Name of the imput file for the simple mass balance model (time, gradabl, gradacc, ela, accmax)|
||`--smb_simple_array`|`[]`|Time dependent parameters for simple mass balance model (time, gradabl, gradacc, ela, accmax)|
||`--type_iceflow`|`emulated`|Type of iceflow: it can emulated (default), solved, or in diagnostic mode to investigate the fidelity of the emulator towads the solver|
||`--emulator`|`myemulator`|Directory path of the deep-learning ice flow model, create a new if empty string|
||`--iceflow_physics`|`2`|2 for blatter, 4 for stokes, this is also the number of DOF (STOKES DOES NOT WORK YET, KEEP IT TO 2)|
||`--init_slidingco`|`10000`|Initial sliding coefficient slidingco|
||`--init_arrhenius`|`78`|Initial arrhenius factor arrhenuis|
||`--regu_glen`|`1e-05`|Regularization parameter for Glen's flow law|
||`--regu_weertman`|`1e-10`|Regularization parameter for Weertman's sliding law|
||`--exp_glen`|`3`|Glen's flow law exponent|
||`--exp_weertman`|`3`|Weertman's law exponent|
||`--gravity_cst`|`9.81`|Gravitational constant|
||`--ice_density`|`910`|Density of ice|
||`--new_friction_param`||ExperimentaL: this describe slidingco differently with slidingco**-(1.0 / exp_weertman) instead of slidingco|
||`--Nz`|`10`|Number of grid point for the vertical discretization|
||`--vert_spacing`|`4.0`|Parameter controlling the discrtuzation density to get more point near the bed than near the the surface. 1.0 means equal vertical spacing.|
||`--thr_ice_thk`|`0.1`|Threshold Ice thickness for computing strain rate|
||`--solve_iceflow_step_size`|`1`|Step size for the optimizer using when solving Blatter-Pattyn in solver mode|
||`--solve_iceflow_nbitmax`|`100`|Maximum number of iteration for the optimizer using when solving Blatter-Pattyn in solver mode|
||`--stop_if_no_decrease`||This permits to stop the solver if the energy does not decrease|
||`--fieldin`|`['thk', 'usurf', 'arrhenius', 'slidingco', 'dX']`|Input fields of the iceflow emulator|
||`--dim_arrhenius`|`2`|Dimension of the arrhenius factor (horizontal 2D or 3D)|
||`--retrain_iceflow_emulator_freq`|`10`|Frequency at which the emulator is retrained, 0 means never, 1 means at each time step, 2 means every two time steps, etc.|
||`--retrain_iceflow_emulator_lr`|`2e-05`|Learning rate for the retraining of the emulator|
||`--retrain_iceflow_emulator_nbit_init`|`1`|Number of iterations done at the first time step for the retraining of the emulator|
||`--retrain_iceflow_emulator_nbit`|`1`|Number of iterations done at each time step for the retraining of the emulator|
||`--retrain_iceflow_emulator_framesizemax`|`750`|Size of the patch used for retraining the emulator, this is usefull for large size arrays, otherwise the GPU memory can be overloaded|
||`--multiple_window_size`|`0`|If a U-net, this force window size a multiple of 2**N|
||`--force_max_velbar`|`0`|This permits to artifically upper-bound velocities, active if > 0|
||`--network`|`cnn`|This is the type of network, it can be cnn or unet|
||`--activation`|`lrelu`|Activation function, it can be lrelu, relu, tanh, sigmoid, etc.|
||`--nb_layers`|`16`|Number of layers in the CNN|
||`--nb_blocks`|`4`|Number of block layer in the U-net|
||`--nb_out_filter`|`32`|Number of output filters in the CNN|
||`--conv_ker_size`|`3`|Size of the convolution kernel|
||`--dropout_rate`|`0`|Dropout rate in the CNN|
||`--exclude_borders_from_iceflow`||This is a quick fix of the border issue, other the physics informed emaulator shows zero velocity at the border|
||`--time_start`|`2000.0`|Start modelling time|
||`--time_end`|`2100.0`|End modelling time|
||`--time_save`|`10`|Save result frequency for many modules (in year)|
||`--cfl`|`0.3`|CFL number for the stability of the mass conservation scheme, it must be below 1|
||`--time_step_max`|`1.0`|Maximum time step allowed, used only with slow ice|
||`--erosion_cst`|`2.7e-07`|Erosion multiplicative factor, here taken from Herman, F. et al.               Erosion by an Alpine glacier. Science 350, 193–195 (2015)|
||`--erosion_exp`|`2`|Erosion exponent factor, here taken from Herman, F. et al.                Erosion by an Alpine glacier. Science 350, 193–195 (2015)|
||`--erosion_update_freq`|`1`|Update the erosion only each X years (Default: 100)|
||`--tracking_method`|`simple`|Method for tracking particles (simple or 3d)|
||`--frequency_seeding`|`50`|Frequency of seeding (unit : year)|
||`--density_seeding`|`0.2`|Density of seeding (1 means we seed all pixels, 0.2 means we seed each 5 grid cell, ect.)|
||`--speed_rockflow`|`1`|Speed of rock flow along the slope in m/y|
||`--vars_to_save_tif_ex`|`['usurf', 'thk']`|List of variables to be recorded in the NetCDF file|
||`--output_file_ncdf_ex`|`output.nc`|Output ncdf data file|
||`--vars_to_save_ncdf_ex`|`['topg', 'usurf', 'thk', 'smb', 'velbar_mag', 'velsurf_mag', 'uvelsurf', 'vvelsurf', 'wvelsurf']`|List of variables to be recorded in the ncdf file|
||`--output_file_ncdf_ts`|`output_ts.nc`|Output ncdf data file (time serie)|
||`--add_topography_to_particles`||Add topg|
||`--editor_plot2d`|`vs`|Optimized for VS code (vs) or spyder (sp) for live plot|
||`--plot_live`||Display plots live the results during computation instead of making png|
||`--plot_particles`||Display particles is True, does not display if False|
||`--varplot`|`velbar_mag`|Name of the variable to plot|
||`--varplot_max`|`250`|Maximum value of the varplot variable used to adjust the scaling of the colorbar|
