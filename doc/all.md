
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
               [--crop_tif_ymax CROP_TIF_YMAX] [--RGI_ID RGI_ID] [--preprocess PREPROCESS]
               [--dx DX] [--border BORDER] [--thk_source THK_SOURCE]
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
               [--type_iceflow TYPE_ICEFLOW] [--emulator EMULATOR]
               [--iceflow_physics ICEFLOW_PHYSICS] [--init_slidingco INIT_SLIDINGCO]
               [--init_arrhenius INIT_ARRHENIUS] [--regu_glen REGU_GLEN]
               [--regu_weertman REGU_WEERTMAN] [--exp_glen EXP_GLEN] [--exp_weertman EXP_WEERTMAN]
               [--gravity_cst GRAVITY_CST] [--ice_density ICE_DENSITY] [--Nz NZ]
               [--vert_spacing VERT_SPACING] [--thr_ice_thk THR_ICE_THK]
               [--solve_iceflow_step_size SOLVE_ICEFLOW_STEP_SIZE]
               [--solve_iceflow_nbitmax SOLVE_ICEFLOW_NBITMAX]
               [--stop_if_no_decrease STOP_IF_NO_DECREASE] [--fieldin FIELDIN]
               [--dim_arrhenius DIM_ARRHENIUS]
               [--retrain_iceflow_emulator_freq RETRAIN_ICEFLOW_EMULATOR_FREQ]
               [--retrain_iceflow_emulator_lr RETRAIN_ICEFLOW_EMULATOR_LR]
               [--retrain_iceflow_emulator_nbit RETRAIN_ICEFLOW_EMULATOR_NBIT]
               [--retrain_iceflow_emulator_framesizemax RETRAIN_ICEFLOW_EMULATOR_FRAMESIZEMAX]
               [--multiple_window_size MULTIPLE_WINDOW_SIZE] [--force_max_velbar FORCE_MAX_VELBAR]
               [--network NETWORK] [--activation ACTIVATION] [--nb_layers NB_LAYERS]
               [--nb_blocks NB_BLOCKS] [--nb_out_filter NB_OUT_FILTER]
               [--conv_ker_size CONV_KER_SIZE] [--dropout_rate DROPOUT_RATE]
               [--time_start TIME_START] [--time_end TIME_END] [--time_save TIME_SAVE] [--cfl CFL]
               [--time_step_max TIME_STEP_MAX] [--erosion_cst EROSION_CST]
               [--erosion_exp EROSION_EXP] [--erosion_update_freq EROSION_UPDATE_FREQ]
               [--tracking_method TRACKING_METHOD] [--frequency_seeding FREQUENCY_SEEDING]
               [--density_seeding DENSITY_SEEDING] [--speed_rockflow SPEED_ROCKFLOW]
               [--vars_to_save VARS_TO_SAVE] [--output_file_ncdf_ex OUTPUT_FILE_NCDF_EX]
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
||`--modules_preproc`|`['prepare_data']`|List of pre-processing modules|
||`--modules_process`|`['flow_dt_thk']`|List of processing modules|
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
||`--RGI_ID`|`RGI60-11.01450`|RGI ID|
||`--preprocess`||Use preprocessing|
||`--dx`|`100`|Spatial resolution (need preprocess false to change it)|
||`--border`|`30`|Safe border margin  (need preprocess false to change it)|
||`--thk_source`|`consensus_ice_thickness`|millan_ice_thickness or consensus_ice_thickness in geology.nc|
||`--include_glathida`||Make observation file (for IGM inverse)|
||`--path_glathida`|`/home/gjouvet/`|Path where the Glathida Folder is store, so that you don't need               to redownload it at any use of the script|
||`--save_input_ncdf`||Write prepared data into a geology file|
||`--input_file`|`input.nc`|Input data file|
||`--coarsen_ncdf`|`1`|coarsen the data to a coarser resolution (default: 1), e.g. 2 would be twice coarser ignore data each 2 grid points|
||`--crop_ncdf`|`False`|Crop the data with xmin, xmax, ymin, ymax (default: False)|
||`--crop_ncdf_xmin`|`None`|X left coordinate for cropping|
||`--crop_ncdf_xmax`|`None`|X right coordinate for cropping|
||`--crop_ncdf_ymin`|`None`|Y bottom coordinate fro cropping|
||`--crop_ncdf_ymax`|`None`|Y top coordinate for cropping|
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
||`--smb_simple_file`|`mb_simple_param.txt`|Name of the imput file for the simple mass balance model|
||`--type_iceflow`|`emulated`|emulated, solved, diagnostic|
||`--emulator`|`f21_pinnbp_GJ_23_a`|Directory path of the deep-learning ice flow model,               create a new if empty string|
||`--iceflow_physics`|`2`|2 for blatter, 4 for stokes, this is also the number of DOF|
||`--init_slidingco`|`10000`|Initial sliding coeeficient slidingco (default: 0)|
||`--init_arrhenius`|`78`|Initial arrhenius factor arrhenuis (default: 78)|
||`--regu_glen`|`1e-05`|Regularization parameter for Glen's flow law|
||`--regu_weertman`|`1e-10`|Regularization parameter for Weertman's sliding law|
||`--exp_glen`|`3`|Glen's flow law exponent|
||`--exp_weertman`|`3`|Weertman's law exponent|
||`--gravity_cst`|`9.81`|Gravitational constant|
||`--ice_density`|`910`|Density of ice|
||`--Nz`|`10`|Nz for the vertical discretization|
||`--vert_spacing`|`4.0`|1.0 for equal vertical spacing, 4.0 otherwise (4.0)|
||`--thr_ice_thk`|`0.1`|Threshold Ice thickness for computing strain rate|
||`--solve_iceflow_step_size`|`1`|solver_step_size|
||`--solve_iceflow_nbitmax`|`100`|solver_nbitmax|
||`--stop_if_no_decrease`||stop_if_no_decrease for the solver|
||`--fieldin`|`['thk', 'usurf', 'arrhenius', 'slidingco', 'dX']`|Input parameter of the iceflow emulator|
||`--dim_arrhenius`|`2`|dimension of the arrhenius factor (horizontal 2D or 3D)|
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
||`--time_start`|`2000.0`|Start modelling time (default 2000)|
||`--time_end`|`2100.0`|End modelling time (default: 2100)|
||`--time_save`|`10`|Save result each X years (default: 10)|
||`--cfl`|`0.3`|CFL number for the stability of the mass conservation scheme,         it must be below 1 (Default: 0.3)|
||`--time_step_max`|`10.0`|Maximum time step allowed, used only with slow ice (default: 10.0)|
||`--erosion_cst`|`2.7e-07`|Erosion multiplicative factor, here taken from Herman, F. et al.               Erosion by an Alpine glacier. Science 350, 193–195 (2015)|
||`--erosion_exp`|`2`|Erosion exponent factor, here taken from Herman, F. et al.                Erosion by an Alpine glacier. Science 350, 193–195 (2015)|
||`--erosion_update_freq`|`1`|Update the erosion only each X years (Default: 100)|
||`--tracking_method`|`simple`|Method for tracking particles (3d or simple)|
||`--frequency_seeding`|`50`|Frequency of seeding (default: 10)|
||`--density_seeding`|`0.2`|Density of seeding (default: 0.2)|
||`--speed_rockflow`|`1`|speed rock flow|
||`--vars_to_save`|`['usurf', 'thk']`|List of variables to be recorded in the ncdf file|
||`--output_file_ncdf_ex`|`output.nc`|Output ncdf data file|
||`--vars_to_save_ncdf_ex`|`['topg', 'usurf', 'thk', 'smb', 'velbar_mag', 'velsurf_mag', 'uvelsurf', 'vvelsurf', 'wvelsurf']`|List of variables to be recorded in the ncdf file|
||`--output_file_ncdf_ts`|`output_ts.nc`|Output ncdf data file (time serie)|
||`--add_topography_to_particles`||Add topg|
||`--editor_plot2d`|`vs`|optimized for VS code (vs) or spyder (sp) for live plot|
||`--plot_live`||Display plots live the results during computation (Default: False)|
||`--plot_particles`||Display particles (Default: True)|
||`--varplot`|`velbar_mag`|variable to plot|
||`--varplot_max`|`250`|maximum value of the varplot variable used to adjust the scaling of the colorbar|
