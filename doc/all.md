
argmark
=======

# Description


IGM
# Usage:


```bash
usage: argmark [-h] [--working_dir WORKING_DIR] [--modules_preproc MODULES_PREPROC] [--modules_process MODULES_PROCESS] [--modules_postproc MODULES_POSTPROC]
               [--logging LOGGING] [--logging_file LOGGING_FILE] [--print_params PRINT_PARAMS] [--lncd_input_file LNCD_INPUT_FILE] [--lncd_coarsen LNCD_COARSEN]
               [--lncd_crop LNCD_CROP] [--lncd_xmin LNCD_XMIN] [--lncd_xmax LNCD_XMAX] [--lncd_ymin LNCD_YMIN] [--lncd_ymax LNCD_YMAX] [--data_dir DATA_DIR]
               [--batch_size BATCH_SIZE] [--freq_test FREQ_TEST] [--train_iceflow_emulator_restart_lr TRAIN_ICEFLOW_EMULATOR_RESTART_LR] [--epochs EPOCHS]
               [--min_arrhenius MIN_ARRHENIUS] [--max_arrhenius MAX_ARRHENIUS] [--min_slidingco MIN_SLIDINGCO] [--max_slidingco MAX_SLIDINGCO] [--min_coarsen MIN_COARSEN]
               [--max_coarsen MAX_COARSEN] [--soft_begining SOFT_BEGINING] [--mask_shapefile MASK_SHAPEFILE] [--mask_invert MASK_INVERT] [--ltif_coarsen LTIF_COARSEN]
               [--ltif_crop LTIF_CROP] [--ltif_xmin LTIF_XMIN] [--ltif_xmax LTIF_XMAX] [--ltif_ymin LTIF_YMIN] [--ltif_ymax LTIF_YMAX] [--opti_vars_to_save OPTI_VARS_TO_SAVE]
               [--opti_init_zero_thk OPTI_INIT_ZERO_THK] [--opti_regu_param_thk OPTI_REGU_PARAM_THK] [--opti_regu_param_slidingco OPTI_REGU_PARAM_SLIDINGCO]
               [--opti_smooth_anisotropy_factor OPTI_SMOOTH_ANISOTROPY_FACTOR] [--opti_convexity_weight OPTI_CONVEXITY_WEIGHT] [--opti_usurfobs_std OPTI_USURFOBS_STD]
               [--opti_velsurfobs_std OPTI_VELSURFOBS_STD] [--opti_thkobs_std OPTI_THKOBS_STD] [--opti_divfluxobs_std OPTI_DIVFLUXOBS_STD] [--opti_control OPTI_CONTROL]
               [--opti_cost OPTI_COST] [--opti_nbitmin OPTI_NBITMIN] [--opti_nbitmax OPTI_NBITMAX] [--opti_step_size OPTI_STEP_SIZE] [--opti_output_freq OPTI_OUTPUT_FREQ]
               [--opti_save_result_in_ncdf OPTI_SAVE_RESULT_IN_NCDF] [--opti_plot2d_live OPTI_PLOT2D_LIVE] [--opti_plot2d OPTI_PLOT2D]
               [--opti_save_iterat_in_ncdf OPTI_SAVE_ITERAT_IN_NCDF] [--opti_editor_plot2d OPTI_EDITOR_PLOT2D] [--oggm_RGI_ID OGGM_RGI_ID] [--oggm_preprocess OGGM_PREPROCESS]
               [--oggm_dx OGGM_DX] [--oggm_border OGGM_BORDER] [--oggm_thk_source OGGM_THK_SOURCE] [--oggm_vel_source OGGM_VEL_SOURCE] [--oggm_incl_glathida OGGM_INCL_GLATHIDA]
               [--oggm_path_glathida OGGM_PATH_GLATHIDA] [--oggm_save_in_ncdf OGGM_SAVE_IN_NCDF] [--smb_simple_update_freq SMB_SIMPLE_UPDATE_FREQ]
               [--smb_simple_file SMB_SIMPLE_FILE] [--smb_simple_array SMB_SIMPLE_ARRAY] [--iflo_type IFLO_TYPE] [--iflo_emulator IFLO_EMULATOR]
               [--iflo_init_slidingco IFLO_INIT_SLIDINGCO] [--iflo_init_arrhenius IFLO_INIT_ARRHENIUS] [--iflo_regu_glen IFLO_REGU_GLEN]
               [--iflo_regu_weertman IFLO_REGU_WEERTMAN] [--iflo_exp_glen IFLO_EXP_GLEN] [--iflo_exp_weertman IFLO_EXP_WEERTMAN] [--iflo_gravity_cst IFLO_GRAVITY_CST]
               [--iflo_ice_density IFLO_ICE_DENSITY] [--iflo_new_friction_param IFLO_NEW_FRICTION_PARAM] [--iflo_Nz IFLO_NZ] [--iflo_vert_spacing IFLO_VERT_SPACING]
               [--iflo_thr_ice_thk IFLO_THR_ICE_THK] [--iflo_solve_step_size IFLO_SOLVE_STEP_SIZE] [--iflo_solve_nbitmax IFLO_SOLVE_NBITMAX]
               [--iflo_solve_stop_if_no_decrease IFLO_SOLVE_STOP_IF_NO_DECREASE] [--iflo_fieldin IFLO_FIELDIN] [--iflo_dim_arrhenius IFLO_DIM_ARRHENIUS]
               [--iflo_retrain_emulator_freq IFLO_RETRAIN_EMULATOR_FREQ] [--iflo_retrain_emulator_lr IFLO_RETRAIN_EMULATOR_LR]
               [--iflo_retrain_emulator_nbit_init IFLO_RETRAIN_EMULATOR_NBIT_INIT] [--iflo_retrain_emulator_nbit IFLO_RETRAIN_EMULATOR_NBIT]
               [--iflo_retrain_emulator_framesizemax IFLO_RETRAIN_EMULATOR_FRAMESIZEMAX] [--iflo_multiple_window_size IFLO_MULTIPLE_WINDOW_SIZE]
               [--iflo_force_max_velbar IFLO_FORCE_MAX_VELBAR] [--iflo_network IFLO_NETWORK] [--iflo_activation IFLO_ACTIVATION] [--iflo_nb_layers IFLO_NB_LAYERS]
               [--iflo_nb_blocks IFLO_NB_BLOCKS] [--iflo_nb_out_filter IFLO_NB_OUT_FILTER] [--iflo_conv_ker_size IFLO_CONV_KER_SIZE] [--iflo_dropout_rate IFLO_DROPOUT_RATE]
               [--iflo_exclude_borders IFLO_EXCLUDE_BORDERS] [--clim_oggm_update_freq CLIM_OGGM_UPDATE_FREQ] [--smb_oggm_file SMB_OGGM_FILE]
               [--clim_oggm_clim_trend_array CLIM_OGGM_CLIM_TREND_ARRAY] [--clim_oggm_ref_period CLIM_OGGM_REF_PERIOD] [--clim_oggm_seed_par CLIM_OGGM_SEED_PAR]
               [--time_start TIME_START] [--time_end TIME_END] [--time_save TIME_SAVE] [--time_cfl TIME_CFL] [--time_step_max TIME_STEP_MAX]
               [--avalanche_update_freq AVALANCHE_UPDATE_FREQ] [--avalanche_angleOfRepose AVALANCHE_ANGLEOFREPOSE] [--gflex_update_freq GFLEX_UPDATE_FREQ]
               [--gflex_default_Te GFLEX_DEFAULT_TE] [--glerosion_cst GLEROSION_CST] [--glerosion_exp GLEROSION_EXP] [--glerosion_update_freq GLEROSION_UPDATE_FREQ]
               [--smb_oggm_update_freq SMB_OGGM_UPDATE_FREQ] [--smb_oggm_ice_density SMB_OGGM_ICE_DENSITY] [--smb_oggm_wat_density SMB_OGGM_WAT_DENSITY]
               [--part_tracking_method PART_TRACKING_METHOD] [--part_frequency_seeding PART_FREQUENCY_SEEDING] [--part_density_seeding PART_DENSITY_SEEDING]
               [--enth_water_density ENTH_WATER_DENSITY] [--enth_spy ENTH_SPY] [--enth_ki ENTH_KI] [--enth_ci ENTH_CI] [--enth_Lh ENTH_LH] [--enth_KtdivKc ENTH_KTDIVKC]
               [--enth_claus_clape ENTH_CLAUS_CLAPE] [--enth_melt_temp ENTH_MELT_TEMP] [--enth_ref_temp ENTH_REF_TEMP] [--enth_till_friction_angle ENTH_TILL_FRICTION_ANGLE]
               [--enth_uthreshold ENTH_UTHRESHOLD] [--enth_drain_rate ENTH_DRAIN_RATE] [--enth_till_wat_max ENTH_TILL_WAT_MAX] [--enth_drain_ice_column ENTH_DRAIN_ICE_COLUMN]
               [--enth_default_bheatflx ENTH_DEFAULT_BHEATFLX] [--rock_flow_speed ROCK_FLOW_SPEED] [--wts_output_file WTS_OUTPUT_FILE]
               [--wpar_add_topography WPAR_ADD_TOPOGRAPHY] [--plt2d_editor PLT2D_EDITOR] [--plt2d_live PLT2D_LIVE] [--plt2d_particles PLT2D_PARTICLES] [--plt2d_var PLT2D_VAR]
               [--plt2d_var_max PLT2D_VAR_MAX] [--wncd_output_file WNCD_OUTPUT_FILE] [--wncd_vars_to_save WNCD_VARS_TO_SAVE] [--wtif_vars_to_save WTIF_VARS_TO_SAVE]

```
# Arguments

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
||`--lncd_input_file`|`input.nc`|NetCDF input data file|
||`--lncd_coarsen`|`1`|Coarsen the data from NetCDF file by a certain (integer) number: 2 would be twice coarser ignore data each 2 grid points|
||`--lncd_crop`|`False`|Crop the data from NetCDF file with given top/down/left/right bounds|
||`--lncd_xmin`|`-100000000000000000000`|X left coordinate for cropping the NetCDF data|
||`--lncd_xmax`|`100000000000000000000`|X right coordinate for cropping the NetCDF data|
||`--lncd_ymin`|`-100000000000000000000`|Y bottom coordinate fro cropping the NetCDF data|
||`--lncd_ymax`|`100000000000000000000`|Y top coordinate for cropping the NetCDF data|
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
||`--ltif_coarsen`|`1`|coarsen the data to a coarser resolution (default: 1), e.g. 2 would be twice coarser ignore data each 2 grid points|
||`--ltif_crop`|`False`|Crop the data with xmin, xmax, ymin, ymax (default: False)|
||`--ltif_xmin`|`-100000000000000000000`|crop_xmin|
||`--ltif_xmax`|`100000000000000000000`|crop_xmax|
||`--ltif_ymin`|`-100000000000000000000`|crop_ymin|
||`--ltif_ymax`|`100000000000000000000`|crop_ymax|
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
||`--oggm_RGI_ID`|`RGI60-11.01450`|RGI ID|
||`--oggm_preprocess`||Use preprocessing|
||`--oggm_dx`|`100`|Spatial resolution (need preprocess false to change it)|
||`--oggm_border`|`30`|Safe border margin  (need preprocess false to change it)|
||`--oggm_thk_source`|`consensus_ice_thickness`|millan_ice_thickness or consensus_ice_thickness|
||`--oggm_vel_source`|`millan_ice_velocity`|Source of the surface velocities (millan_ice_velocity or its_live)|
||`--oggm_incl_glathida`||Make observation file (for IGM inverse)|
||`--oggm_path_glathida`|``|Path where the Glathida Folder is store, so that you don't need               to redownload it at any use of the script, if empty it will be in the home directory|
||`--oggm_save_in_ncdf`||Write prepared data into a geology file|
||`--smb_simple_update_freq`|`1`|Update the mass balance each X years (1)|
||`--smb_simple_file`|`smb_simple_param.txt`|Name of the imput file for the simple mass balance model (time, gradabl, gradacc, ela, accmax)|
||`--smb_simple_array`|`[]`|Time dependent parameters for simple mass balance model (time, gradabl, gradacc, ela, accmax)|
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
||`--clim_oggm_update_freq`|`1`|Update the climate each X years|
||`--smb_oggm_file`|`smb_oggm_param.txt`|Name of the imput file for the climate outide the given datatime frame (time, delta_temp, prec_scali)|
||`--clim_oggm_clim_trend_array`|`[['time', 'delta_temp', 'prec_scal'], [1900, 0.0, 1.0], [2020, 0.0, 1.0]]`|Define climate trend outside available time window|
||`--clim_oggm_ref_period`|`[1960, 1990]`|Define the reference period to pick year outside available time window|
||`--clim_oggm_seed_par`|`123`|Seeding parameter to fix for pickying randomly yer in the ref period|
||`--time_start`|`2000.0`|Start modelling time|
||`--time_end`|`2100.0`|End modelling time|
||`--time_save`|`10`|Save result frequency for many modules (in year)|
||`--time_cfl`|`0.3`|CFL number for the stability of the mass conservation scheme, it must be below 1|
||`--time_step_max`|`1.0`|Maximum time step allowed, used only with slow ice|
||`--avalanche_update_freq`|`1`|Update avalanche each X years (1)|
||`--avalanche_angleOfRepose`|`30`|Angle of repose (30°)|
||`--gflex_update_freq`|`100.0`|Update gflex each X years (1)|
||`--gflex_default_Te`|`50000`|Default value for Te (Elastic thickness [m]) if not given as ncdf file|
||`--glerosion_cst`|`2.7e-07`|Erosion multiplicative factor, here taken from Herman, F. et al.               Erosion by an Alpine glacier. Science 350, 193–195 (2015)|
||`--glerosion_exp`|`2`|Erosion exponent factor, here taken from Herman, F. et al.                Erosion by an Alpine glacier. Science 350, 193–195 (2015)|
||`--glerosion_update_freq`|`1`|Update the erosion only each X years (Default: 100)|
||`--smb_oggm_update_freq`|`1`|Update the mass balance each X years |
||`--smb_oggm_ice_density`|`910.0`|Density of ice for conversion of SMB into ice equivalent|
||`--smb_oggm_wat_density`|`1000.0`|Density of water|
||`--part_tracking_method`|`simple`|Method for tracking particles (simple or 3d)|
||`--part_frequency_seeding`|`50`|Frequency of seeding (unit : year)|
||`--part_density_seeding`|`0.2`|Density of seeding (1 means we seed all pixels, 0.2 means we seed each 5 grid cell, ect.)|
||`--enth_water_density`|`1000`|Constant of the Water density [kg m-3]|
||`--enth_spy`|`31556926`|Number of seconds per years [s y-1]|
||`--enth_ki`|`2.1`|Conductivity of cold ice [W m-1 K-1] (Aschwanden and al, JOG, 2012)|
||`--enth_ci`|`2009`|Specific heat capacity of ice [W s kg-1 K-1] (Aschwanden and al, JOG, 2012)|
||`--enth_Lh`|`334000.0`|latent heat of fusion [W s kg-1] = [E] (Aschwanden and al, JOG, 2012)|
||`--enth_KtdivKc`|`0.1`|Ratio of temp vs cold ice diffusivity Kt / Kc [no unit] (Aschwanden and al, JOG, 2012)|
||`--enth_claus_clape`|`7.9e-08`|Clausius-Clapeyron constant [K Pa-1] (Aschwanden and al, JOG, 2012)|
||`--enth_melt_temp`|`273.15`|Melting point at standart pressure [K] (Aschwanden and al, JOG, 2012)|
||`--enth_ref_temp`|`223.15`|Reference temperature [K] (Aschwanden and al, JOG, 2012)|
||`--enth_till_friction_angle`|`30`|Till friction angle in the Mohr-Coulomb friction law [deg]|
||`--enth_uthreshold`|`100`|uthreshold [m/y]|
||`--enth_drain_rate`|`0.001`|Drain rate at 1 mm/y  [m y-1] (Bueler and Pelt, GMD, 2015)|
||`--enth_till_wat_max`|`2`|Maximum water till tickness [m] (Bueler and Pelt, GMD, 2015)|
||`--enth_drain_ice_column`||Transform the water content beyond a thr=0.01 into water, drain it, and add it to basal melt rate|
||`--enth_default_bheatflx`|`0.065`|Geothermal heat flux [W m-2]|
||`--rock_flow_speed`|`1`|Speed of rock flow along the slope in m/y|
||`--wts_output_file`|`output_ts.nc`|Output ncdf data file (time serie)|
||`--wpar_add_topography`||Add topg|
||`--plt2d_editor`|`vs`|Optimized for VS code (vs) or spyder (sp) for live plot|
||`--plt2d_live`||Display plots live the results during computation instead of making png|
||`--plt2d_particles`||Display particles is True, does not display if False|
||`--plt2d_var`|`velbar_mag`|Name of the variable to plot|
||`--plt2d_var_max`|`250`|Maximum value of the varplot variable used to adjust the scaling of the colorbar|
||`--wncd_output_file`|`output.nc`|Output ncdf data file|
||`--wncd_vars_to_save`|`['topg', 'usurf', 'thk', 'smb', 'velbar_mag', 'velsurf_mag', 'uvelsurf', 'vvelsurf', 'wvelsurf']`|List of variables to be recorded in the ncdf file|
||`--wtif_vars_to_save`|`['usurf', 'thk']`|List of variables to be recorded in the NetCDF file|
