#!/usr/bin/env python3

"""
This utility script is used to update the names of the modules and parameters of igm
after a major update of the code done on Oct 24, 2023.

Procedure:

- Go to the folder where you have your igm parameter file and user-defined modules
- Run the script with the name of the parameter file and the name of the modules as arguments:

        python patch_reform_names.py params.json 
        python patch_reform_names.py mymodule.py 
        
- Both will produce a copy of the former (params_old.json, mymodule_old.py) and the new 
(reformed) ones (params.json, mymodule.py), check the code run with the new files. If yes, 
you can delete the old files. If not, you must investigate the differences between the old,
possibly correct manually. You can also look manually the correspondance table below between
alod and new parameters and modules names.
"""

import os, shutil, glob, re, sys, json

# dictionnary listing the changes of the modules names from old to new
module_changes = {
    "oggm_data_prep": "oggm_shop",
    "load_ncdf_data": "load_ncdf",
    "load_tif_data": "load_tif",
    "time_step": "time",
    "vertical_iceflow": "vert_flow",
    "topg_glacial_erosion": "glerosion",
    "write_tif_ex": "write_tif",
    "write_ncdf_ex": "write_ncdf",
    "write_ncdf_ts": "write_ts",
    "print_all_comp_info": "print_comp",
    "write_plot2d": "plot2d",
    "anim3d_from_ncdf_ex": "anim_mayavi",
    "anim_plotly_from_ncdf_ex": "anim_plotly",
    "anim_mp4_from_ncdf_ex": "anim_video",
}

# dictionnary listing the changes of the parameters names from old to new
param_changes = {
    "input_file": "lncd_input_file",
    "coarsen_ncdf": "lncd_coarsen",
    "crop_ncdf_xmin": "lncd_xmin",
    "crop_ncdf_xmax": "lncd_xmax",
    "crop_ncdf_ymin": "lncd_ymin",
    "crop_ncdf_ymax": "lncd_ymax",
    "crop_ncdf": "lncd_crop",
    "coarsen_tif": "ltif_coarsen",
    "crop_tif_xmin": "ltif_xmin",
    "crop_tif_xmax": "ltif_xmax",
    "crop_tif_ymin": "ltif_ymin",
    "crop_tif_ymax": "ltif_ymax",
    "crop_tif": "ltif_crop",
    "RGI_ID": "oggm_RGI_ID",
    "preprocess": "oggm_preprocess",
    "dx": "oggm_dx",
    "border": "oggm_border",
    "thk_source": "oggm_thk_source",
    "vel_source": "oggm_vel_source",
    "include_glathida": "oggm_include_glathida",
    "path_glathida": "oggm_path_glathida",
    "save_input_ncdf": "oggm_save_in_ncdf",
    "geology_optimized_file": "opti_save_result_in_ncdf",
    "plot2d_live_inversion": "opti_plot2d_live",
    "plot2d_inversion": "opti_plot2d",
    "write_ncdf_optimize": "opti_save_iterat_in_ncdf",
    "editor_plot2d_optimize": "opti_editor_plot2d",
    "tracking_method": "part_tracking_method",
    "frequency_seeding": "part_frequency_seeding",
    "density_seeding": "part_density_seeding",
    "speed_rockflow": "rock_flow_speed",
    "smb_update_freq": "smb_simple_update_freq",
    "cfl": "time_cfl",
    "erosion_cst": "glerosion_cst",
    "erosion_exp": "glerosion_exp",
    "erosion_update_freq": "glerosion_update_freq",
    "type_iceflow": "iflo_type",
    "emulator": "iflo_emulator",
    "init_slidingco": "iflo_init_slidingco",
    "init_arrhenius": "iflo_init_arrhenius",
    "regu_glen": "iflo_regu_glen",
    "regu_weertman": "iflo_regu_weertman",
    "exp_glen": "iflo_exp_glen",
    "exp_weertman": "iflo_exp_weertman",
    "gravity_cst": "iflo_gravity_cst",
    "ice_density": "iflo_ice_density",
    "new_friction_param": "iflo_new_friction_param",
    "Nz": "iflo_Nz",
    "vert_spacing": "iflo_vert_spacing",
    "thr_ice_thk": "iflo_thr_ice_thk",
    "solve_iceflow_step_size": "iflo_solve_step_size",
    "solve_iceflow_nbitmax": "iflo_solve_nbitmax",
    "stop_if_no_decrease": "iflo_solve_stop_if_no_decrease",
    "fieldin": "iflo_fieldin",
    "dim_arrhenius": "iflo_dim_arrhenius",
    "retrain_iceflow_emulator_freq": "iflo_retrain_emulator_freq",
    "retrain_iceflow_emulator_lr": "iflo_retrain_emulator_lr",
    "retrain_iceflow_emulator_nbit_init": "iflo_retrain_emulator_nbit_init",
    "retrain_iceflow_emulator_nbit": "iflo_retrain_emulator_nbit",
    "retrain_iceflow_emulator_framesizemax": "iflo_retrain_emulator_framesizemax",
    "multiple_window_size": "iflo_multiple_window_size",
    "force_max_velbar": "iflo_force_max_velbar",
    "network": "iflo_network",
    "activation": "iflo_activation",
    "nb_layers": "iflo_nb_layers",
    "nb_blocks": "iflo_nb_blocks",
    "nb_out_filter": "iflo_nb_out_filter",
    "conv_ker_size": "iflo_conv_ker_size",
    "dropout_rate": "iflo_dropout_rate",
    "exclude_borders_from_iceflow": "iflo_exclude_borders",
    "spy": "enth_spy",
    "ki": "enth_ki",
    "ci": "enth_ci",
    "KtdivKc": "enth_KtdivKc",
    "claus_clape_cst": "enth_claus_clape",
    "melt_temp": "enth_melt_temp",
    "ref_temp": "enth_ref_temp",
    "till_friction_angle": "enth_till_friction_angle",
    "uthreshold": "enth_uthreshold",
    "drain_rate": "enth_drain_rate",
    "till_wat_max": "enth_till_wat_max",
    "drain_ice_column": "enth_drain_ice_column",
    "default_bheatflx": "enth_default_bheatflx",
    "Lh": "enth_Lh",
    "water_density": "enth_water_density",
    "mb_update_freq": "smb_accpdd_update_freq",
    "refreeze_factor": "smb_accpdd_refreeze_factor",
    "thr_temp_snow": "smb_accpdd_thr_temp_snow",
    "thr_temp_rain": "smb_accpdd_thr_temp_rain",
    "melt_factor_snow": "smb_accpdd_melt_factor_snow",
    "melt_factor_ice": "smb_accpdd_melt_factor_ice",
    "shift_hydro_year": "smb_accpdd_shift_hydro_year",
    "output_file_ncdf_ex": "wncd_output_file",
    "vars_to_save_ncdf_ex": "wncd_vars_to_save",
    "vars_to_save_tif_ex": "wtif_vars_to_save",
    "output_file_ncdf_ts": "wts_output_file",
    "add_topography_to_particles": "wpar_add_topography",
    "editor_plot2d": "plt2d_editor",
    "plot_live": "plt2d_live",
    "plot_particles": "plt2d_particles",
    "varplot": "plt2d_var",
    "varplot_max": "plt2d_varmax",
}

l = sys.argv[1]
filename = l.split(".")[0]
ext = l.split(".")[-1]

if os.path.exists(filename + "_old." + ext):
    print(
        "the Backup file "
        + filename
        + "_old."
        + ext
        + " already exists, please remove it before running this script"
    )

else:
    shutil.copy(l, filename + "_old." + ext)

    with open(l, "r") as input_file:
        input_text = input_file.read()

    for old_word, new_word in param_changes.items():
        if ext == "py":
            input_text = re.sub(
                r"\b%s\b" % "params." + old_word, "params." + new_word, input_text
            )
            input_text = input_text.replace("--" + old_word, "--" + new_word)
        else:
            input_text = re.sub(r"\b%s\b" % old_word, new_word, input_text)

    for old_word, new_word in module_changes.items():
        input_text = re.sub(r"\b%s\b" % old_word, new_word, input_text)

    with open(l, "w") as output_file:
        output_file.write(input_text)
