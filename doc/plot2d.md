### <h1 align="center" id="title">IGM module `plot2d` </h1>

# Description:

This IGM module produces 2D plan-view plots of variable defined by parameter `plt2d_var` (e.g. `plt2d_var` can be set to `thk`, or `ubar`, ...). The saving frequency is given by parameter `time_save` defined in module `time`.  The scale range of the colobar is controlled by parameter `plt2d_varmax`.

By default, the plots are saved as png files in the working directory. However, one may display the plot "in live" by setting `plt2d_live` to True. Note that if you use the spyder python editor, you need to turn `plt2d_editor` to 'sp'.
 
If the `particles` module is activated, one may plot particles on the top setting `plt2d_particles` to True, or remove them form the plot seeting it to False.
 
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
||`--plt2d_editor`|`vs`|Optimized for VS code (vs) or spyder (sp) for live plot|
||`--plt2d_live`||Display plots live the results during computation instead of making png|
||`--plt2d_particles`||Display particles is True, does not display if False|
||`--plt2d_var`|`velbar_mag`|Name of the variable to plot|
||`--plt2d_var_max`|`250`|Maximum value of the varplot variable used to adjust the scaling of the colorbar|
