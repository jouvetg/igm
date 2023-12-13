### <h1 align="center" id="title">IGM module `print_info` </h1>

# Description:

This IGM modules prints in the terminal basic live-time modelling informations such that the iteration number, time, time step, and ice volume (km^3).
 
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
