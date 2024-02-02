### <h1 align="center" id="title">IGM module `write_ts` </h1>

# Description:

This IGM module writes time serie variables (ice glaciated area and volume) into the NetCDF output file defined by parameter `wts_output_file` (default output_ts.nc). The saving frequency is given by parameter `time_save` defined in module `time`.
 
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
||`--wts_output_file`|`output_ts.nc`|Output ncdf data file (time serie)|
