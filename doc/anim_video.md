### <h1 align="center" id="title">IGM module `anim_video`` </h1>

# Description:

This IGM module makes an animated mp4 video of ice thickness over time from 
the NetCDF output (default output.nc) file produced by module `write_ncdf`. It only implements the 'finalize' function, the rest ('initialize', 'update') doing nothing.

This module depends on `xarray` library.
 
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
