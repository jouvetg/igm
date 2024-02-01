### <h1 align="center" id="title">IGM isostasy_gfex module  </h1>

# Description:

This IGM module permits to model the isostasy or upward motion of the 
lithosphere when loaded with thick ice, it uses the 
[gflex](https://gmd.copernicus.org/articles/9/997/2016/)
python module writen by Andy Wickert.

This function was implemented in IGM by Jürgen Mey.
 
Parameters are the update frequency `gflex_update_freq` and the Elastic thickness [m] (Te) `gflex_default_Te`.

This module only runs on CPU, which may be an issue for treating very large arrays.
On the other hand, we do not expect a frequent update, therefore, this module should not be overall too consuming.



 
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
||`--gflex_update_freq`|`100.0`|Update gflex each X years (1)|
||`--gflex_default_Te`|`50000`|Default value for Te (Elastic thickness [m]) if not given as ncdf file|
