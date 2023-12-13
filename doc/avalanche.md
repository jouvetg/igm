### <h1 align="center" id="title">IGM avalanche module  </h1>

# Description:

This IGM module permits to model redistribution of snow due to avalanches.
This routine move ice/snow downslope until the ice surface is everywhere
at angle of repose. This function was adapted from 
[Mark Kessler's GC2D](https://github.com/csdms-contrib/gc2d)
program and implemented in IGM by Jürgen Mey with support from Guillaume Jouvet.
 
 
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
||`--avalanche_update_freq`|`1`|Update avalanche each X years (1)|
||`--avalanche_angleOfRepose`|`30`|Angle of repose (30°)|
