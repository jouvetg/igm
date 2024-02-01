
### <h1 align="center" id="title">IGM module `rockflow` </h1>

# Description:

This module extends the ice flow outside the glaciated area, by giving a constant speed and along-slope flow direction. This modules serves to track rock-like particles (with module `particles`) everywhere in ice-free and ice-filled areas, particles being either advected at constant steep (controlled by parameter `rock_flow_speed`) following the stepest gradient of the ice-free terrain in 2D, or by ice flow in 3D.
 
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
||`--rock_flow_speed`|`1`|Speed of rock flow along the slope in m/y|
