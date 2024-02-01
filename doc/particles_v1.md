
### <h1 align="center" id="title">IGM module `particle_v1` </h1>

# Description:

This IGM module implments the former particle tracking routine
associated with iceflow_v1 (check at the doc of particles).
 
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
||`--part_tracking_method`|`3d`|Method for tracking particles (3d or simple)|
||`--part_frequency_seeding`|`10`|Frequency of seeding (default: 10)|
||`--part_density_seeding`|`0.2`|Density of seeding (default: 0.2)|
