### <h1 align="center" id="title">IGM module `write_particles` </h1>

# Description:

This IGM module writes particle time-position in csv files computed by module `particles`. The saving frequency is given by parameter `time_save` defined in module `time`.

The data are stored in folder 'trajectory' (created if does not exist). Files 'traj-TIME.csv' reports the space-time position of the particles at time TIME with the following structure:

```
ID,  state.xpos,  state.ypos,  state.zpos, state.rhpos,  state.tpos, state.englt
X,            X,           X,           X,           X,           X,           X,
X,            X,           X,           X,           X,           X,           X,
X,            X,           X,           X,           X,           X,           X,
```

providing in turn the particle ID, x,y,z positions, the relative height within the ice column, the seeding time, and the englacial time.
 
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
||`--wpar_add_topography`||Add topg|
