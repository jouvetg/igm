### <h1 align="center" id="title">IGM module `print_comp` </h1>

# Description:

This module reports the computational times taken by any IGM modules at the end of the model run directly in the terminal output, as well as in a file ("computational-statistics.txt"). It also produces a camember-like plot ( "computational-pie.png") displaying the relative importance of each module, computationally-wise. 

Note: These numbers must be interepreted with care: Leaks of computational times from one to another module are sometime observed (likely) due to asynchronous GPU calculations.
 
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
