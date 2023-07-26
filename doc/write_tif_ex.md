### <h1 align="center" id="title">IGM module write_tif_ex </h1>

# Description:

This IGM module saves state variables in tiff file at a given frequency.
Variables to be saved are provided as list in parameter vars_to_save.
Files will be created with names like thk-000040.tif in the working directory.

# I/O:

Input: variables to be saved
Output: tiff files
 
# Parameters: 


|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--vars_to_save`|`['usurf', 'thk']`|List of variables to be recorded in the ncdf file|
