### <h1 align="center" id="title">IGM module `write_ncdf` </h1>

# Description:

This IGM module writes 2D field variables defined in the paramer list `wncd_vars_to_save` into the NetCDF output file given by parameter `wncd_output_file` (default output.nc). The saving frequency is given by parameter `time_save` defined in module `time`.

 
# Parameters: 


|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--wncd_output_file`|`output.nc`|Output ncdf data file|
||`--wncd_vars_to_save`|`['topg', 'usurf', 'thk', 'smb', 'velbar_mag', 'velsurf_mag', 'uvelsurf', 'vvelsurf', 'wvelsurf']`|List of variables to be recorded in the ncdf file|
