### <h1 align="center" id="title">IGM module `write_ncdf_ex` </h1>

# Description:

This IGM module writes 2D field variables defined in the paramer list `vars_to_save_ncdf_ex` into the NetCDF output file given by parameter `output_file_ncdf_ex` (default output.nc). The saving frequency is given by parameter `time_save` defined in module `time_step`.

 
# Parameters: 


|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--output_file_ncdf_ex`|`output.nc`|Output ncdf data file|
||`--vars_to_save_ncdf_ex`|`['topg', 'usurf', 'thk', 'smb', 'velbar_mag', 'velsurf_mag', 'uvelsurf', 'vvelsurf', 'wvelsurf']`|List of variables to be recorded in the ncdf file|
