### <h1 align="center" id="title">IGM module write_ncdf_ex </h1>

# Description:

This IGM modules write 2D field variables defined in the list 
params.vars_to_save_ncdf_ex into the ncdf output file output.nc

The module takes variables defined in params.vars_to_save_ncdf_ex
as input and return output.nc
 
# Parameters: 


|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--vars_to_save_ncdf_ex`|`['topg', 'usurf', 'thk', 'smb', 'velbar_mag', 'velsurf_mag', 'uvelsurf', 'vvelsurf', 'wvelsurf']`|List of variables to be recorded in the ncdf file|
