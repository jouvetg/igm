### <h1 align="center" id="title">IGM module write_ncdf_ts </h1>

# Description:

This IGM module write time serie variables (ice glaciated area and volume)
into the ncdf output file ts.nc

The module takes IG variables (state.thk, state.dx,  ...) and produce a ts.nc file.
 
# Parameters: 


|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--output_file_ncdf_ex`|`output.nc`|Output ncdf data file|
||`--vars_to_save_ncdf_ex`|`['topg', 'usurf', 'thk', 'smb', 'velbar_mag', 'velsurf_mag', 'uvelsurf', 'vvelsurf', 'wvelsurf']`|List of variables to be recorded in the ncdf file|
