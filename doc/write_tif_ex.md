### <h1 align="center" id="title">IGM module `write_tif_ex` </h1>

# Description:

This IGM module writes 2D field variables defined in the paramer list `vars_to_save_tif_ex` into tif output files. Files will be created with names composed by the variable name and the time (e.g., thk-000040.tif, usurf-000090.tif) in the working directory. The saving frequency is given by parameter `time_save` defined in module `time_step`. If input file were call with module `load_tif_data`, then the tif meta information are saved, and provided with the final tiff files.

This module depends on the `rasterio` library.



  
# Parameters: 


|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--vars_to_save_tif_ex`|`['usurf', 'thk']`|List of variables to be recorded in the NetCDF file|
