### <h1 align="center" id="title">IGM module `load_ncdf_data` </h1>

# Description:

This IGM module loads spatial 2D raster data from a NetCDF file (parameter `input_file`, default: input.nc) and transform all existing 2D fields into tensorflow variables. It is expected here to import at least basal topography (variable `topg`). It also complete the data, e.g. the basal topography from ice thickness and surface topography. However, any other field present in NetCDF file will be passed as tensorflow variables, and will therefore be available in the code through `state.myvar` (e.g. variable `icemask` can be provided, and served to define an accumulation area -- this is usefull for modelling an individual glaciers, and prevent overflowing in neighbouring catchements). The module also contains the two functions for resampling (parameter `coarsen_ncdf` should be increased to 2,3,4 ..., default 1 value means no coarsening) and cropping the data (parameter `crop_ncdf` should be set to True, and the bounds must be definined as wished).

It is possible to restart an IGM run by reading data in an nNetCDF file obtained as a previous IGM run. To that aim, one needs to provide the NETcdf output file as input to IGM. IGM will look for the data that corresponds to the starting time `params.time_start`, and then intialize it with this time.

This module depends on `netCDF4`.
 
# Parameters: 


|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--input_file`|`input.nc`|NetCDF input data file|
||`--coarsen_ncdf`|`1`|Coarsen the data from NetCDF file by a certain (integer) number: 2 would be twice coarser ignore data each 2 grid points|
||`--crop_ncdf`|`False`|Crop the data from NetCDF file with given top/down/left/right bounds|
||`--crop_ncdf_xmin`|`None`|X left coordinate for cropping the NetCDF data|
||`--crop_ncdf_xmax`|`None`|X right coordinate for cropping the NetCDF data|
||`--crop_ncdf_ymin`|`None`|Y bottom coordinate fro cropping the NetCDF data|
||`--crop_ncdf_ymax`|`None`|Y top coordinate for cropping the NetCDF data|
