### <h1 align="center" id="title">IGM module `load_ncdf` </h1>

# Description:

This IGM module loads spatial 2D raster data from a NetCDF file (parameter `lncd_input_file`, default: input.nc) and transform all existing 2D fields into tensorflow variables. It is expected here to import at least basal topography (variable `topg`). It also complete the data, e.g. the basal topography from ice thickness and surface topography. However, any other field present in NetCDF file will be passed as tensorflow variables, and will therefore be available in the code through `state.myvar` (e.g. variable `icemask` can be provided, and served to define an accumulation area -- this is usefull for modelling an individual glaciers, and prevent overflowing in neighbouring catchements). The module also contains the two functions for resampling (parameter `lncd_coarsen` should be increased to 2,3,4 ..., default 1 value means no coarsening) and cropping the data (parameter `lncd_crop` should be set to True, and the bounds must be definined as wished).

It is possible to restart an IGM run by reading data in an nNetCDF file obtained as a previous IGM run. To that aim, one needs to provide the NETcdf output file as input to IGM. IGM will look for the data that corresponds to the starting time `params.time_start`, and then intialize it with this time.

This module depends on `netCDF4`.
