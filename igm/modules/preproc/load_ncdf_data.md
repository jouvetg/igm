### <h1 align="center" id="title">IGM module load_ncdf_data </h1>

# Description:

This IGM module loads spatial raster data from a netcdf file (geology.nc) and
transform the fields into tensorflow variables. It also complete the data,
e.g. ge the basal topography from ice thickness and surface topography.
(there is no update function defined). It contains the two functions for
resampling and cropping the data.

# I/O:

Input: geology.nc
Output: variables contained inside as tensorflow objects
