### <h1 align="center" id="title">IGM module load_ncdf_data </h1>

# Description:

This IGM module loads spatial raster data from a netcdf file (default: geology.nc) and transform the fields into tensorflow variables. It also complete the data, e.g. the basal topography from ice thickness and surface topography. It contains the two functions for resampling and cropping the data.

The module takes an input ncdf file and return variables contained inside as tensorflow objects
