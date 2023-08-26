### <h1 align="center" id="title">IGM module load_ncdf_data </h1>

# Description:

This IGM module loads spatial raster data from a netcdf file (default: geology.nc) and transform the fields into tensorflow variables. It also complete the data, e.g. the basal topography from ice thickness and surface topography. It contains the two functions for resampling and cropping the data.

The module takes an input ncdf file and return variables contained inside as tensorflow objects
 
# Parameters: 


|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--input_file`|`input.nc`|Input data file|
||`--resample`|`1`|Resample the data to a coarser resolution (default: 1), e.g. 2 would be twice coarser ignore data each 2 grid points|
||`--crop_data`|`False`|Crop the data with xmin, xmax, ymin, ymax (default: False)|
||`--crop_xmin`|`None`|X left coordinate for cropping|
||`--crop_xmax`|`None`|X right coordinate for cropping|
||`--crop_ymin`|`None`|Y bottom coordinate fro cropping|
||`--crop_ymax`|`None`|Y top coordinate for cropping|
