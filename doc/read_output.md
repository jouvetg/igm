### <h1 align="center" id="title">IGM read_output  </h1>

# Description:

This IGM module permits to read an output NetCDF file produced previously and to run igm as if these quantities 
were shortly computed, this is mainly usefull for testing postprocessing module independently.
 
# Parameters: 


|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--rncd_input_file`|`output.nc`|NetCDF input data file|
||`--rncd_crop`|`False`|Crop the data from NetCDF file with given top/down/left/right bounds|
||`--rncd_xmin`|`-100000000000000000000`|X left coordinate for cropping the NetCDF data|
||`--rncd_xmax`|`100000000000000000000`|X right coordinate for cropping the NetCDF data|
||`--rncd_ymin`|`-100000000000000000000`|Y bottom coordinate fro cropping the NetCDF data|
||`--rncd_ymax`|`100000000000000000000`|Y top coordinate for cropping the NetCDF data|
