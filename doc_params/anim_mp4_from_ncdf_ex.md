### <h1 align="center" id="title">IGM module anim_mp4_from_ncdf_ex </h1>

# Description:

This IGM module makes an animated mp4 video of ice thickness overt tim from 
the netcdf ex.nc file produced by module write_ncdf_ex. It only implements the
'final' function, the rest ('init', 'update') doing nothing.

# I/O:

Takes  : ncdf file ex.nc of transient 2d gridded data
Return : mp4 animation of the ice thickness over time


|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
 
# Parameters: 
